import argparse
import csv
import json
import importlib
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SP = importlib.import_module("analysis.recoverability.06_synthetic_power")
L, EFF, splits, C = SP.L, SP.EFF, SP.splits, SP.C

SIMS = (("archer_yeast", "multinomial", "n50"),
        ("gse139131_unsel", "dm", "n50"),
        ("gse174329_eif3b", "dm", ""))
BIN = 10
SIM_DEPTHS = (5, 10, 20, 50, 100, 200, 500)
ANCHORS = (5, 10, 20, 50, 100, 200, 500, 2000, 10000, 50000)
GRID = np.unique(np.round(np.concatenate([np.geomspace(5, 50000, 81), ANCHORS]), 6))
Z_CI = stats.norm.ppf(0.975)
Z_NULL = stats.norm.ppf(0.95)
POWER_OK = 0.8
GROUP_RATIO = 1.12
OUT = "theory"
LAM = 1.0 / EFF.PRIOR_SD ** 2


def prep(key, noise):
    SP._sim_init(key, BIN, noise, "all")
    S = dict(SP._S)
    ft, c_all = S["ft"], S["depth_real"].astype(float)
    fold = np.full(S["n"], -1)
    for k in S["folds_used"]:
        _, te = splits.train_test(ft.chrom, S["species"], k)
        fold[te] = k
    tx = np.nonzero((c_all > 0) & (fold >= 0))[0]
    kappa = float(S["alpha"]) if noise == "dm" else np.inf
    return {"key": key, "noise": noise, "kappa": kappa, "ft": ft, "eta_pos": S["eta_pos"],
            "tx": tx, "c": c_all[tx], "fold": fold[tx], "K": len(S["folds_used"]),
            "med": float(np.median(c_all)), "T_panel": S["n"]}


def truth_eta(D, truth, size):
    comps = SP.planted_components(D["ft"], size)
    eta, coefs = D["eta_pos"].copy(), {}
    for fam in SP.PLANTED[truth]:
        eta += comps[fam][0]
        coefs.update(comps[fam][1])
    return eta, coefs


def flat_p(ft, eta):
    T = len(ft.tx_start) - 1
    return np.exp(eta - L._seg_logsumexp(eta, ft.seg, T)[ft.seg])


def weights(D, depth):
    m = EFF.MIN_READS
    mu = D["c"] * depth / D["med"]
    w1 = mu * stats.poisson.sf(m - 2, mu)
    w2 = mu * mu * stats.poisson.sf(m - 3, mu) + w1
    k = D["kappa"]
    wF = w1 if not np.isfinite(k) else (w2 + k * w1) / (1 + k)
    return w1, w2, wF, w2 - w1 * w1, stats.poisson.sf(m - 1, mu)


def geometry(prob, beta, p=None):
    if p is None:
        logp = L.nucleotide_logp(prob, beta)
        p = np.exp(logp)
        lq = L.bin_logq(prob, logp)
    else:
        lq = np.log(np.maximum(np.bincount(prob.bin, weights=p, minlength=prob.n_bins), 1e-300))
    q = np.exp(lq)
    n = len(p)
    Xp = sp.diags(p) @ prob.X
    Sb = sp.csr_matrix((np.ones(n), (prob.bin, np.arange(n))), shape=(prob.n_bins, n))
    St = sp.csr_matrix((np.ones(n), (prob.seg, np.arange(n))), shape=(prob.T, n))
    xb = (Sb @ Xp).toarray() / q[:, None]
    xt = (St @ Xp).toarray()
    R = (xb - xt[prob.bin_tx]) * np.sqrt(q)[:, None]
    return p, q, lq, R, xb


def fisher_rows(prob, beta):
    _, _, lq, R, _ = geometry(prob, beta)
    return R, lq


def newton_fit(prob, yb, beta, max_iter=200, tol=1e-14):
    beta = np.array(beta, dtype=np.float64)
    N_t = np.bincount(prob.bin_tx, weights=yb, minlength=prob.T)
    N = N_t.sum()
    Xt = prob.X.T.tocsr()
    pos_y = yb > 0

    def obj(b):
        lq = L.bin_logq(prob, L.nucleotide_logp(prob, b))
        return (-(yb[pos_y] * lq[pos_y]).sum() + 0.5 * LAM * b @ b) / N

    f = obj(beta)
    for it in range(max_iter):
        p, q, _, R, _ = geometry(prob, beta)
        g = Xt @ (yb[prob.bin] / q[prob.bin] * p - N_t[prob.seg] * p) - LAM * beta
        I = R.T @ (R * N_t[prob.bin_tx, None])
        I[np.diag_indices_from(I)] += LAM
        step = np.linalg.solve(I, g)
        t = 1.0
        while True:
            fn = obj(beta + t * step)
            if fn <= f or t < 1e-10:
                break
            t *= 0.5
        if not fn <= f:
            break
        beta, change, f = beta + t * step, f - fn, fn
        if change < tol:
            break
    return beta


def model_terms(prob, beta, yb):
    Nt = np.bincount(prob.bin_tx, weights=yb, minlength=prob.T)
    qs = yb / np.maximum(Nt[prob.bin_tx], 1e-300)
    p, q, lq, R, xb = geometry(prob, beta)
    wb = Nt[prob.bin_tx]
    rq = qs / q
    A = R.T @ (R * wb[:, None])
    A += (prob.X.T @ sp.diags(wb[prob.bin] * (1 - rq[prob.bin]) * p) @ prob.X).toarray()
    A -= xb.T @ (xb * (wb * (q - qs))[:, None])
    m = np.zeros((prob.T, prob.K))
    np.add.at(m, prob.bin_tx, (qs / np.sqrt(q))[:, None] * R)
    return {"A": 0.5 * (A + A.T), "R": R, "q": q, "lq": lq, "qs": qs, "m": m,
            "sbar": m.T @ Nt, "bin_tx": prob.bin_tx}


def cross_cov(a, b, wF, vN):
    w = wF[a["bin_tx"]] * a["qs"] / np.sqrt(a["q"] * b["q"])
    C = a["R"].T @ (b["R"] * w[:, None]) - a["m"].T @ (b["m"] * (wF - vN)[:, None])
    return C


def cov_with_gain(a, wF, vN, dl, g):
    return a["R"].T @ (wF[a["bin_tx"]] * a["qs"] / np.sqrt(a["q"]) * dl) - a["m"].T @ ((wF - vN) * g)


def stack(a, b, wF, vN):
    ab = cross_cov(a, b, wF, vN)
    S = np.block([[cross_cov(a, a, wF, vN), ab], [ab.T, cross_cov(b, b, wF, vN)]])
    return 0.5 * (S + S.T)


_D = {}


def _init(key, noise):
    _D.clear()
    _D.update(prep(key, noise))


def cell(task):
    truth, size, fam = task
    D = _D
    t0 = time.time()
    ft, tx, K = D["ft"], D["tx"], D["K"]
    eta, coefs = truth_eta(D, truth, size)
    pall = flat_p(ft, eta)
    nT = len(ft.tx_start) - 1
    per = {a: {} for a in ANCHORS}
    for k in range(K):
        tr_, te_ = D["fold"] != k, D["fold"] == k
        P1 = L.Problem(ft, tx[tr_], BIN, SP.FAMILIES[fam])
        P0 = L.Problem(ft, tx[tr_], BIN, [])
        T1 = L.Problem(ft, tx[te_], BIN, SP.FAMILIES[fam])
        T0 = L.Problem(ft, tx[te_], BIN, [])
        g_ = P0.K
        b0 = b1 = None
        for a in sorted(ANCHORS, reverse=True):
            w1, w2, wF, vN, pinc = weights(D, a)

            def exp_counts(prob, sel):
                w = np.zeros(nT)
                w[tx[sel]] = w1[sel]
                return prob.bin_counts(pall * w[ft.seg])

            y1, y0 = exp_counts(P1, tr_), exp_counts(P0, tr_)
            t1, t0b = exp_counts(T1, te_), exp_counts(T0, te_)
            if b0 is None:
                b0 = newton_fit(P0, y0, np.zeros(P0.K))
                init = np.zeros(P1.K)
                init[:g_] = b0
                for blk, val in coefs.items():
                    if blk in P1.slices:
                        init[P1.slices[blk]] = val
                b1 = init
            else:
                b0 = newton_fit(P0, y0, b0)
            b1 = newton_fit(P1, y1, b1)
            assert P1.n_bins == P0.n_bins and T1.n_bins == T0.n_bins
            g1, g0 = model_terms(P1, b1, y1), model_terms(P0, b0, y0)
            SZ = stack(g1, g0, wF[tr_], vN[tr_])
            h1, h0 = model_terms(T1, b1, t1), model_terms(T0, b0, t0b)
            dl = h1["lq"] - h0["lq"]
            qs = h1["qs"]
            gt = np.bincount(T1.bin_tx, weights=qs * dl, minlength=T1.T)
            Vt = np.maximum(np.bincount(T1.bin_tx, weights=qs * dl * dl, minlength=T1.T) - gt * gt, 0)
            Bt = stack(h1, h0, wF[te_], vN[te_])
            Ct = np.concatenate([cov_with_gain(h1, wF[te_], vN[te_], dl, gt),
                                 cov_with_gain(h0, wF[te_], vN[te_], dl, gt)])
            K1 = P1.K
            H1 = np.linalg.inv(g1["A"] + LAM * np.eye(K1))
            H0 = np.linalg.inv(g0["A"] + LAM * np.eye(g_))
            Z1, Z0 = np.zeros((K1, g_)), np.zeros((g_, K1))
            W = np.block([[-0.5 * H1 @ h1["A"] @ H1, Z1], [Z0, 0.5 * H0 @ h0["A"] @ H0]])
            Mm = np.block([[H1, Z1], [Z0, -H0]])
            l = np.concatenate([H1 @ h1["sbar"], -H0 @ h0["sbar"]])
            WS = W @ SZ
            Vbil = float(np.sum((Mm.T @ Bt @ Mm) * SZ))
            wte1 = w1[te_]
            A1 = g1["A"]
            Ate = A1[g_:, g_:] - A1[g_:, :g_] @ np.linalg.solve(A1[:g_, :g_] + LAM * np.eye(g_), A1[:g_, g_:])
            per[a][k] = {
                "N": float(wte1.sum()), "n_tx": float(pinc[te_].sum()),
                "G": float(np.sum(wte1 * gt)),
                "ET": float(np.trace(WS)), "Vquad": float(2 * np.sum(WS * WS.T)), "Vbil": Vbil,
                "l": l, "Bt": Bt, "Ct": Ct,
                "Vnu": float(np.sum(wF[te_] * Vt) + np.sum(vN[te_] * gt ** 2)),
                "eps": Vbil / max(float(np.sum(wF[te_])), 1e-300),
                "g": gt, "V": Vt, "idx": np.nonzero(te_)[0],
                "df": float(np.trace(Ate @ np.linalg.inv(Ate + LAM * np.eye(len(Ate))))),
            }
    rows = []
    for a in ANCHORS:
        w1, w2, wF, vN, pinc = weights(D, a)
        F = per[a]
        N = sum(F[k]["N"] for k in F)
        if N <= 0:
            continue
        mu = sum(F[k]["G"] + F[k]["ET"] for k in F) / N
        comp = {"quad": sum(F[k]["Vquad"] for k in F), "bil": sum(F[k]["Vbil"] for k in F),
                "shift": 0.0, "cross": 0.0, "nu": 0.0}
        for f in F:
            Lf = sum(F[k]["l"] for k in F if k != f)
            comp["shift"] += Lf @ F[f]["Bt"] @ Lf
            comp["cross"] += 2 * Lf @ F[f]["Ct"]
            comp["nu"] += F[f]["Vnu"]
        var = sum(comp.values())
        g_t = np.zeros(len(tx))
        V_t = np.zeros(len(tx))
        e_t = np.zeros(len(tx))
        for k in F:
            g_t[F[k]["idx"]] = F[k]["g"]
            V_t[F[k]["idx"]] = F[k]["V"]
            e_t[F[k]["idx"]] = F[k]["eps"]
        gbar = np.sum(w1 * g_t) / N
        het = np.sum(w2 * (g_t - gbar) ** 2) / N ** 2
        se2 = np.sum(wF * (V_t + e_t)) / N ** 2 + het
        rows.append({"dataset": D["key"], "noise": D["noise"], "kappa": D["kappa"], "truth": truth,
                     "size_index": size, "family": fam, "depth": float(a), "K": K,
                     "N": N, "n_tx": sum(F[k]["n_tx"] for k in F), "mu": mu,
                     "sd": float(np.sqrt(max(var, 0.0)) / N), "se": float(np.sqrt(se2)),
                     "signal": float(gbar), "penalty": -sum(F[k]["ET"] for k in F) / N,
                     "Fbar": float(wF.sum() / w1.sum()), "df_eff": float(np.mean([F[k]["df"] for k in F])),
                     "het": float(het), "T_eff": float(w1.sum() ** 2 / w2.sum()),
                     **{"var_" + k: float(v) / N ** 2 for k, v in comp.items()}})
    print("[theory]   {} {} size {} {}: {:.0f}s".format(D["key"], truth, size, fam, time.time() - t0),
          flush=True)
    return rows


def detect(mu, sd, se, cA):
    sd = max(sd, 1e-300)
    thr = max(Z_CI * se, cA)
    return (float(1 - stats.norm.cdf((Z_CI * se - mu) / sd)),
            float(1 - stats.norm.cdf((cA - mu) / sd)),
            float(1 - stats.norm.cdf((thr - mu) / sd)))


def closed_form(signal, d, Fbar, N, K, het=0.0):
    rho = 1.0 / (K - 1)
    Ntr = N * (K - 1) / K
    mu = signal - Fbar * d / (2 * Ntr)
    var = 2 * Fbar * signal / N + K * Fbar ** 2 * d * rho * (1 + rho / 2) / N ** 2
    se2 = 2 * Fbar * signal / N + Fbar ** 2 * d / (Ntr * N) + het
    mu0 = -Fbar * d / (2 * Ntr)
    sd0 = np.sqrt(K * Fbar ** 2 * d * rho * (1 + rho / 2)) / N
    return mu, float(np.sqrt(max(var, 1e-300))), float(np.sqrt(se2)), mu0 + Z_NULL * sd0


def required_neff(s, d, K, z=None):
    z = Z_CI + stats.norm.ppf(POWER_OK) if z is None else z
    if s <= 0:
        return np.inf
    dr = d * K / (K - 1)
    if dr <= 0:
        return 4 * z * z / s
    return dr ** 2 / (s * (np.sqrt(z * z + dr) - z) ** 2)


def finish(rows):
    null = {(r["dataset"], r["family"], r["depth"]): (r["mu"], r["sd"]) for r in rows
            if r["truth"] == "A"}
    for r in rows:
        mu0, sd0 = null[(r["dataset"], r["family"], r["depth"])]
        r["cA"] = mu0 + Z_NULL * sd0
        r["p_lo"], r["p_null"], r["p_det"] = detect(r["mu"], r["sd"], r["se"], r["cA"])
        cf = closed_form(r["signal"], r["df_eff"], r["Fbar"], r["N"], r["K"], r["het"])
        r["mu_cf"], r["sd_cf"], r["se_cf"], r["cA_cf"] = cf
        r["p_det_cf"] = detect(*cf)[2]
    grid = []
    cells = sorted({(r["dataset"], r["truth"], r["size_index"], r["family"]) for r in rows})
    for c in cells:
        rs = sorted((r for r in rows if (r["dataset"], r["truth"], r["size_index"], r["family"]) == c),
                    key=lambda r: r["depth"])
        x = np.log([r["depth"] for r in rs])
        for d in GRID:
            ld = np.log(d)
            mu = np.interp(ld, x, [r["mu"] for r in rs])
            sd = np.exp(np.interp(ld, x, np.log([max(r["sd"], 1e-300) for r in rs])))
            se = np.exp(np.interp(ld, x, np.log([max(r["se"], 1e-300) for r in rs])))
            cA = np.interp(ld, x, [r["cA"] for r in rs])
            grid.append({"dataset": c[0], "truth": c[1], "size_index": c[2], "family": c[3],
                         "depth": float(d), "p_det_raw": detect(mu, sd, se, cA)[2]})
    for c in cells:
        gs = [g for g in grid if (g["dataset"], g["truth"], g["size_index"], g["family"]) == c]
        best = 0.0
        for g in sorted(gs, key=lambda g: g["depth"]):
            best = max(best, g["p_det_raw"])
            g["p_det"] = best
    return rows, grid


DECOMP_BINS = (1, 5, 10, 20)


def efficient_unit_signal(D, fam, b, chunk=120):
    ft, tx, c = D["ft"], D["tx"], D["c"]
    pall = flat_p(ft, D["eta_pos"])
    coefs = SP.planted_components(ft, 1)[fam][1]
    I = None
    for i in range(0, len(tx), chunk):
        prob = L.Problem(ft, tx[i:i + chunk], b, SP.FAMILIES[fam])
        _, _, _, R, _ = geometry(prob, None, p=pall[prob.pos])
        J = R.T @ (R * c[i:i + chunk][prob.bin_tx, None])
        I = J if I is None else I + J
    g = prob.slices["pos_five"].stop
    beta = np.zeros(prob.K)
    for blk, v in coefs.items():
        if blk in prob.slices:
            beta[prob.slices[blk]] = v
    Igg = I[:g, :g] + 1e-9 * np.trace(I[:g, :g]) / g * np.eye(g)
    Ieff = I[g:, g:] - I[g:, :g] @ np.linalg.solve(Igg, I[:g, g:])
    bf = beta[g:]
    ev = np.linalg.eigvalsh(0.5 * (Ieff + Ieff.T))
    rank = int(np.sum(ev > 1e-8 * ev.max()))
    return float(bf @ Ieff @ bf / c.sum()), rank


def size_factor(fam, size):
    return (SP.SIZE_K[size] / SP.SIZE_K[1]) ** 2 if fam == "AUG" else (SP.SIZE_S[size] / SP.SIZE_S[1]) ** 2


def decompose(pred, grid, sims_d80, Ds):
    rows, unit = [], {}
    for key, noise, _ in SIMS:
        if not any(r["dataset"] == key for r in pred):
            continue
        D = Ds.setdefault(key, prep(key, noise) if key not in Ds else Ds[key])
        T_eff_inf = D["c"].sum() ** 2 / (D["c"] ** 2).sum()
        ceiling = (1 + D["kappa"]) * T_eff_inf
        cache = C.results_path(OUT, "cache", "unit_signal_{}.json".format(key))
        cached = json.load(open(cache)) if os.path.exists(cache) else {}
        for fam in ("SW", "ST", "AUG"):
            for b in DECOMP_BINS:
                ck = "{}_{}".format(fam, b)
                if ck not in cached:
                    cached[ck] = efficient_unit_signal(D, fam, b)
                    json.dump(cached, open(cache, "w"))
                unit[(key, fam, b)] = tuple(cached[ck])
                print("[theory] {} {} {} nt: s_unit {:.3g} (rank {})".format(
                    key, fam, b, *unit[(key, fam, b)]), flush=True)
        for truth, fam in (("B", "SW"), ("C", "ST"), ("D", "AUG")):
            for size in (0, 1, 2):
                P = sorted((r for r in pred if r["dataset"] == key and r["truth"] == truth
                            and r["size_index"] == size and r["family"] == fam), key=lambda r: r["depth"])
                if not P:
                    continue
                top = P[-1]
                obs = min(P, key=lambda r: abs(np.log(r["depth"] / D["med"])))
                s_b = {b: size_factor(fam, size) * unit[(key, fam, b)][0] for b in DECOMP_BINS}
                g_inf = top["signal"]
                d, K = top["df_eff"], top["K"]
                n_star = required_neff(2 * max(g_inf, 0), d, K)
                n_star_1nt = required_neff(s_b[1], unit[(key, fam, 1)][1], K)
                neff_obs = obs["N"] / obs["Fbar"]
                gr = sorted((g["depth"], g["p_det"]) for g in grid if g["dataset"] == key
                            and g["truth"] == truth and g["size_index"] == size and g["family"] == fam)
                dp, op = d80([x[0] for x in gr], [x[1] for x in gr])
                sim = next((x for x in sims_d80 if x["dataset"] == key and x["truth"] == truth
                            and x["size_index"] == size and x["family"] == fam), None)
                if obs["p_det"] >= POWER_OK:
                    verdict = "recoverable at observed depth"
                elif g_inf <= 0:
                    verdict = "never: signal below transcript-transfer loss"
                elif n_star > ceiling:
                    verdict = "never at this library: overdispersion ceiling"
                else:
                    verdict = "depth-limited"
                regime = ("dimension-limited" if d * K / (K - 1) > (Z_CI + stats.norm.ppf(POWER_OK)) ** 2
                          else "noise-limited")
                rows.append({
                    "dataset": key, "kappa": D["kappa"], "family": fam, "truth": truth, "size_index": size,
                    "effect": SP.SIZE_K[size] if fam == "AUG" else SP.SIZE_S[size],
                    "s_1nt": s_b[1], "s_5nt": s_b[5], "s_10nt": s_b[10], "s_20nt": s_b[20],
                    "retained_10_vs_1": s_b[10] / s_b[1] if s_b[1] > 0 else np.nan,
                    "retained_20_vs_1": s_b[20] / s_b[1] if s_b[1] > 0 else np.nan,
                    "gain_exact_10nt_inf_depth": g_inf, "gain_quadratic_10nt": 0.5 * s_b[10],
                    "rank_1nt": unit[(key, fam, 1)][1], "rank_10nt": unit[(key, fam, 10)][1],
                    "df_eff_10nt": d, "T_eff": T_eff_inf, "neff_ceiling": ceiling,
                    "observed_median_depth": D["med"], "neff_at_observed": neff_obs,
                    "neff_required_80": n_star, "neff_required_80_at_1nt": n_star_1nt,
                    "pred_power_at_observed": obs["p_det"], "pred_d80": dp, "pred_d80_op": op,
                    "sim_d80": sim["sim_d80"] if sim else np.nan,
                    "sim_d80_op": sim["sim_d80_op"] if sim else "",
                    "verdict": verdict, "regime": regime})
    return rows, unit


def real_data_bounds(unit, Ds):
    eff = C.read_tsv(C.results_path("effects", "nested_effects.tsv"))
    out = []
    for key, noise, _ in SIMS:
        if key not in Ds:
            continue
        D = Ds[key]
        kappa = D["kappa"]
        c = D["c"]
        F = np.ones_like(c) if not np.isfinite(kappa) else (c + kappa) / (1 + kappa)
        Fbar = np.sum(c * F) / c.sum()
        T_eff = c.sum() ** 2 / (c ** 2).sum()
        ceiling = (1 + kappa) * T_eff
        K = D["K"]
        for fam in ("SW", "ST", "AUG"):
            rs = [r for r in eff if r["dataset"] == key and r["model"] == fam and r["bin_width"] == "10"
                  and r["metric"] == "heldout_dll_read_weighted"]
            cross = [r for r in rs if r["replicate"].split("->")[0] != r["replicate"].split("->")[1]]
            use = cross or rs
            if not use:
                continue
            N = float(np.mean([float(r["n_reads"]) for r in use]))
            d = unit[(key, fam, 10)][1]
            pen = Fbar * d / (2 * N * (K - 1) / K)
            v = float(np.mean([float(r["value"]) for r in use]))
            hi = max(float(r["ci_hi"]) for r in use)
            lo = min(float(r["ci_lo"]) for r in use)
            g_hat, g_hi = v + pen, hi + pen
            g_unit = 0.5 * unit[(key, fam, 10)][0]
            ref = SP.SIZE_K[1] if fam == "AUG" else SP.SIZE_S[1]
            n_star_hi = required_neff(2 * max(g_hi, 0), d, K)
            n_star_pt = required_neff(2 * max(g_hat, 0), d, K)
            out.append({"dataset": key, "family": fam, "kappa": kappa,
                        "evidence": "cross-replicate" if cross else "within-library",
                        "dll_observed": v, "dll_ci_lo": lo, "dll_ci_hi": hi, "n_test_reads": N,
                        "Fbar": Fbar, "df": d, "penalty": pen, "implied_gain": g_hat,
                        "implied_gain_upper": g_hi,
                        "planted_size_equivalent": ref * np.sqrt(max(g_hat, 0) / g_unit) if g_unit > 0 else np.nan,
                        "planted_size_equivalent_upper": ref * np.sqrt(max(g_hi, 0) / g_unit) if g_unit > 0 else np.nan,
                        "neff_now": N / Fbar, "neff_ceiling": ceiling,
                        "neff_required_80_at_point": n_star_pt,
                        "neff_required_80_at_upper": n_star_hi,
                        "reachable_by_depth_at_point": bool(n_star_pt <= ceiling),
                        "reachable_by_depth_at_upper": bool(n_star_hi <= ceiling),
                        "kappa_note": "" if np.isfinite(kappa) else
                        "single library: kappa not estimable, multinomial assumed (no ceiling)"})
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def sim_cells(key, noise, tag):
    stem = "{}_{}{}".format(key, noise, "_" + tag if tag else "")
    sims = {}
    for r in C.read_tsv(C.results_path("synthetic", "power_{}.tsv".format(stem))):
        k = (r["truth"], int(float(r["size_index"])), r["model"], int(float(r["target_median_depth"])),
             r["replicate"], int(float(r["seed"])))
        sims.setdefault(k, {})[r["metric"]] = float(r["value"])
    agg = {}
    for (truth, size, fam, depth, rep, seed), m in sims.items():
        agg.setdefault((truth, size, fam, depth, rep), []).append(m)
    out = {}
    for k, ms in agg.items():
        dll = np.array([m["dll"] for m in ms])
        lo = np.array([m["lo"] for m in ms])
        ok = np.isfinite(dll) & np.isfinite(lo)
        out[k] = (dll[ok], lo[ok])
    res = {}
    for (truth, size, fam, depth, rep), (dll, lo) in out.items():
        nd = out.get(("A", 0, fam, depth, rep))
        thr = np.percentile(nd[0], 95) if nd is not None and len(nd[0]) else np.nan
        rec = (lo > 0) & (dll > thr)
        n = len(dll)
        res[(truth, size, fam, depth, rep)] = {
            "n_sim": n, "mean": float(np.mean(dll)) if n else np.nan,
            "sd": float(np.std(dll, ddof=1)) if n > 1 else np.nan,
            "p_lo": float(np.mean(lo > 0)) if n else np.nan,
            "p_rec": float(np.mean(rec)) if n else np.nan, "k_rec": int(rec.sum()), "thr": float(thr)}
    return res


def d80(depths, power):
    depths, power = np.asarray(depths, float), np.asarray(power, float)
    if power[0] >= POWER_OK:
        return float(depths[0]), "<="
    for i in range(1, len(depths)):
        if power[i] >= POWER_OK:
            a, b = power[i - 1], power[i]
            t = (POWER_OK - a) / (b - a) if b > a else 1.0
            return float(np.exp(np.log(depths[i - 1]) + t * np.log(depths[i] / depths[i - 1]))), "="
    return float(depths[-1]), ">"


def validate(pred, grid):
    rows, d80_rows = [], []
    for key, noise, tag in SIMS:
        P = [r for r in pred if r["dataset"] == key]
        if not P:
            continue
        sims = sim_cells(key, noise, tag)
        for r in P:
            d = int(round(r["depth"]))
            if d not in SIM_DEPTHS:
                continue
            for rep in ("rep1->rep2", "rep1->rep1"):
                s = sims.get((r["truth"], r["size_index"], r["family"], d, rep))
                if s is None:
                    continue
                lo, hi = wilson(s["k_rec"], s["n_sim"])
                rows.append({"dataset": key, "noise": noise, "kappa": r["kappa"], "truth": r["truth"],
                             "size_index": r["size_index"], "family": r["family"], "depth": d,
                             "direction": rep, "n_sim": s["n_sim"],
                             "sim_mean": s["mean"], "pred_mean": r["mu"], "pred_mean_cf": r["mu_cf"],
                             "sim_sd": s["sd"], "pred_sd": r["sd"],
                             "sim_null_thr": s["thr"], "pred_null_thr": r["cA"],
                             "sim_p_lo": s["p_lo"], "pred_p_lo": r["p_lo"],
                             "sim_power": s["p_rec"], "sim_power_lo": lo, "sim_power_hi": hi,
                             "pred_power": r["p_det"], "pred_power_cf": r["p_det_cf"],
                             "pred_se": r["se"], "signal": r["signal"], "penalty": r["penalty"],
                             "Fbar": r["Fbar"], "df_eff": r["df_eff"], "N": r["N"]})
        cells = sorted({(r["truth"], r["size_index"], r["family"]) for r in P if r["truth"] != "A"})
        for truth, size, fam in cells:
            gr = sorted((g["depth"], g["p_det"]) for g in grid if g["dataset"] == key
                        and g["truth"] == truth and g["size_index"] == size and g["family"] == fam)
            dp, op = d80([x[0] for x in gr], [x[1] for x in gr])
            sp_ = [sims.get((truth, size, fam, d, "rep1->rep2")) for d in SIM_DEPTHS]
            if any(x is None for x in sp_):
                continue
            ds, os_ = d80(SIM_DEPTHS, [x["p_rec"] for x in sp_])
            pr = {int(round(r["depth"])): r["p_det"] for r in P if r["truth"] == truth
                  and r["size_index"] == size and r["family"] == fam}
            dg, og = d80(SIM_DEPTHS, [pr[d] for d in SIM_DEPTHS])
            d80_rows.append({"dataset": key, "truth": truth, "size_index": size, "family": fam,
                             "pred_d80": dp, "pred_d80_op": op, "pred_d80_sim_grid": dg,
                             "pred_d80_sim_grid_op": og, "sim_d80": ds, "sim_d80_op": os_,
                             "n_sim": sp_[0]["n_sim"]})
    return rows, d80_rows


def calibration(rows):
    out = []
    for key in sorted({r["dataset"] for r in rows}) + ["all"]:
        R = [r for r in rows if (key == "all" or r["dataset"] == key) and r["truth"] != "A"
             and r["direction"] == "rep1->rep2"]
        if not R:
            continue
        inf = [r for r in R if 0.02 < r["sim_power"] < 0.98 or 0.02 < r["pred_power"] < 0.98]
        e = np.array([r["pred_power"] - r["sim_power"] for r in R])
        ecf = np.array([r["pred_power_cf"] - r["sim_power"] for r in R])
        ei = np.array([r["pred_power"] - r["sim_power"] for r in inf]) if inf else np.array([np.nan])
        inside = np.mean([r["sim_power_lo"] - 1e-9 <= r["pred_power"] <= r["sim_power_hi"] + 1e-9
                          for r in R])
        N = [r for r in rows if (key == "all" or r["dataset"] == key) and r["truth"] == "A"
             and r["direction"] == "rep1->rep2"]
        out.append({"dataset": key, "n_cells": len(R), "n_informative": len(inf),
                    "power_mae": float(np.mean(np.abs(e))), "power_bias": float(np.mean(e)),
                    "power_mae_informative": float(np.mean(np.abs(ei))),
                    "power_max_abs_err": float(np.max(np.abs(e))),
                    "power_mae_closed_form": float(np.mean(np.abs(ecf))),
                    "frac_within_wilson_ci": float(inside),
                    "null_mean_median_rel_err": float(np.median(
                        [abs(r["pred_mean"] / r["sim_mean"] - 1) for r in N if r["sim_mean"]])),
                    "null_sd_median_rel_err": float(np.median(
                        [abs(r["pred_sd"] / r["sim_sd"] - 1) for r in N if r["sim_sd"] > 0])),
                    "mean_median_abs_err_over_sim_sd": float(np.median(
                        [abs(r["pred_mean"] - r["sim_mean"]) / r["sim_sd"] for r in R if r["sim_sd"] > 0]))})
    return out


def write(rows, name):
    path = C.results_path(OUT, name)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("{:.6g}".format(v) if isinstance(v, float) else v) for k, v in r.items()})
    print("[theory] -> {}".format(os.path.relpath(path, C.ROOT)), flush=True)


TEXT = ("dataset", "noise", "truth", "family", "direction", "pred_d80_op", "sim_d80_op",
        "pred_d80_sim_grid_op")


def read(name):
    out = []
    for r in C.read_tsv(C.results_path(OUT, name)):
        rr = {}
        for k, v in r.items():
            rr[k] = v if k in TEXT else float(v)
        for k in ("size_index", "K"):
            if k in rr:
                rr[k] = int(rr[k])
        out.append(rr)
    return out


def cells_for():
    out = [("A", 0, fam) for fam in ("SW", "ST", "AUG")]
    for truth in ("B", "C", "D", "E"):
        for size in (0, 1, 2):
            out += [(truth, size, fam) for fam in SP.PLANTED[truth]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(s[0] for s in SIMS))
    ap.add_argument("--stage", default="all", choices=("cells", "validate", "decompose", "all"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    want = args.datasets.split(",")
    if args.stage in ("cells", "all"):
        new = []
        for key, noise, _ in SIMS:
            if key not in want:
                continue
            t0 = time.time()
            with mp.get_context("spawn").Pool(args.workers, initializer=_init,
                                              initargs=(key, noise)) as pool:
                for rows in pool.imap_unordered(cell, cells_for()):
                    new += rows
            print("[theory] {} done in {:.0f} min".format(key, (time.time() - t0) / 60), flush=True)
        path = C.results_path(OUT, "theory_cells_raw.tsv")
        old = [r for r in (read("theory_cells_raw.tsv") if os.path.exists(path) else [])
               if r["dataset"] not in want]
        write(old + new, "theory_cells_raw.tsv")
    if args.stage in ("validate", "all"):
        pred, grid = finish(read("theory_cells_raw.tsv"))
        write(pred, "theory_predictions.tsv")
        write(grid, "theory_power_grid.tsv")
        rows, d80_rows = validate(pred, grid)
        write(rows, "theory_validation.tsv")
        write(d80_rows, "theory_required_depth.tsv")
        cal = calibration(rows)
        write(cal, "theory_calibration.tsv")
        for c in cal:
            print("[theory] {dataset}: power MAE {power_mae:.3f} (informative cells {power_mae_informative:.3f}, "
                  "n={n_informative}; closed form {power_mae_closed_form:.3f}), within Wilson CI "
                  "{frac_within_wilson_ci:.0%}, null mean rel err {null_mean_median_rel_err:.0%}, "
                  "null SD rel err {null_sd_median_rel_err:.0%}".format(**c), flush=True)
    if args.stage in ("decompose", "all"):
        pred, grid = read("theory_predictions.tsv"), read("theory_power_grid.tsv")
        Ds = {}
        rows, unit = decompose(pred, grid, read("theory_required_depth.tsv"), Ds)
        write(rows, "theory_decomposition.tsv")
        write(real_data_bounds(unit, Ds), "theory_real_data.tsv")
    if args.stage in ("validate", "all"):
        C.update_manifest("recoverability_theory", {
            "bin_width": BIN, "anchors": list(ANCHORS), "criterion":
                "CI excludes 0 AND > truth-A 95th percentile (as 06)",
            "simulations": [list(s) for s in SIMS], "calibration": cal})


if __name__ == "__main__":
    main()
