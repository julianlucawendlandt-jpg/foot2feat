import importlib
import os

import numpy as np
from scipy import stats

TH = importlib.import_module("analysis.recoverability.11_recoverability_theory")
SP, L, EFF, C = TH.SP, TH.L, TH.EFF, TH.C

Z95, Z_NULL = stats.norm.ppf(0.975), stats.norm.ppf(0.95)
GROUP_RATIO = 1.15
OUT = "design"
_PREP = {}


def _prep(key):
    if key not in _PREP:
        _PREP[key] = TH.prep(key, "multinomial")
    return _PREP[key]


class Info(object):
    def __init__(self, key, fam, b=10):
        self.key, self.fam, self.b = key, fam, b
        path = C.results_path(OUT, "cache", "info_{}_{}_{}.npz".format(key, fam, b))
        if not os.path.exists(path):
            np.savez(path, **self._build())
        z = np.load(path)
        for k in z.files:
            setattr(self, k, z[k])
        self.K, self.gam, self.med, self.reads = int(self.K), int(self.gam), float(self.med), float(self.reads)

    def _build(self):
        D = _prep(self.key)
        ft, tx, c, fold, K = D["ft"], D["tx"], D["c"], D["fold"], D["K"]
        lg = np.floor(np.log(c) / np.log(GROUP_RATIO)).astype(int)
        _, grp = np.unique(lg, return_inverse=True)
        G = int(grp.max()) + 1
        ctil = np.array([np.exp(np.mean(np.log(c[grp == g]))) for g in range(G)])
        pall = TH.flat_p(ft, D["eta_pos"])
        fam_blocks = SP.FAMILIES[self.fam]
        probe = L.Problem(ft, tx[:1], self.b, fam_blocks)
        P = probe.K
        J1 = np.zeros((K, G, P, P))
        J2 = np.zeros((K, G, P, P))
        pos_of = {t: i for i, t in enumerate(tx)}
        for s in range(0, len(tx), 150):
            sub = np.sort(tx[s:s + 150])
            prob = L.Problem(ft, sub, self.b, fam_blocks)
            _, _, _, R, _ = TH.geometry(prob, None, p=pall[prob.pos])
            ii = np.array([pos_of[t] for t in sub])[prob.bin_tx]
            key = fold[ii] * G + grp[ii]
            for kk in np.unique(key):
                rows = key == kk
                Rr, cw = R[rows], c[ii[rows]]
                f, g = divmod(int(kk), G)
                J1[f, g] += Rr.T @ (Rr * cw[:, None])
                J2[f, g] += Rr.T @ (Rr * (cw * cw)[:, None])
        beta = np.zeros(P)
        for blk, v in SP.planted_components(ft, 1)[self.fam][1].items():
            if blk in probe.slices:
                beta[probe.slices[blk]] = v
        n = np.zeros((K, G))
        cs = np.zeros((K, G))
        c2 = np.zeros((K, G))
        np.add.at(n, (fold, grp), 1)
        np.add.at(cs, (fold, grp), c)
        np.add.at(c2, (fold, grp), c * c)
        return {"J1": J1, "J2": J2, "ctil": ctil, "n_fg": n, "c_fg": cs, "c2_fg": c2, "beta": beta,
                "gam": probe.slices["pos_five"].stop, "K": K, "med": D["med"], "reads": c.sum()}

    def depth_from_leader_reads(self, leader_reads):
        return self.med * leader_reads / self.reads

    def leader_reads_from_depth(self, depth):
        return depth * self.reads / self.med


def _sf(k, mu):
    return stats.poisson.sf(k, mu)


def _p_all_exceed(m, v, rho, t, R, n_mc=200000, seed=0):
    sd = np.sqrt(max(v, 1e-300))
    if R == 1:
        return float(1 - stats.norm.cdf((t - m) / sd))
    if rho >= 0:
        x, w = np.polynomial.hermite_e.hermegauss(120)
        w = w / np.sqrt(2 * np.pi)
        a = (m - t + np.sqrt(rho) * sd * x) / (sd * np.sqrt(max(1 - rho, 1e-12)))
        return float(np.sum(w * stats.norm.cdf(a) ** R))
    cov = v * ((1 - rho) * np.eye(R) + rho * np.ones((R, R)))
    z = np.random.default_rng(seed).multivariate_normal(np.full(R, m), cov, size=n_mc)
    return float(np.mean(np.all(z > t, axis=1)))


def loo_design(info, depth, R, kappa, size=1.0, cv2=1.0, shared_depth=False, return_parts=False):
    if shared_depth and R != 2:
        raise ValueError("shared_depth is 06's two-library design")
    m, K, g_ = EFF.MIN_READS, info.K, info.gam
    lam = depth / info.med
    mu = info.ctil * lam
    oth = (R - 1) * mu
    if shared_depth:
        te1, te2 = lam * _sf(m - 2, mu), lam * lam * _sf(m - 3, mu)
        tr1, tr2 = te1, te2
        p_el = _sf(m - 1, mu)
    else:
        p_oth = _sf(m - 1, oth)
        te1, te2 = lam * _sf(m - 2, mu) * p_oth, lam * lam * _sf(m - 3, mu) * p_oth
        tr1, tr2 = lam * _sf(m - 2, oth), lam * lam * _sf(m - 3, oth)
        p_el = _sf(m - 1, mu) * p_oth
    dm = 0.0 if not np.isfinite(kappa) else 1.0 / (1.0 + kappa)
    I_te = np.tensordot(te1, info.J1, axes=([0], [1]))
    B_te = I_te + dm * np.tensordot(te2, info.J2, axes=([0], [1]))
    I_tr = np.tensordot(tr1, info.J1, axes=([0], [1]))
    B_tr = I_tr + dm * np.tensordot(tr2, info.J2, axes=([0], [1]))
    N_k = info.c_fg @ te1
    N_dir = float(N_k.sum())
    if N_dir <= 0:
        return None
    P = I_te.shape[1]
    Lam = TH.LAM * np.eye(P)
    sI, sB = I_tr.sum(0), B_tr.sum(0)
    beta = info.beta
    U = []
    for k in range(K):
        Mtr = (R - 1) * (sI - I_tr[k])
        Btr = (R - 1) * (sB - B_tr[k])
        H1 = np.linalg.inv(Mtr + Lam)
        H0 = np.zeros((P, P))
        H0[:g_, :g_] = np.linalg.inv(Mtr[:g_, :g_] + Lam[:g_, :g_])
        dl = beta.copy()
        dl[:g_] = -np.linalg.solve(Mtr[:g_, :g_] + Lam[:g_, :g_], Mtr[:g_, g_:] @ beta[g_:])
        I = I_te[k]
        W = 0.5 * (H0 @ I @ H0 - H1 @ I @ H1)
        Mm = H1 - H0
        U.append({"G": 0.5 * dl @ I @ dl, "W": W, "M": Mm, "l": -H0 @ I @ dl, "d": dl,
                  "Btr": Btr, "mfree": float(np.sum(W * Btr)),
                  "s_self": float(dl @ B_te[k] @ dl), "e_self": float(np.sum((Mm @ B_te[k] @ Mm) * Btr))})

    def Cmat(k, kk, same):
        n = (R - 1) if same else (R - 2)
        if n <= 0:
            return None
        excl = {k, kk}
        return n * sum(B_tr[f] for f in range(K) if f not in excl)

    def pair(k, kk, same):
        a, b = U[k], U[kk]
        Cm = Cmat(k, kk, same)
        fv = bv = 0.0
        if Cm is not None:
            fv += 2 * np.sum((a["W"] @ Cm) * (b["W"] @ Cm).T)
            bv += a["l"] @ Cm @ b["l"]
        if same and k == kk:
            fv += a["e_self"]
            bv += a["s_self"]
        if not same and k != kk:
            fv += np.sum((a["M"] @ B_te[kk]) * (b["M"] @ B_te[k]).T)
            if Cm is not None:
                fv += 2 * np.sum((a["W"] @ B_te[kk] @ b["M"]) * Cm.T)
                fv += 2 * np.sum((b["W"] @ B_te[k] @ a["M"]) * Cm.T)
            bv += a["d"] @ B_te[k] @ b["l"] + b["d"] @ B_te[kk] @ a["l"]
        return fv, bv

    free = {"same": 0.0, "diff": 0.0}
    bet = {"same": 0.0, "diff": 0.0}
    pairs = {}
    for k in range(K):
        for kk in range(K):
            for same in (True, False):
                fv, bv = pair(k, kk, same)
                pairs[(k, kk, same)] = (fv, bv)
                free["same" if same else "diff"] += fv
                bet["same" if same else "diff"] += bv
    s2 = size * size
    Gsum = sum(u["G"] for u in U)
    mfree = sum(u["mfree"] for u in U)
    mean = (s2 * Gsum + mfree) / N_dir
    v = (free["same"] + s2 * bet["same"]) / N_dir ** 2
    cv_ = (free["diff"] + s2 * bet["diff"]) / N_dir ** 2
    rho = cv_ / v if v > 0 else 0.0
    v0, c0 = free["same"] / N_dir ** 2, free["diff"] / N_dir ** 2
    gain = s2 * Gsum / N_dir
    w1 = float(info.c_fg.sum(0) @ te1)
    w2 = float(info.c2_fg.sum(0) @ te2 + info.c_fg.sum(0) @ te1)
    T_eff = w1 * w1 / w2 if w2 > 0 else np.nan
    dm_inflation = (w1 + dm * float(info.c2_fg.sum(0) @ te2)) / w1 if w1 > 0 else np.nan
    het = cv2 * gain * gain / T_eff
    num = sum(s2 * u["s_self"] + u["e_self"] for u in U) / N_dir ** 2
    se_dir = np.sqrt(num + het)
    se_pool = np.sqrt(num / R + het)
    var_pool = (v + (R - 1) * cv_) / R
    var0_pool = (v0 + (R - 1) * c0) / R
    mean0 = mfree / N_dir
    thr_dir = max(Z95 * se_dir, mean0 + Z_NULL * np.sqrt(max(v0, 0)))
    thr_pool = max(Z95 * se_pool, mean0 + Z_NULL * np.sqrt(max(var0_pool, 0)))
    p_dir = float(1 - stats.norm.cdf((thr_dir - mean) / np.sqrt(max(v, 1e-300))))
    p_pool = float(1 - stats.norm.cdf((thr_pool - mean) / np.sqrt(max(var_pool, 1e-300))))
    p_every = _p_all_exceed(mean, v, rho, thr_dir, R)
    Mall = R * sI
    Seff = Mall[g_:, g_:] - Mall[g_:, :g_] @ np.linalg.solve(Mall[:g_, :g_] + Lam[:g_, :g_], Mall[:g_, g_:])
    one = sI[g_:, g_:] - sI[g_:, :g_] @ np.linalg.solve(sI[:g_, :g_] + Lam[:g_, :g_], sI[:g_, g_:])
    bf = size * beta[g_:]
    out = {"R": R, "depth": float(depth), "mean_dll": float(mean), "gain_per_read": float(gain),
           "beta_information_total": float(bf @ Seff @ bf), "beta_information_one_library": float(bf @ one @ bf),
           "penalty": float(-mean0), "sd_direction": float(np.sqrt(v)), "loo_corr": float(rho),
           "power_pooled": p_pool, "power_every": p_every, "power_one_direction": p_dir,
           "power_every_independent": p_dir ** R,
           "test_reads_per_library": N_dir, "leader_reads_per_library": float(info.leader_reads_from_depth(depth)),
           "eligible_transcripts": float(info.n_fg.sum(0) @ p_el), "T_eff": float(T_eff),
           "dm_inflation": float(dm_inflation)}
    if return_parts:
        out["parts"] = {"U": U, "B_te": B_te, "B_tr": B_tr, "I_te": I_te, "I_tr": I_tr,
                        "free": free, "bet": bet, "pairs": pairs, "N_k": N_k, "size": size}
    return out


def size_from_gain(info, depth, kappa, obs_dll, R=2):
    r = loo_design(info, depth, R, kappa, size=1.0)
    g_unit, pen = r["gain_per_read"], r["penalty"]
    s2 = (obs_dll + pen) / g_unit
    ok = bool(s2 > 0 and (obs_dll >= 0 or -obs_dll <= pen))
    return float(np.sqrt(max(s2, 0.0))), ok, pen


def cv2_from_se(info, depth, kappa, size, se_obs, R=2):
    r = loo_design(info, depth, R, kappa, size=size, cv2=0.0, return_parts=True)
    if r is None or r["gain_per_read"] <= 0:
        return 1.0
    U, N = r["parts"]["U"], r["test_reads_per_library"]
    num = sum(size * size * u["s_self"] + u["e_self"] for u in U) / N ** 2
    return float(max(se_obs ** 2 - num, 0.0) * r["T_eff"] / r["gain_per_read"] ** 2)


def unit_covariance(res):
    pr = res["parts"]
    R, K, s2 = res["R"], len(pr["U"]), pr["size"] ** 2
    n = R * K
    S = np.zeros((n, n))
    for r in range(R):
        for k in range(K):
            for q in range(R):
                for kk in range(K):
                    fv, bv = pr["pairs"][(k, kk, r == q)]
                    S[r * K + k, q * K + kk] = fv + s2 * bv
    return S


def pooled_dll(unit_gains, unit_reads):
    return float(np.sum(unit_gains) / np.sum(unit_reads))
