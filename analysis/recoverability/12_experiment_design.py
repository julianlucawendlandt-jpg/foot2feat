import argparse
import importlib
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TH = importlib.import_module("analysis.recoverability.11_recoverability_theory")
SP, L, EFF, splits, C = TH.SP, TH.L, TH.EFF, TH.splits, TH.C

COST_LIBRARY = 400.0
COST_PER_MILLION = 3.0
FAMS = ("SW", "ST", "AUG")
Z95, Z_NULL = stats.norm.ppf(0.975), stats.norm.ppf(0.95)
POWER_OK = 0.8
OUT = "design"
NOISE = {"gse174329_eif3b": "dm", "gse139131_unsel": "dm", "archer_yeast": "multinomial",
         "gse139131_eif3b": "multinomial"}


class Pilot(object):
    def __init__(self, key, b=10, kappa=None):
        recs, meta = EFF.load_panel(key)
        reps = sorted(meta["replicates"])
        reads = {r: EFF.leader_reads(recs, r) for r in reps}
        keep = np.max([reads[r] for r in reps], axis=0) >= EFF.MIN_READS
        species = C.DATASETS[key]["species"]
        fo = splits.fold_of(species)
        infold = np.array([fo.get(t["chrom"], -1) >= 0 for t in recs])
        c = np.mean([reads[r] for r in reps], axis=0)
        sel = keep & infold & (c > 0)
        self.key, self.b, self.reps, self.species = key, b, reps, species
        self.c = c[sel].astype(float)
        self.med = float(np.median(c[keep]))
        self.K = splits.load(species)["n_folds"]
        self.reads_per_library = float(np.mean([reads[r][keep].sum() for r in reps]))
        raw = {r["replicate"]: float(r["value"]) for r in C.read_tsv(C.results_path("qc", "qc_summary.tsv"))
               if r["dataset"] == key and r["metric"] == "raw_reads"}
        self.yield_ = float(np.mean([reads[r][keep].sum() / raw[r] for r in reps if r in raw])) if raw else np.nan
        self.kappa, self.kappa_source = self._kappa(kappa, recs, reps, reads, keep)
        self.s_unit, self.rank = {}, {}
        self._unit_signal()
        self.cv2 = self._cv2()

    def _kappa(self, kappa, recs, reps, reads, keep):
        if kappa is not None:
            return float(kappa), "user"
        m = C.measured("synthetic_dm_alpha_" + self.key, {})
        if "alpha" in m:
            return float(m["alpha"]), "replicate pair (06 calibration)"
        if len(reps) >= 2:
            from data.synthetic import calibrate_concentration
            pairs = [(t["counts"][reps[0]][:t["msi"] + 3], t["counts"][reps[1]][:t["msi"] + 3])
                     for t, k in zip(recs, keep) if k]
            a, _, _ = calibrate_concentration(pairs, np.random.default_rng(0))
            return float(a), "replicate pair (calibrated here)"
        return np.inf, "single library: not estimable, multinomial assumed"

    def _unit_signal(self):
        path = C.results_path("theory", "cache", "unit_signal_{}.json".format(self.key))
        cached = json.load(open(path)) if os.path.exists(path) else {}
        D = None
        for fam in FAMS:
            k = "{}_{}".format(fam, self.b)
            if k not in cached:
                D = D or TH.prep(self.key, "multinomial")
                cached[k] = TH.efficient_unit_signal(D, fam, self.b)
                json.dump(cached, open(path, "w"))
            self.s_unit[fam], self.rank[fam] = cached[k][0], int(cached[k][1])

    def _cv2(self):
        out = {f: 1.0 for f in FAMS}
        path = C.results_path("theory", "theory_cells_raw.tsv")
        if not os.path.exists(path):
            return out
        truth = {"SW": "B", "ST": "C", "AUG": "D"}
        for r in C.read_tsv(path):
            f = r["family"]
            if (r["dataset"] == self.key and r["truth"] == truth.get(f) and r["size_index"] in ("1", "1.0")
                    and float(r["depth"]) == 500.0 and float(r["signal"]) > 0):
                out[f] = float(r["het"]) * float(r["T_eff"]) / float(r["signal"]) ** 2
        return out

    def depth_from_leader_reads(self, leader_reads_per_library):
        return self.med * leader_reads_per_library / self.reads_per_library

    def sequenced_from_depth(self, depth):
        return depth * self.reads_per_library / self.med / self.yield_

    def depth_from_sequenced(self, seq_per_library):
        return self.depth_from_leader_reads(seq_per_library * self.yield_)


def planted_gain(pilot, fam, size):
    ref = SP.SIZE_K[1] if fam == "AUG" else SP.SIZE_S[1]
    return 0.5 * pilot.s_unit[fam] * (size / ref) ** 2


def observed_gain(pilot, fam):
    rs = [r for r in C.read_tsv(C.results_path("effects", "nested_effects.tsv"))
          if r["dataset"] == pilot.key and r["model"] == fam and r["bin_width"] == str(pilot.b)
          and r["metric"] == "heldout_dll_read_weighted"]
    cross = [r for r in rs if r["replicate"].split("->")[0] != r["replicate"].split("->")[1]]
    use = cross or rs
    v = float(np.mean([float(r["value"]) for r in use]))
    N = float(np.mean([float(r["n_reads"]) for r in use]))
    F = fbar(pilot, pilot.med)
    return v + F * pilot.rank[fam] / (2 * N * (pilot.K - 1) / pilot.K), v


def fbar(pilot, depth):
    mu = pilot.c * depth / pilot.med
    if not np.isfinite(pilot.kappa):
        return 1.0
    return float(np.sum(mu * (mu + pilot.kappa)) / (1 + pilot.kappa) / mu.sum())


def design(pilot, gain, fam, depth, R=2, cv2=None):
    m, K, d, kap = EFF.MIN_READS, pilot.K, pilot.rank[fam], pilot.kappa
    mu = pilot.c * depth / pilot.med
    w1 = mu * stats.poisson.sf(m - 2, mu)
    w2 = mu * mu * stats.poisson.sf(m - 3, mu) + w1
    wF = w1 if not np.isfinite(kap) else (w2 + kap * w1) / (1 + kap)
    p_tr = stats.poisson.sf(m - 1, (R - 1) * mu)
    eligible = stats.poisson.sf(m - 1, mu) * p_tr
    Nlib = float(np.sum(w1 * p_tr))
    if Nlib <= 0:
        return None
    F = float(np.sum(wF * p_tr) / Nlib)
    Ntr = (K - 1) / K * float(np.sum((R - 1) * mu * stats.poisson.sf(m - 2, (R - 1) * mu)))
    T_eff = Nlib ** 2 / float(np.sum(w2 * p_tr))
    het = (pilot.cv2[fam] if cv2 is None else cv2) * gain ** 2 / T_eff
    mean = gain - F * d / (2 * Ntr)

    def rule(N, S):
        rho = (N / S) / Ntr
        tv = S * F ** 2 * d * rho * (1 + rho / 2) / N ** 2
        sd = np.sqrt(2 * F * max(gain, 0) / N + tv)
        se = np.sqrt(2 * F * max(gain, 0) / N + F ** 2 * d / (Ntr * N) + het)
        cnull = -F * d / (2 * Ntr) + Z_NULL * np.sqrt(tv)
        return float(1 - stats.norm.cdf((max(Z95 * se, cnull) - mean) / sd))

    p_pool = rule(R * Nlib, R * K)
    p_dir = rule(Nlib, K)
    T_inf = pilot.c.sum() ** 2 / (pilot.c ** 2).sum()
    return {"depth": float(depth), "R": R, "mean_dll": float(mean), "power_pooled": p_pool,
            "power_every": p_dir ** R, "power_one_direction": p_dir,
            "eligible_leaders": float(eligible.sum()), "eligible_frac": float(eligible.mean()),
            "Fbar": F, "effective_reads": R * Nlib / F,
            "ceiling": R * (1 + kap) * T_inf if np.isfinite(kap) else np.inf}


E = importlib.import_module("analysis.recoverability.design_engine")
VALIDATED_BIN = 10


def status(R):
    return "validated" if R == 2 else "extrapolated"


def bin_status(b):
    return "simulation-validated (10 nt)" if b == VALIDATED_BIN else "analytic only (not simulation-validated)"


def size_units(fam, x):
    return x / (SP.SIZE_K[1] if fam == "AUG" else SP.SIZE_S[1])


def design_row(pilot, info, size, R, total_seq, cv2):
    per = total_seq / R
    depth = info.depth_from_leader_reads(per * pilot.yield_)
    base = {"dataset": pilot.key, "family": info.fam, "bin_width": info.b, "R": R,
            "total_sequenced_reads": total_seq, "sequenced_reads_per_library": per,
            "leader_reads_per_library": per * pilot.yield_, "median_depth": depth,
            "validation_status": status(R), "bin_width_status": bin_status(info.b)}
    r = E.loo_design(info, depth, R, pilot.kappa, size=size, cv2=cv2)
    if r is None:
        return dict(base, power_pooled=0.0, power_every=0.0, power_every_independent=0.0,
                    eligible_transcripts=0.0)
    keep = ("power_pooled", "power_every", "power_every_independent", "power_one_direction",
            "eligible_transcripts", "T_eff", "mean_dll", "gain_per_read", "penalty",
            "beta_information_total", "beta_information_one_library", "dm_inflation")
    out = dict(base, **{k: r[k] for k in keep})
    out["loo_corr_mean"] = out["loo_corr_min"] = out["loo_corr_max"] = r["loo_corr"]
    return out


def frontier(pilot, info, size, Rs, budgets, cv2):
    return [design_row(pilot, info, size, R, B, cv2) for R in Rs for B in budgets]


def need_from_frontier(rows, R, rule="power_pooled", target=POWER_OK):
    pts = sorted((r["total_sequenced_reads"], r.get(rule, 0.0)) for r in rows if r["R"] == R)
    for (b0, p0), (b1, p1) in zip(pts, pts[1:]):
        if p0 < target <= p1:
            t = (target - p0) / (p1 - p0)
            return float(np.exp(np.log(b0) + t * (np.log(b1) - np.log(b0))))
    return float(pts[0][0]) if pts and pts[0][1] >= target else np.inf


def reads_for_power(pilot, info, size, R, cv2, rule="power_pooled", target=POWER_OK, lo=1e5, hi=1e11):
    def f(B):
        return design_row(pilot, info, size, R, B, cv2).get(rule, 0.0) - target
    if f(hi) < 0:
        return np.inf
    if f(lo) >= 0:
        return lo
    while hi / lo > 1.02:
        mid = np.sqrt(lo * hi)
        lo, hi = (lo, mid) if f(mid) >= 0 else (mid, hi)
    return float(hi)


def depth_for_power(pilot, info, size, cv2, rule="power_pooled", R=2, target=POWER_OK):
    B = reads_for_power(pilot, info, size, R, cv2, rule, target)
    return info.depth_from_leader_reads(B / R * pilot.yield_) if np.isfinite(B) else np.inf


def cost_overlay(reads_needed, c_lib=COST_LIBRARY, c_mill=COST_PER_MILLION):
    rows = [{"R": R, "reads_needed": B, "cost_eur": R * c_lib + B / 1e6 * c_mill if np.isfinite(B) else np.inf,
             "cost_library_eur": c_lib, "cost_per_million_eur": c_mill, "validation_status": "illustrative"}
            for R, B in sorted(reads_needed.items())]
    ok = [r for r in rows if np.isfinite(r["cost_eur"])]
    return rows, (min(ok, key=lambda r: r["cost_eur"]) if ok else None)


def observed_size(pilot, info, fam, depth=None):
    rs = [r for r in C.read_tsv(C.results_path("effects", "nested_effects.tsv"))
          if r["dataset"] == pilot.key and r["model"] == fam and r["bin_width"] == str(info.b)
          and r["metric"] == "heldout_dll_read_weighted"]
    cross = [r for r in rs if r["replicate"].split("->")[0] != r["replicate"].split("->")[1]]
    use = cross or rs
    obs = float(np.mean([float(r["value"]) for r in use]))
    se = float(np.mean([(float(r["ci_hi"]) - float(r["ci_lo"])) / (2 * Z95) for r in use]))
    depth = info.med if depth is None else depth
    size, ok, pen = E.size_from_gain(info, depth, pilot.kappa, obs)
    cv2 = E.cv2_from_se(info, depth, pilot.kappa, size, se) if size > 0 else pilot.cv2[fam]
    return {"size": size, "self_check": ok, "observed_dll": obs, "penalty": pen, "cv2": cv2}


IDF = importlib.import_module("analysis.recoverability.06b_identifiability")
MECHANISMS = {
    "commitment_dropoff": ("share_logit", ["occupancy", "occupancy+cds30", "occupancy+80S_unknown_scale",
                                           "occupancy+80S_known_scale"]),
    "leak_blur": ("leak_logit", ["occupancy", "occupancy@3nt", "occupancy@5nt", "occupancy+cds30"]),
    "reinit_direct": ("reinit_logit", ["main_AUG_output_only", "occupancy@10nt", "occupancy"]),
}


def mechanism_information(mech, obs, N):
    import torch
    torch.set_default_dtype(torch.float64)
    b = int(obs.split("@")[1].rstrip("nt")) if "@" in obs else 1
    if mech == "commitment_dropoff":
        n_cds = 30 if "cds30" in obs else 0
        n, start, main, _ = IDF.template(n_cds=n_cds)
        ex, a_step, a_stay, rho = (torch.tensor(v) for v in (0.4, 0.2, 0.4, 2.0))

        def run(t):
            s = torch.sigmoid(t[0])
            return IDF.chain_profile(n, start, main, (a_step, s * ex, a_stay, (1 - s) * ex))

        th = torch.tensor([0.5])
        J = IDF.fisher_occupancy(lambda t: run(t)[0], th, N, b)
        names = ["share_logit"]
        commit = lambda t: run(t)[1][IDF.COMMIT_MAIN]
        if "80S_known" in obs:
            g = torch.autograd.functional.jacobian(lambda t: rho * commit(t) * N, th).reshape(1, -1)
            J = J + g.T @ g / (rho * commit(th) * N)
        if "80S_unknown" in obs:
            th2 = torch.tensor([0.5, float(torch.log(rho))])
            mu2 = lambda t: torch.exp(t[1]) * commit(t[:1]) * N
            g = torch.autograd.functional.jacobian(mu2, th2).reshape(1, -1)
            J2 = torch.zeros(2, 2)
            J2[0, 0] = J[0, 0]
            J, names = J2 + g.T @ g / mu2(th2), ["share_logit", "log_rho"]
        return J.detach().numpy(), names
    if mech == "leak_blur":
        n, start, main, _ = IDF.template(n_cds=30 if "cds30" in obs else 0)
        ex = torch.tensor(0.3)

        def fn(t):
            leak = torch.sigmoid(t[0]) * 0.6
            prof, _ = IDF.chain_profile(n, start, main, (leak, ex * 0.999, 1 - ex - leak, ex * 0.001))
            return IDF.blur(prof, 0.33 * torch.sigmoid(t[1]))
        J = IDF.fisher_occupancy(fn, torch.tensor([-1.0, -0.5]), N, b)
        return J.detach().numpy(), ["leak_logit", "blur_logit"]
    if mech == "reinit_direct":
        n, start, main, ret = IDF.template(uorf=(30, 57))
        aug = tuple(torch.tensor(v) for v in (0.1, 0.5, 0.3, 0.1))
        run = lambda t: IDF.chain_profile(n, start, main, aug, uorf_logits=(t[0], t[1]), ret=ret)
        th = torch.tensor([6.0, 0.0])
        if obs == "main_AUG_output_only":
            g = torch.autograd.functional.jacobian(lambda t: run(t)[1][IDF.COMMIT_MAIN] * N, th).reshape(1, -1)
            J = g.T @ g / (run(th)[1][IDF.COMMIT_MAIN] * N)
        else:
            J = IDF.fisher_occupancy(lambda t: run(t)[0], th, N, b)
        return J.detach().numpy(), ["uorf_commit_logit", "reinit_logit"]
    raise KeyError(mech)


def score_observables(mech, candidates=None, N=500, eps=1e-8):
    target, default = MECHANISMS[mech]
    rows = []
    for obs in candidates or default:
        J, names = mechanism_information(mech, obs, N)
        ev = np.linalg.eigvalsh(J)
        ident = ev[0] / max(ev[-1], 1e-300) > 1e-10 and ev[0] / N > 1e-10
        se = float(np.sqrt(np.linalg.inv(J)[names.index(target), names.index(target)])) if ident else np.inf
        rows.append({"mechanism": mech, "observable": obs, "reads_per_transcript": N, "target": target,
                     "identifiable": bool(ident), "target_se": se, "min_eig_per_read": float(ev[0] / N),
                     "logdet": float(np.linalg.slogdet(J + eps * N * np.eye(len(J)))[1]),
                     "n_params": len(J)})
    return rows


def retro_depth(pilots, infos):
    rows = []
    dc = [r for r in C.read_tsv(C.results_path("depth", "depth_curve.tsv"))
          if r["metric"] == "heldout_dll_read_weighted" and r["cross_replicate"] == "True"]
    for key, matched in (("gse174329_eif3b", ""), ("gse139131_unsel", "matched:gse174329_eif3b,gse139131_unsel")):
        P = pilots[key]
        for fam in FAMS:
            info = infos[(key, fam)]
            rs = [r for r in dc if r["dataset"] == key and r["model"] == fam and r["matched"] == matched]
            fr = sorted({float(r["depth_fraction"]) for r in rs})
            by = {f: [r for r in rs if float(r["depth_fraction"]) == f] for f in fr}
            dep = {f: info.depth_from_leader_reads(np.mean([float(r["n_reads"]) for r in by[f]]) / len(P.reps))
                   for f in fr}
            obs = {f: float(np.mean([float(r["value"]) for r in by[f]])) for f in fr}
            se = {f: float(np.mean([(float(r["ci_hi"]) - float(r["ci_lo"])) / (2 * Z95) for r in by[f]]))
                  for f in fr}
            rec = {f: float(np.mean([all(float(r["ci_lo"]) > 0 for r in by[f] if r["seed"] == s)
                                     for s in {r["seed"] for r in by[f]}])) for f in fr}
            for f0 in fr[:-1]:
                size, ok, pen = E.size_from_gain(info, dep[f0], P.kappa, obs[f0])
                cv2 = E.cv2_from_se(info, dep[f0], P.kappa, size, se[f0]) if size > 0 else P.cv2[fam]
                for f in fr:
                    if f <= f0:
                        continue
                    pr = E.loo_design(info, dep[f], 2, P.kappa, size=size, cv2=cv2)
                    rows.append({"dataset": key, "family": fam, "pilot_fraction": f0, "pilot_depth": dep[f0],
                                 "pilot_valid": ok, "pilot_cv2": cv2, "fraction": f, "depth": dep[f],
                                 "observed_dll": obs[f], "predicted_dll": pr["mean_dll"],
                                 "observed_recovery_both_dirs": rec[f],
                                 "predicted_power_both_dirs": pr["power_every"], "validation_status": "validated"})
    return rows


def retro_assays(pilots, infos):
    rows = []
    ac = C.read_tsv(C.results_path("missing_eval", "assay_comparison.tsv"))
    for fam in FAMS:
        obs = next(r for r in ac if r["family"] == fam and r["depth_mode"] == "observed")
        mat = [r for r in ac if r["family"] == fam and r["depth_mode"] == "matched"]
        pred = {}
        for side, key in (("selected", "gse174329_eif3b"), ("unselected", "gse139131_unsel")):
            P, info = pilots[key], infos[(key, fam)]
            frac = float(np.mean([float(r["matched_leader_target"]) for r in mat])) / float(obs[side + "_reads_common"])
            size, _, _ = E.size_from_gain(info, info.med, P.kappa, float(obs[side + "_dll"]))
            pred[side] = E.loo_design(info, info.med * frac, 2, P.kappa, size=size)["mean_dll"]
        rows.append({"family": fam, "pred_selected_matched": pred["selected"],
                     "obs_selected_matched": float(np.mean([float(r["selected_dll"]) for r in mat])),
                     "pred_unselected_matched": pred["unselected"],
                     "obs_unselected_matched": float(np.mean([float(r["unselected_dll"]) for r in mat])),
                     "pred_difference": pred["selected"] - pred["unselected"],
                     "obs_difference": float(np.mean([float(r["difference"]) for r in mat])),
                     "obs_difference_ci_excludes_0": all(float(r["ci_lo"]) > 0 or float(r["ci_hi"]) < 0 for r in mat),
                     "validation_status": "validated"})
    return rows


def r2_vs_simulation(infos, pilots):
    v2 = {}
    for r in C.read_tsv(C.results_path("theory", "theory_validation.tsv")):
        if r["direction"] == "rep1->rep2":
            v2[(r["dataset"], r["truth"], int(float(r["size_index"])), r["family"], int(float(r["depth"])))] = r
    rows = []
    for key, P in pilots.items():
        for truth, fam in (("B", "SW"), ("C", "ST"), ("D", "AUG")):
            info = infos.get((key, fam))
            for si in (0, 1, 2):
                for d in (5, 10, 20, 50, 100, 200, 500):
                    k = (key, truth, si, fam, d)
                    if info is None or k not in v2:
                        continue
                    sz = (SP.SIZE_K if fam == "AUG" else SP.SIZE_S)[si]
                    r = E.loo_design(info, d, 2, P.kappa, size=size_units(fam, sz), cv2=P.cv2[fam], shared_depth=True)
                    old = design(P, planted_gain(P, fam, sz), fam, d, 2)
                    rows.append({"dataset": key, "family": fam, "size_index": si, "depth": d,
                                 "sim_power": float(v2[k]["sim_power"]), "n_sim": int(float(v2[k]["n_sim"])),
                                 "engine_power": r["power_one_direction"],
                                 "previous_planner_power": old["power_one_direction"] if old else 0.0,
                                 "validated_theory_power": float(v2[k]["pred_power"]), "validation_status": "validated"})
    return rows


def make_figure(pilots, infos, out="experiment_design", costs=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from analysis.recoverability.make_figures import BLUE, GREY, INK, INK2, ORANGE, AQUA, save
    sel, uns = pilots["gse174329_eif3b"], pilots["gse139131_unsel"]
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 9.6))
    fig.subplots_adjust(wspace=0.45, hspace=0.8)
    iS, iU = infos[("gse174329_eif3b", "SW")], infos[("gse139131_unsel", "SW")]
    s01, s25 = size_units("SW", 0.1), size_units("SW", 0.25)
    budgets = np.geomspace(1e6, 1e10, 33)
    Rcols = {2: INK, 3: AQUA, 4: BLUE, 6: ORANGE, 8: GREY}

    a = ax[0, 0]
    for R, col in Rcols.items():
        p = [design_row(sel, iS, s01, R, B, sel.cv2["SW"])["power_pooled"] for B in budgets]
        a.plot(budgets, p, "-" if R == 2 else "--", color=col, lw=1.6, label="R = {}".format(R))
    a.axhline(POWER_OK, color=INK2, lw=0.6, ls=":")
    a.set_xscale("log")
    a.set_ylim(-0.03, 1.03)
    a.set_xlabel("total sequenced reads")
    a.set_ylabel("pooled LOO P(recovered)")
    a.set_title("a  design frontier: SW s=0.1, Sel-TCP", loc="left", color=INK)
    a.text(0.02, 0.6, "R = 2 solid: validated design\nR > 2 dashed: analytic extrapolation",
           transform=a.transAxes, fontsize=7, color=INK2)
    a.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=5, frameon=False)

    b = ax[0, 1]
    Rs = np.arange(2, 11)
    rows = [design_row(sel, iS, s01, int(R), 5e7, sel.cv2["SW"]) for R in Rs]
    b.plot(Rs, [r["power_pooled"] for r in rows], "-o", ms=3, color=BLUE, label="pooled LOO (primary)")
    b.plot(Rs, [r["power_every"] for r in rows], "-s", ms=3, color=ORANGE, label="every direction (correlated)")
    b.plot(Rs, [r["power_every_independent"] for r in rows], ":", color=GREY, label="every direction if independent (P_dir^R)")
    b2 = b.twinx()
    b2.plot(Rs, [r["loo_corr_mean"] for r in rows], "--", lw=0.8, color=INK2)
    b2.set_ylabel("LOO correlation (dashed)", color=INK2, fontsize=8)
    b2.set_ylim(-0.05, 1.0)
    b.axvspan(2.5, 10.5, color=GREY, alpha=0.07, lw=0)
    b.axhline(POWER_OK, color=INK2, lw=0.6, ls=":")
    b.set_ylim(-0.03, 1.03)
    b.set_xlabel("libraries R at 5·10^7 total reads (shaded: extrapolated)")
    b.set_ylabel("P(recovered)")
    b.set_title("b  replicates at a fixed budget", loc="left", color=INK)
    b.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=False)

    c = ax[0, 2]
    for P, info, col, lab in ((sel, iS, BLUE, "eIF3B Sel-TCP"), (uns, iU, ORANGE, "unselected 40S")):
        for R, ls in ((2, "-"), (4, "--")):
            p = [design_row(P, info, s25, R, B, P.cv2["SW"])["power_pooled"] for B in budgets]
            c.plot(budgets, p, ls, color=col, label="{} R={}{}".format(lab, R, "" if R == 2 else " (extrap.)"))
    c.axhline(POWER_OK, color=INK2, lw=0.6, ls=":")
    c.set_xscale("log")
    c.set_ylim(-0.03, 1.03)
    c.set_xlabel("total sequenced reads")
    c.set_ylabel("pooled LOO P(recovered)")
    c.set_title("c  SW s=0.25: assay at equal budget", loc="left", color=INK)
    c.text(0.98, 0.3, "leader reads per sequenced read:\nSel-TCP {:.1%}, unselected {:.2%}".format(
        sel.yield_, uns.yield_), transform=c.transAxes, va="center", ha="right", fontsize=7, color=INK2)
    c.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)

    d = ax[1, 0]
    labels, ys = [], 0
    for mech, title in (("commitment_dropoff", "commit vs drop-off"), ("leak_blur", "leak vs blur")):
        for r in score_observables(mech, N=500):
            if np.isfinite(r["target_se"]):
                d.plot(r["target_se"], ys, "o", color=BLUE)
            else:
                d.annotate("", xy=(30, ys), xytext=(12, ys), arrowprops=dict(arrowstyle="->", color=ORANGE))
                d.text(11, ys, "not identifiable ", fontsize=6.5, color=ORANGE, ha="right", va="center")
            labels.append("{}: {}".format(title, r["observable"].replace("occupancy", "40S")))
            ys += 1
        ys += 0.6
    d.set_yticks([i if i < 4 else i + 0.6 for i in range(len(labels))])
    d.set_yticklabels(labels, fontsize=7)
    d.set_ylim(ys - 0.4, -0.6)
    d.set_xscale("log")
    d.set_xlim(0.05, 30)
    d.set_xlabel("SE of the target parameter (logit), 500 reads/transcript")
    d.set_title("d  which extra observable", loc="left", color=INK)

    e = ax[1, 1]
    rd = retro_depth(pilots, infos)
    for key, col in (("gse174329_eif3b", BLUE), ("gse139131_unsel", ORANGE)):
        for valid in (True, False):
            R_ = [r for r in rd if r["dataset"] == key and r["pilot_valid"] == valid]
            if R_:
                e.plot([r["observed_dll"] for r in R_], [r["predicted_dll"] for r in R_], "o", ms=4,
                       color=col, mfc=col if valid else "none", alpha=0.85 if valid else 0.5)
    lim = [-0.03, 0.022]
    e.plot(lim, lim, color=INK2, lw=0.8)
    e.set_xlim(lim)
    e.set_ylim(lim)
    ok = [r for r in rd if r["pilot_valid"]]
    calls = np.mean([(r["predicted_power_both_dirs"] >= 0.5) == (r["observed_recovery_both_dirs"] >= 0.5) for r in ok])
    err = np.median([abs(r["predicted_dll"] - r["observed_dll"]) / max(abs(r["observed_dll"]), 1e-4) for r in ok])
    e.text(0.03, 0.97, "valid pilots (filled): median error {:.0%},\nrecovery calls {:.0%} correct\n"
           "open: pilot fails self-check".format(err, calls), transform=e.transAxes, va="top", fontsize=7, color=INK2)
    e.set_xlabel("observed held-out ΔLL/read (deeper thinning)")
    e.set_ylabel("predicted from shallower pilot")
    e.set_title("e  retrospective: pilot → deeper data (R=2)", loc="left", color=INK)

    f = ax[1, 2]
    if not costs:
        f.remove()
        return fig if out is None else save(fig, out)
    for P, info, size, col, lab in ((sel, iS, s01, BLUE, "Sel-TCP SW 0.1"), (uns, iU, s25, ORANGE, "unselected SW 0.25")):
        need = {R: reads_for_power(P, info, size, R, P.cv2["SW"]) for R in range(2, 11)}
        crow, best = cost_overlay(need)
        x = [r["R"] for r in crow if np.isfinite(r["cost_eur"])]
        y = [r["cost_eur"] for r in crow if np.isfinite(r["cost_eur"])]
        f.plot(x, y, "-o", ms=3, color=col, label=lab)
        if best:
            f.plot(best["R"], best["cost_eur"], "*", ms=12, color=col)
    f.set_yscale("log")
    f.set_xlabel("libraries R")
    f.set_ylabel("cost to reach 80% pooled power (EUR)")
    f.set_title("f  ILLUSTRATIVE costs only", loc="left", color=ORANGE)
    f.text(0.02, 0.97, "placeholder prices: €{:.0f}/library, €{:.0f}/M reads\n★ minimum under these prices only".format(
        COST_LIBRARY, COST_PER_MILLION), transform=f.transAxes, va="top", fontsize=7, color=INK2)
    f.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    return fig if out is None else save(fig, out)


def write(rows, name):
    TH.OUT, old = OUT, TH.OUT
    try:
        TH.write(rows, name)
    finally:
        TH.OUT = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    pilots = {k: Pilot(k) for k in ("gse174329_eif3b", "gse139131_unsel")}
    infos = {(k, f): E.Info(k, f) for k in pilots for f in FAMS}
    budgets = np.geomspace(1e6, 1e10, 33)
    Rs = (2, 3, 4, 5, 6, 8)
    front, need_rows, depth_rows = [], [], []
    for key, P in pilots.items():
        for fam in FAMS:
            info = infos[(key, fam)]
            effects = [("size_{}".format(x), size_units(fam, x), P.cv2[fam])
                       for x in ((0.1, 0.25, 0.5) if fam != "AUG" else (0.25, 0.5, 1.0))]
            o = observed_size(P, info, fam)
            effects.append(("observed", o["size"], o["cv2"]))
            for lab, size, cv2 in effects:
                fr = [dict(r, effect=lab) for r in frontier(P, info, size, Rs, budgets, cv2)]
                front += fr
                for R in Rs:
                    for rule in ("power_pooled", "power_every"):
                        B = need_from_frontier(fr, R, rule)
                        need_rows.append({"dataset": key, "family": fam, "effect": lab, "R": R, "rule": rule,
                                          "total_reads_needed": B, "validation_status": status(R),
                                          "bin_width_status": bin_status(info.b)})
                for rule in ("power_pooled", "power_every"):
                    B2 = need_from_frontier(fr, 2, rule)
                    dd = info.depth_from_leader_reads(B2 / 2 * P.yield_) if np.isfinite(B2) else np.inf
                    depth_rows.append({"dataset": key, "family": fam, "effect": lab, "rule": rule,
                                       "median_depth_needed_R2": dd,
                                       "sequenced_reads_per_library_R2": P.sequenced_from_depth(dd) if np.isfinite(dd) else np.inf,
                                       "validation_status": "validated", "bin_width_status": bin_status(info.b)})
    write(front, "design_frontier.tsv")
    write(need_rows, "design_reads_needed.tsv")
    write(depth_rows, "design_required_depth_R2.tsv")
    cost = []
    for key, P in pilots.items():
        for lab, x in (("size_0.1", 0.1), ("size_0.25", 0.25)):
            need = {r["R"]: r["total_reads_needed"] for r in need_rows if r["dataset"] == key and r["family"] == "SW"
                    and r["effect"] == lab and r["rule"] == "power_pooled"}
            rows, best = cost_overlay(need)
            cost += [dict(r, dataset=key, family="SW", effect=lab,
                          minimum_cost_design=bool(best and r["R"] == best["R"])) for r in rows]
    write(cost, "design_cost_overlay_ILLUSTRATIVE.tsv")
    write(sum((score_observables(m, N=n) for m in MECHANISMS for n in (100, 500, 2000)), []), "design_observables.tsv")
    write(retro_depth(pilots, infos), "design_retro_depth.tsv")
    write(retro_assays(pilots, infos), "design_retro_assays.tsv")
    vp = dict(pilots, archer_yeast=Pilot("archer_yeast"))
    vi = dict(infos)
    vi.update({("archer_yeast", f): E.Info("archer_yeast", f) for f in FAMS})
    write(r2_vs_simulation(vi, vp), "design_r2_vs_simulation.tsv")
    if not args.no_figure:
        make_figure(pilots, infos)
    C.update_manifest("experiment_design", {
        "engine": "design_engine.loo_design: summed library-specific Fisher information, exact LOO covariance",
        "primary_criterion": "pooled leave-one-library-out, read-weighted",
        "secondary_criterion": "every LOO direction (multivariate normal of correlated statistics)",
        "status": {"R=2": "validated", "R>2": "extrapolated", "costs": "illustrative",
                   "bin widths other than 10 nt": "analytic only"},
        "cost_library_eur": COST_LIBRARY, "cost_per_million_eur": COST_PER_MILLION, "cost_note": "placeholders",
        "pilots": {k: {"kappa": P.kappa, "kappa_source": P.kappa_source, "yield": P.yield_} for k, P in pilots.items()}})


if __name__ == "__main__":
    main()
