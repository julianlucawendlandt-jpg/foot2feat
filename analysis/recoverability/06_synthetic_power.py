import argparse
import importlib
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.recoverability import config as C
from analysis.recoverability import likelihood as L
from analysis.recoverability import splits
from analysis.recoverability.bootstrap import delta_ll_ci
from data.synthetic import calibrate_concentration, draw_counts

EFF = importlib.import_module("analysis.recoverability.03_nested_effects")

SIZE_S = (0.1, 0.25, 0.5)
SIZE_K = (0.25, 0.5, 1.0)
FAMILIES = {"SW": L.MODELS["SW"], "ST": L.MODELS["ST"], "AUG": L.MODELS["AUG"]}
PLANTED = {"A": [], "B": ["SW"], "C": ["ST"], "D": ["AUG"], "E": ["ST", "AUG"]}


def planted_components(ft, size, seed=12345):
    s, k = SIZE_S[size], SIZE_K[size]
    rng = np.random.default_rng(seed)
    pat = rng.normal(0, 1, (21, 4))
    pat -= pat.mean(1, keepdims=True)
    sw = ft.cat["sw"]
    flat = np.concatenate([pat.reshape(-1), [0.0]])
    e_sw = flat[np.where(sw < 0, len(flat) - 1, sw)].sum(1)
    scale = s / e_sw.std()
    e_sw *= scale
    e_st = s * ft.cont["paired"][:, 1]
    kz = ft.cat["kozak"][:, 0]
    e_aug = np.where(kz == 0, k, np.where(kz == 2, -k, 0.0))
    return {"SW": (e_sw, {"sw": (pat * scale).reshape(-1)}),
            "ST": (e_st, {"paired": np.array([0.0, s]), "dG": np.zeros(2)}),
            "AUG": (e_aug, {"kozak": np.array([k, 0.0, -k])})}


def coef_agreement(fit, truth):
    a, b = [], []
    for blk, v in truth.items():
        if blk not in fit.slices:
            continue
        f = fit.coef(blk)
        if blk == "sw":
            f = (f.reshape(21, 4) - f.reshape(21, 4).mean(1, keepdims=True)).reshape(-1)
        elif blk == "kozak":
            f, v = f - f.mean(), v - v.mean()
        a.append(f)
        b.append(v)
    if not a:
        return np.nan
    a, b = np.concatenate(a), np.concatenate(b)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def sign_ok(fit, fam):
    if fam == "ST":
        return float(fit.coef("paired")[1] > 0)
    if fam == "AUG":
        kz = fit.coef("kozak")
        return float(kz[0] > kz[2])
    return np.nan


def analyse(ft, y, reads, species, b, truth_coefs, families, folds_used):
    all_ok = np.minimum(reads["rep1"], reads["rep2"]) >= EFF.MIN_READS
    res, base = {}, {}
    for fam in families:
        blocks = FAMILIES[fam]
        per, coef_r, signs = {}, [], []
        for k in folds_used:
            tr_mask, te_mask = splits.train_test(ft.chrom, species, k)
            te = np.nonzero(te_mask & all_ok)[0]
            tr = np.nonzero(tr_mask & (reads["rep1"] >= EFF.MIN_READS))[0]
            if len(te) == 0 or len(tr) == 0:
                continue
            if k not in base:
                p0 = L.Problem(ft, tr, b, [])
                f0 = L.fit(p0, p0.bin_counts(y["rep1"]), prior_sd=EFF.PRIOR_SD)
                q0 = L.Problem(ft, te, b, [])
                base[k] = (f0, {B: L.loglik(f0, q0, q0.bin_counts(y[B]), per_transcript=True)
                                for B in ("rep1", "rep2")})
            f0, ll0 = base[k]
            pt = L.Problem(ft, tr, b, blocks)
            f = L.fit(pt, pt.bin_counts(y["rep1"]), prior_sd=EFF.PRIOR_SD, init=f0.as_dict())
            if fam in truth_coefs:
                coef_r.append(coef_agreement(f, truth_coefs[fam]))
                signs.append(sign_ok(f, fam))
            pe = L.Problem(ft, te, b, blocks)
            for B in ("rep1", "rep2"):
                ll, n = L.loglik(f, pe, pe.bin_counts(y[B]), per_transcript=True)
                dd = per.setdefault(B, [[], []])
                dd[0].append(ll - ll0[B][0])
                dd[1].append(n)
        for B, (dl, nl) in per.items():
            ci = delta_ll_ci(np.concatenate(dl), np.concatenate(nl), n_boot=300, seed=0)
            res[(fam, ("rep1", B))] = {"dll": ci["read_weighted"], "lo": ci["rw_lo"],
                                       "hi": ci["rw_hi"],
                                       "coef_r": np.nanmean(coef_r) if coef_r else np.nan,
                                       "sign_ok": np.nanmean(signs) if signs else np.nan,
                                       "n_tx": ci["n_transcripts"]}
    return res


_S = {}


def _sim_init(key, b, noise, folds_arg):
    d = C.DATASETS[key]
    recs, meta, reps_real, reads_real, ft, y_real = EFF.prepare(key)
    n_folds = splits.load(d["species"])["n_folds"]
    folds_used = (list(range(n_folds)) if folds_arg == "all"
                  else [int(x) for x in folds_arg.split(",")])
    r0 = reps_real[0]
    pp = L.Problem(ft, np.arange(len(recs)), 1, [])
    fpos = L.fit(pp, pp.bin_counts(y_real[r0]), prior_sd=EFF.PRIOR_SD)
    eta_pos = np.zeros(len(ft.seg))
    eta_pos[pp.pos] = pp.X @ fpos.beta
    alpha = (C.measured("synthetic_dm_alpha_" + key, {}).get("alpha")
             if noise == "dm" else None)
    _S.update(ft=ft, n=len(recs), eta_pos=eta_pos,
              depth_real=reads_real[r0].astype(float), species=d["species"], b=b,
              noise=noise, alpha=alpha, folds_used=folds_used)


def _sim_one(task):
    truth, size, target, sim = task
    S = _S
    ft = S["ft"]
    comps = planted_components(ft, size)
    eta = S["eta_pos"].copy()
    truth_coefs = {}
    for fam in PLANTED[truth]:
        eta += comps[fam][0]
        truth_coefs[fam] = comps[fam][1]
    rng = np.random.default_rng(zlib.crc32("{}|{}|{}|{}|{}".format(
        truth, size, target, sim, S["noise"]).encode()))
    depth = rng.poisson(S["depth_real"] * target / np.median(S["depth_real"]))
    y = {}
    for rep in ("rep1", "rep2"):
        v = np.zeros(len(ft.seg))
        for t in range(S["n"]):
            a, e = ft.tx_start[t], ft.tx_start[t + 1]
            v[a:e] = draw_counts(eta[a:e], depth[t], rng, S["alpha"])
        y[rep] = v
    reads = {rep: np.bincount(ft.seg, weights=y[rep], minlength=S["n"]) for rep in y}
    families = list(FAMILIES) if truth == "A" else PLANTED[truth]
    return task, analyse(ft, y, reads, S["species"], S["b"], truth_coefs, families,
                         S["folds_used"])


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def power_curve_rows(rows, key, noise, b):
    sims = {}
    for r in rows:
        k = (r["truth"], r["model"], int(float(r["size_index"])),
             int(float(r["target_median_depth"])), int(float(r["seed"])), r["replicate"])
        sims.setdefault(k, {})[r["metric"]] = float(r["value"])
    out = []
    for direction, suffix in (("rep1->rep2", ""), ("rep1->rep1", "_within")):
        null_vals = {}
        for (truth, fam, _size, depth, _seed, rep), m in sims.items():
            if truth == "A" and rep == direction:
                null_vals.setdefault((fam, depth), []).append(m["dll"])
        null_failed = {k: int(np.sum(~np.isfinite(v))) for k, v in null_vals.items()}
        null_vals = {k: [x for x in v if np.isfinite(x)] for k, v in null_vals.items()}
        groups = {}
        for (truth, fam, size, depth, _seed, rep), m in sims.items():
            if truth == "A" or rep != direction or fam not in PLANTED[truth]:
                continue
            q = null_vals.get((fam, depth), [])
            thr = float(np.percentile(q, 95)) if q else np.nan
            g = groups.setdefault((truth, fam, size, depth),
                                  {"ok": [], "dll": [], "coef_r": [], "sign": [],
                                   "thr": thr, "n_null": len(q),
                                   "n_null_failed": null_failed.get((fam, depth), 0),
                                   "n_failed": 0})
            if not (np.isfinite(m["dll"]) and np.isfinite(m["lo"])):
                g["n_failed"] += 1
                continue
            ok = m["lo"] > 0 and (np.isnan(thr) or m["dll"] > thr)
            g["ok"].append(ok)
            g["dll"].append(m["dll"])
            g["coef_r"].append(m["coef_r"])
            g["sign"].append(m["sign_ok"])
        for (truth, fam, size, depth), g in sorted(groups.items()):
            n_ok, n = int(np.sum(g["ok"])), len(g["ok"])
            lo, hi = wilson(n_ok, n)
            base = {"dataset": "synthetic:" + key, "replicate": direction,
                    "split": "chrom_heldout", "seed": "", "bin_width": b, "model": fam,
                    "truth": truth, "size_index": size, "effect_s": SIZE_S[size],
                    "effect_k": SIZE_K[size], "target_median_depth": depth, "noise": noise,
                    "n_sim": n, "null_threshold": g["thr"], "n_null_sims": g["n_null"],
                    "n_sim_failed": g["n_failed"], "n_null_failed": g["n_null_failed"]}
            out += [dict(base, metric="p_recovered" + suffix, value=n_ok / n if n else np.nan,
                         p_lo=lo, p_hi=hi, n_recovered=n_ok),
                    dict(base, metric="mean_dll" + suffix, value=float(np.mean(g["dll"]))),
                    dict(base, metric="mean_coef_corr" + suffix,
                         value=float(np.nanmean(g["coef_r"])) if np.isfinite(g["coef_r"]).any() else np.nan),
                    dict(base, metric="sign_recovery" + suffix,
                         value=float(np.nanmean(g["sign"])) if np.isfinite(g["sign"]).any() else np.nan)]
    return out


def ensure_alpha(key):
    if "synthetic_dm_alpha_" + key in C.load_manifest():
        return
    recs, meta, reps, reads, ft, y = EFF.prepare(key)
    if len(reps) < 2:
        sys.exit("--noise dm needs a dataset with two replicates")
    pairs = [(t["counts"][reps[0]][:t["msi"] + 3], t["counts"][reps[1]][:t["msi"] + 3])
             for t in recs]
    alpha, obs, sim = calibrate_concentration(pairs, np.random.default_rng(0))
    print("[synth] Dirichlet-multinomial alpha {:.2f} (dispersion ratio {:.2f}, "
          "simulated {:.2f})".format(alpha, obs, sim), flush=True)
    C.update_manifest("synthetic_dm_alpha_" + key,
                      {"alpha": alpha, "observed_ratio": obs, "simulated_ratio": sim,
                       "resolution": "1 nt, leader through main AUG"})


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gse174329_eif3b", choices=sorted(C.DATASETS))
    ap.add_argument("--truths", default="A,B,C,D,E")
    ap.add_argument("--depths", default="5,10,20,50,100,200,500")
    ap.add_argument("--sizes", default="0,1,2")
    ap.add_argument("--n-sim", type=int, default=3)
    ap.add_argument("--n-null-sim", type=int, default=6,
                    help="truth-A simulations per depth (they define the null)")
    ap.add_argument("--noise", choices=("multinomial", "dm"), default="multinomial")
    ap.add_argument("--bin", type=int, default=10)
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel worker processes; default sized to cores and RAM")
    ap.add_argument("--folds", default="all",
                    help="'all' or comma-separated fold indices (cost control; recorded)")
    ap.add_argument("--tag", default="",
                    help="suffix for the output files and manifest section, so a new run "
                         "never overwrites a frozen one (e.g. --tag n50)")
    args = ap.parse_args()
    args.workers = args.workers or C.default_workers()
    C.ensure_layout()
    t0 = time.time()
    key = args.dataset
    if args.noise == "dm":
        ensure_alpha(key)
    depths = [int(x) for x in args.depths.split(",")]
    tasks = []
    for truth in args.truths.split(","):
        if truth == "A":
            tasks += [("A", 0, dp, sim) for dp in depths for sim in range(args.n_null_sim)]
            continue
        for size in [int(x) for x in args.sizes.split(",")]:
            tasks += [(truth, size, dp, sim) for dp in depths for sim in range(args.n_sim)]
    stem = "{}_{}{}".format(key, args.noise, "_" + args.tag if args.tag else "")
    out = C.results_path("synthetic", "power_{}.tsv".format(stem))
    rows = []
    with mp.get_context("spawn").Pool(args.workers, initializer=_sim_init,
                 initargs=(key, args.bin, args.noise, args.folds)) as pool:
        for n_done, ((truth, size, target, sim), res) in enumerate(
                pool.imap_unordered(_sim_one, tasks), 1):
            for (fam, (A, B)), v in res.items():
                base = {"dataset": "synthetic:" + key, "replicate": "{}->{}".format(A, B),
                        "split": "chrom_heldout", "seed": sim, "bin_width": args.bin,
                        "model": fam, "n_transcripts": v["n_tx"], "truth": truth,
                        "planted": fam in PLANTED[truth], "size_index": size,
                        "effect_s": SIZE_S[size], "effect_k": SIZE_K[size],
                        "target_median_depth": target, "noise": args.noise,
                        "cross_replicate": A != B}
                for m in ("dll", "lo", "hi", "coef_r", "sign_ok"):
                    rows.append(dict(base, metric=m, value=v[m]))
            print("[synth] {}/{} truth {} size {} depth {} sim {} ({:.0f}s)".format(
                n_done, len(tasks), truth, size, target, sim, time.time() - t0), flush=True)
            if n_done % 10 == 0:
                C.write_tsv(rows, out)
    C.write_tsv(rows, out)
    curve = power_curve_rows(C.read_tsv(out), key, args.noise, args.bin)
    C.write_tsv(curve, C.results_path("synthetic", "power_curve_{}.tsv".format(stem)))
    C.update_manifest("synthetic_" + stem, {
        "truths": args.truths, "depths": args.depths, "sizes": args.sizes,
        "SIZE_S": SIZE_S, "SIZE_K": SIZE_K, "n_sim": args.n_sim,
        "n_null_sim": args.n_null_sim, "bin": args.bin, "folds": args.folds,
        "null": "95th percentile of truth-A held-out dLL at the same depth",
        "training_direction": "rep1 only, scored in rep1 and rep2",
        "dm_alpha": C.measured("synthetic_dm_alpha_" + key, {}).get("alpha")
        if args.noise == "dm" else None,
        "binomial_ci": "Wilson 95%", "command": " ".join(sys.argv)})
    print("[synth] -> {} ({:.0f}s)".format(out, time.time() - t0))


if __name__ == "__main__":
    main()
