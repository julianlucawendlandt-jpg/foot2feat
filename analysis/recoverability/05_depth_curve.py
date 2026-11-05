import argparse
import importlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.recoverability import config as C
from analysis.recoverability import likelihood as L
from analysis.recoverability import splits
from analysis.recoverability.features import FeatureTable
from analysis.recoverability.panel import load_panel

EFF = importlib.import_module("analysis.recoverability.03_nested_effects")
RES = importlib.import_module("analysis.recoverability.02_replicate_resolution")

FEATURES = {"ST": L.MODELS["ST"], "SW": L.MODELS["SW"], "AUG": L.MODELS["AUG"]}


def thin(recs, reps, frac, rng):
    out = []
    for t in recs:
        r = dict(t)
        r["counts"] = {rep: rng.binomial(t["counts"][rep], frac) for rep in reps}
        out.append(r)
    return out


def agreement(recs, reps, b):
    r1, r2 = reps[:2]
    xs, ys, ax, ay, pr = [], [], [], [], []
    tot = {r: np.array([t["counts"][r][:t["msi"] + 3].sum() for t in recs]) for r in (r1, r2)}
    per = []
    for i, t in enumerate(recs):
        if tot[r1][i] < RES.MIN_READS or tot[r2][i] < RES.MIN_READS:
            continue
        x, y = RES.binned(t["counts"][r1], t["msi"], b), RES.binned(t["counts"][r2], t["msi"], b)
        if len(x) < 3:
            continue
        lc = int(np.searchsorted(RES.LEN_CLASSES, t["msi"], side="right"))
        per.append((x, y, lc))
    pm = {}
    for x, y, lc in per:
        px, py = x / x.sum(), y / y.sum()
        for k in range(len(x)):
            s = pm.setdefault((len(x) - 1 - k, lc), [0.0, 0])
            s[0] += 0.5 * (px[k] + py[k])
            s[1] += 1
    for x, y, lc in per:
        px, py = x / x.sum(), y / y.sum()
        xs.append(px - px.mean())
        ys.append(py - py.mean())
        m = np.array([pm[(len(x) - 1 - k, lc)][0] / pm[(len(x) - 1 - k, lc)][1] for k in range(len(x))])
        ax.append(px - m)
        ay.append(py - m)
        pr.append(RES.corr(x, y))
    if not per:
        return {}, 0
    return {"pooled_centered_r": RES.corr(np.concatenate(xs), np.concatenate(ys)),
            "pooled_posadj_r": RES.corr(np.concatenate(ax), np.concatenate(ay)),
            "pearson_mean": float(np.nanmean(pr))}, len(per)


def effects(recs, reps, species, b):
    from analysis.recoverability.bootstrap import delta_ll_ci
    reads = {r: EFF.leader_reads(recs, r) for r in reps}
    keep = np.nonzero(np.max([reads[r] for r in reps], axis=0) >= EFF.MIN_READS)[0]
    recs = [recs[i] for i in keep]
    reads = {r: v[keep] for r, v in reads.items()}
    ft = FeatureTable(recs)
    y = {r: ft.counts(recs, r) for r in reps}
    all_ok = np.min([reads[r] for r in reps], axis=0) >= EFF.MIN_READS
    n_folds = splits.load(species)["n_folds"]
    per, base = {}, {}
    for k in range(n_folds):
        tr_mask, te_mask = splits.train_test(ft.chrom, species, k)
        te = np.nonzero(te_mask & all_ok)[0]
        if len(te) == 0:
            continue
        for A in reps:
            tr = np.nonzero(tr_mask & (reads[A] >= EFF.MIN_READS))[0]
            if len(tr) == 0:
                continue
            p0 = L.Problem(ft, tr, b, [])
            f0 = L.fit(p0, p0.bin_counts(y[A]), prior_sd=EFF.PRIOR_SD)
            q0 = L.Problem(ft, te, b, [])
            ll0 = {B: L.loglik(f0, q0, q0.bin_counts(y[B]), per_transcript=True) for B in reps}
            for feat, blocks in FEATURES.items():
                pt = L.Problem(ft, tr, b, blocks)
                f = L.fit(pt, pt.bin_counts(y[A]), prior_sd=EFF.PRIOR_SD, init=f0.as_dict())
                pe = L.Problem(ft, te, b, blocks)
                for B in reps:
                    ll, n = L.loglik(f, pe, pe.bin_counts(y[B]), per_transcript=True)
                    d = per.setdefault((feat, A, B), [[], []])
                    d[0].append(ll - ll0[B][0])
                    d[1].append(n)
    out = {}
    for key_, (dl, nl) in per.items():
        out[key_] = delta_ll_ci(np.concatenate(dl), np.concatenate(nl), n_boot=500, seed=0)
    med = float(np.median(np.concatenate([reads[r] for r in reps])))
    return out, len(recs), med


def run(key, fractions, seeds, b, target_total=None, tag=""):
    d = C.DATASETS[key]
    recs, meta = load_panel(key)
    reps = sorted(meta["replicates"])
    splits.load(d["species"])
    rows, t0 = [], time.time()
    totals = {r: sum(int(t["counts"][r][:t["msi"] + 3].sum()) for t in recs) for r in reps}
    for frac in fractions:
        for seed in seeds:
            rng = np.random.default_rng(1000 + seed)
            if target_total is not None:
                th = []
                for t in recs:
                    r = dict(t)
                    r["counts"] = {rep: rng.binomial(t["counts"][rep],
                                                     min(1.0, frac * target_total / totals[rep]))
                                   for rep in reps}
                    th.append(r)
            else:
                th = recs if frac >= 1.0 else thin(recs, reps, frac, rng)
            base = {"dataset": key, "split": "chrom_heldout", "seed": seed, "bin_width": b,
                    "depth_fraction": frac, "matched": tag}
            leader_total = sum(int(t["counts"][r][:t["msi"] + 3].sum()) for t in th for r in reps)
            if len(reps) >= 2:
                ag, n_ag = agreement(th, reps, b)
                for m, v in ag.items():
                    rows.append(dict(base, replicate="{}_vs_{}".format(*reps[:2]), model="",
                                     metric=m, value=v, n_transcripts=n_ag, n_reads=leader_total))
            eff, n_tx, med = effects(th, reps, d["species"], b)
            for (feat, A, B), ci in eff.items():
                rows.append(dict(base, replicate="{}->{}".format(A, B), model=feat,
                                 metric="heldout_dll_read_weighted", value=ci["read_weighted"],
                                 ci_lo=ci["rw_lo"], ci_hi=ci["rw_hi"],
                                 n_transcripts=ci["n_transcripts"], n_reads=leader_total,
                                 median_reads_per_transcript=med, cross_replicate=A != B))
            print("[depth] {} frac {:.2f} seed {}: {} tx, median {:.0f} reads ({:.0f}s)".format(
                key, frac, seed, n_tx, med, time.time() - t0))
            if frac >= 1.0 and target_total is None:
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(C.DATASETS))
    ap.add_argument("--matched", help="comma-separated datasets thinned to a common leader total")
    ap.add_argument("--fractions", default="0.1,0.25,0.5,0.75,1.0")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--bin", type=int, default=10)
    args = ap.parse_args()
    C.ensure_layout()
    fr = [float(x) for x in args.fractions.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    rows = []
    if args.matched:
        keys = args.matched.split(",")
        tot = []
        for k in keys:
            recs, meta = load_panel(k)
            tot += [sum(int(t["counts"][r][:t["msi"] + 3].sum()) for t in recs)
                    for r in meta["replicates"]]
        target = min(tot)
        print("[depth] matched leader total per replicate: {}".format(target))
        for k in keys:
            rows += run(k, fr, seeds, args.bin, target_total=target, tag="matched:" + args.matched)
    else:
        rows += run(args.dataset, fr, seeds, args.bin)
    out = C.results_path("depth", "depth_curve.tsv")
    done = {(r["dataset"], r.get("matched", "")) for r in rows}
    C.merge_tsv(out, rows, lambda r: (r["dataset"], r.get("matched", "")) not in done)
    print("[depth] -> {}".format(out))


if __name__ == "__main__":
    main()
