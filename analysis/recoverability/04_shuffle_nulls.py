import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.recoverability import config as C
from analysis.recoverability import likelihood as L
from analysis.recoverability import splits
from analysis.recoverability.features import START_HALF

STRATUM_BIN = 10
DIST_BIN_START = 25

NULL_SPECS = {
    "SW": (["sw"], ["sw"], "sequence"),
    "ST": (["paired", "dG"], ["paired", "dG"], "structure"),
    "AUG": (["start", "m3", "p4", "kozak"], ["start", "m3", "p4", "kozak"], "start"),
    "S1": (["nt1"], ["nt1"], "sequence"),
    "S3": (["k3"], ["k3"], "sequence"),
    "S5": (["k5"], ["k5"], "sequence"),
    "ST_paired": (["paired"], ["paired"], "structure"),
    "ST_dG": (["dG"], ["dG"], "structure"),
    "UORF": (["start", "m3", "p4", "kozak", "nuaug", "in_uorf", "uorf_geom"],
             ["nuaug", "in_uorf", "uorf_geom"], "sequence"),
}
NULL_BLOCKS = {k: v[0] for k, v in NULL_SPECS.items()}


def _permute_within(groups, rng, n):
    perm = np.arange(n)
    order = np.argsort(groups, kind="stable")
    g = groups[order]
    cuts = np.nonzero(np.diff(g))[0] + 1
    for block in np.split(order, cuts):
        perm[block] = rng.permutation(block)
    return perm


def position_strata(ft):
    da = np.clip(ft.dAUG, -10, L.POS_MAX) // STRATUM_BIN
    d5 = np.clip(ft.d5, 0, L.POS_MAX) // STRATUM_BIN
    return (da + 20) * 1000 + d5


def perm_sequence(ft, rng):
    return _permute_within(position_strata(ft), rng, len(ft.seg))


def perm_structure(ft, rng):
    q = np.quantile(ft.gc, [0.2, 0.4, 0.6, 0.8])
    gcq = np.searchsorted(q, ft.gc)
    return _permute_within(position_strata(ft) * 10 + gcq, rng, len(ft.seg))


def perm_start_units(ft, rng):
    perm = np.arange(len(ft.seg))
    centers = np.nonzero(ft.start_center)[0]
    d = ft.dAUG[centers]
    group = np.where(d == 0, -1, d // DIST_BIN_START)
    if len(centers) == 0:
        return perm
    target = centers[_permute_within(group, rng, len(centers))]
    lo_tx = ft.tx_start[ft.seg[centers]]
    hi_tx = ft.tx_start[ft.seg[centers] + 1]
    lo_tg = ft.tx_start[ft.seg[target]]
    hi_tg = ft.tx_start[ft.seg[target] + 1]
    for o in sorted(range(-START_HALF, START_HALF + 1), key=lambda o: -abs(o)):
        src, dst = centers + o, target + o
        ok = (src >= lo_tx) & (src < hi_tx) & (dst >= lo_tg) & (dst < hi_tg)
        perm[src[ok]] = dst[ok]
    return perm


PERMUTERS = {"sequence": perm_sequence, "structure": perm_structure, "start": perm_start_units}


def heldout_dll(ft, y, reps, reads, chroms, species, blocks, b, min_reads, prior_sd,
                permute=None, cache=None):
    n_folds = splits.load(species)["n_folds"]
    all_ok = np.min([reads[r] for r in reps], axis=0) >= min_reads
    acc = {}
    for k in range(n_folds):
        tr_mask, te_mask = splits.train_test(chroms, species, k)
        te = np.nonzero(te_mask & all_ok)[0]
        if len(te) == 0:
            continue
        for A in reps:
            tr = np.nonzero(tr_mask & (reads[A] >= min_reads))[0]
            key = (k, A)
            if key not in cache:
                p0 = L.Problem(ft, tr, b, [])
                f0 = L.fit(p0, p0.bin_counts(y[A]), prior_sd=prior_sd)
                q0 = L.Problem(ft, te, b, [])
                cache[key] = (f0, {B: L.loglik(f0, q0, q0.bin_counts(y[B]), per_transcript=True)
                                   for B in reps})
            f0, ll0 = cache[key]
            pt = L.Problem(ft, tr, b, blocks, permute=permute)
            f = L.fit(pt, pt.bin_counts(y[A]), prior_sd=prior_sd, init=f0.as_dict())
            pe = L.Problem(ft, te, b, blocks, permute=permute)
            for B in reps:
                ll, n = L.loglik(f, pe, pe.bin_counts(y[B]), per_transcript=True)
                s = acc.setdefault((A, B), [0.0, 0.0])
                s[0] += float((ll - ll0[B][0]).sum())
                s[1] += float(n.sum())
    return {k: v[0] / v[1] for k, v in acc.items()}


_W = {}


def _worker_init(key):
    eff = __import__("analysis.recoverability.03_nested_effects", fromlist=["x"])
    recs, meta, reps, reads, ft, y = eff.prepare(key)
    _W.update(eff=eff, reps=reps, reads=reads, ft=ft, y=y,
              species=C.DATASETS[key]["species"], cache={})


def _one(task):
    feat, i, b, seed = task
    W = _W
    blocks, permuted, kind = NULL_SPECS[feat]
    permute = None
    if i >= 0:
        rng = np.random.default_rng([seed, list(NULL_SPECS).index(feat), i])
        perm = PERMUTERS[kind](W["ft"], rng)
        permute = {blk: perm for blk in permuted}
    res = heldout_dll(W["ft"], W["y"], W["reps"], W["reads"], W["ft"].chrom, W["species"],
                      blocks, b, W["eff"].MIN_READS, W["eff"].PRIOR_SD,
                      permute=permute, cache=W["cache"])
    return feat, i, res


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(C.DATASETS))
    ap.add_argument("--features", default="SW,ST,AUG")
    ap.add_argument("--bin", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=C.N_NULL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel worker processes; default sized to cores and RAM")
    args = ap.parse_args()
    args.workers = args.workers or C.default_workers()
    C.ensure_layout()
    eff = __import__("analysis.recoverability.03_nested_effects", fromlist=["x"])
    d = C.DATASETS[args.dataset]
    recs, meta, reps, reads, ft, y = eff.prepare(args.dataset)
    eff.ensure_folds(d["species"], recs, reps, args.dataset)
    del recs, ft, y
    feats = [f for f in args.features.split(",") if f]
    tasks = [(f, i, args.bin, args.seed) for f in feats for i in range(-1, args.n_perm)]
    real, null, draws = {}, {}, []
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_worker_init, initargs=(args.dataset,)) as pool:
        for n_done, (feat, i, res) in enumerate(pool.imap_unordered(_one, tasks), 1):
            for k, v in res.items():
                if i < 0:
                    real[(feat, k)] = v
                else:
                    null.setdefault((feat, k), []).append(v)
                    draws.append({"dataset": args.dataset, "replicate": "{}->{}".format(*k),
                                  "split": "chrom_heldout", "seed": args.seed, "bin_width": args.bin,
                                  "model": feat, "metric": "null_heldout_dll_read_weighted",
                                  "value": v, "perm": i})
            if n_done % 20 == 0:
                print("[null] {}/{} fits-sets ({:.0f}s)".format(n_done, len(tasks), time.time() - t0),
                      flush=True)
    rows = []
    for (feat, k), v in sorted(real.items()):
        nv = np.array(null.get((feat, k), []))
        sd = nv.std(ddof=1) if len(nv) > 1 else np.nan
        base = {"dataset": args.dataset, "replicate": "{}->{}".format(*k),
                "split": "chrom_heldout", "seed": args.seed, "bin_width": args.bin,
                "model": feat, "n_perm": len(nv), "cross_replicate": k[0] != k[1]}
        p = float((1 + (nv >= v).sum()) / (1 + len(nv)))
        rows += [dict(base, metric="real_dll", value=v),
                 dict(base, metric="null_mean", value=float(nv.mean())),
                 dict(base, metric="null_sd", value=float(sd)),
                 dict(base, metric="empirical_p", value=p),
                 dict(base, metric="z", value=float((v - nv.mean()) / sd) if sd > 0 else np.nan)]
        print("[null] {} {}: real {:.5f}  null {:.5f} +/- {:.5f}  p={:.3f}".format(
            feat, base["replicate"], v, nv.mean(), sd, p))
    out = C.results_path("nulls", "shuffle_nulls.tsv")
    done = {(r["dataset"], r["model"], str(r["bin_width"])) for r in rows}
    C.merge_tsv(out, rows, lambda r: (r["dataset"], r["model"], str(r["bin_width"])) not in done)
    C.write_tsv(draws, C.results_path("nulls", "null_draws_{}_b{}.tsv".format(args.dataset, args.bin)))
    C.update_manifest("nulls_{}_b{}".format(args.dataset, args.bin),
                      {"n_perm": args.n_perm, "features": feats, "seed": args.seed,
                       "workers": args.workers, "stratum_bin": STRATUM_BIN,
                       "start_distance_bin": DIST_BIN_START})
    print("[null] -> {}".format(out))


if __name__ == "__main__":
    main()
