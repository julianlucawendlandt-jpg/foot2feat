import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.recoverability import config as C
from analysis.recoverability import likelihood as L
from analysis.recoverability import splits
from analysis.recoverability.bootstrap import delta_ll_ci
from analysis.recoverability.features import FeatureTable
from analysis.recoverability.panel import load_panel

MIN_READS = 10
PRIOR_SD = 1.0


def leader_reads(recs, rep):
    return np.array([int(t["counts"][rep][:t["msi"] + 3].sum()) for t in recs])


def ensure_folds(species, recs, reps, key):
    path = C.results_path(splits.FOLD_FILES[species])
    if os.path.exists(path):
        return splits.load(species)
    depth = {}
    for t in recs:
        n = sum(int(t["counts"][r][:t["msi"] + 3].sum()) for r in reps)
        depth[t["chrom"]] = depth.get(t["chrom"], 0) + n
    out = splits.freeze(species, depth, "leader reads of {} ({} summed)".format(key, "+".join(reps)))
    print("[effects] froze {} folds -> {}".format(species, path))
    return out


def prepare(key, cds=False):
    recs, meta = load_panel(key)
    reps = sorted(meta["replicates"])
    reads = {r: leader_reads(recs, r) for r in reps}
    keep = np.nonzero(np.max([reads[r] for r in reps], axis=0) >= MIN_READS)[0]
    recs = [recs[i] for i in keep]
    reads = {r: v[keep] for r, v in reads.items()}
    ft = FeatureTable(recs, cds=cds)
    y = {r: ft.counts(recs, r) for r in reps}
    return recs, meta, reps, reads, ft, y


def run(key, models, bins, cds=False, seed=0, n_boot=C.N_BOOT):
    d = C.DATASETS[key]
    t0 = time.time()
    recs, meta, reps, reads, ft, y = prepare(key, cds=cds)
    folds = ensure_folds(d["species"], recs, reps, key)
    chroms = ft.chrom
    print("[effects] {}: {} transcripts with >= {} reads in some replicate, P = {} positions "
          "({:.0f}s)".format(key, len(recs), MIN_READS, len(ft.seg), time.time() - t0))

    all_ok = np.min([reads[r] for r in reps], axis=0) >= MIN_READS
    heldout = {}
    train_rows, coef_rows = [], []
    for k in range(folds["n_folds"]):
        tr_mask, te_mask = splits.train_test(chroms, d["species"], k)
        te = np.nonzero(te_mask & all_ok)[0]
        for A in reps:
            tr = np.nonzero(tr_mask & (reads[A] >= MIN_READS))[0]
            for b in bins:
                base_fit, base_train = None, None
                seen = {}
                for name in models:
                    blocks = L.MODELS[name]
                    pt = L.Problem(ft, tr, b, blocks)
                    yb = pt.bin_counts(y[A])
                    t1 = time.time()
                    init = dict(base_fit.as_dict()) if base_fit else {}
                    init.update({k: v for k, v in seen.items() if not k.startswith("pos_")})
                    f = L.fit(pt, yb, prior_sd=PRIOR_SD, init=init or None)
                    seen.update(f.as_dict())
                    if name == "P":
                        base_fit = f
                        base_train = f.info["train_ll"]
                    train_rows.append({"dataset": key, "replicate": A, "split": "fold{}_train".format(k),
                                       "seed": seed, "bin_width": b, "model": name,
                                       "n_transcripts": f.info["n_transcripts"],
                                       "n_reads": f.info["n_reads"],
                                       "metric": "train_dll_per_read",
                                       "value": (f.info["train_ll"] - base_train) / f.info["n_reads"],
                                       "iterations": f.info["iterations"],
                                       "converged": f.info["converged"],
                                       "fit_seconds": round(time.time() - t1, 2)})
                    for blk, s in f.slices.items():
                        if blk.startswith("pos_") and name != "P":
                            continue
                        for lvl, v in enumerate(f.beta[s]):
                            coef_rows.append({"dataset": key, "replicate": A, "split": "fold{}".format(k),
                                              "seed": seed, "bin_width": b, "model": name,
                                              "block": blk, "level": lvl, "value": float(v)})
                    if len(te) == 0:
                        continue
                    pe = L.Problem(ft, te, b, blocks)
                    for B in reps:
                        ll, n = L.loglik(f, pe, pe.bin_counts(y[B]), per_transcript=True)
                        heldout.setdefault((A, B, b, name), []).append((te, ll, n))
                print("[effects] fold {} train {} b={:>2}: {} train tx, {} test tx ({:.0f}s)".format(
                    k, A, b, len(tr), len(te), time.time() - t0))

    rows = []
    store = {}
    for (A, B, b, name), parts in heldout.items():
        tx = np.concatenate([p[0] for p in parts])
        ll = np.concatenate([p[1] for p in parts])
        n = np.concatenate([p[2] for p in parts])
        store["{}|{}|{}|{}".format(A, B, b, name)] = np.stack([tx, ll, n])
    for (A, B, b, name), _ in heldout.items():
        if name == "P":
            continue
        m = store["{}|{}|{}|{}".format(A, B, b, name)]
        p = store["{}|{}|{}|P".format(A, B, b)]
        assert np.array_equal(m[0], p[0])
        ci = delta_ll_ci(m[1] - p[1], m[2], n_boot=n_boot, seed=seed)
        direction = "{}->{}".format(A, B)
        base = {"dataset": key, "replicate": direction, "split": "chrom_heldout",
                "seed": seed, "bin_width": b, "model": name,
                "n_transcripts": ci["n_transcripts"], "n_reads": ci["n_reads"],
                "cross_replicate": A != B}
        rows.append(dict(base, metric="heldout_dll_read_weighted", value=ci["read_weighted"],
                         ci_lo=ci["rw_lo"], ci_hi=ci["rw_hi"], p_le_0=ci["p_rw_le_0"]))
        rows.append(dict(base, metric="heldout_dll_equal_transcript", value=ci["equal_transcript"],
                         ci_lo=ci["et_lo"], ci_hi=ci["et_hi"]))
    agg = {}
    for r in train_rows:
        agg.setdefault((r["replicate"], r["bin_width"], r["model"]), []).append(r["value"])
    for (A, b, name), v in agg.items():
        rows.append({"dataset": key, "replicate": A, "split": "chrom_train_mean", "seed": seed,
                     "bin_width": b, "model": name, "metric": "train_dll_per_read",
                     "value": float(np.mean(v)), "n_transcripts": "", "n_reads": ""})

    out = C.results_path("effects", "nested_effects.tsv")
    tag = key + ("_cds" if cds else "")
    for r in rows:
        r["panel"] = tag
    C.merge_tsv(out, rows, lambda r: r.get("panel", r["dataset"]) != tag)
    C.write_tsv(train_rows, C.results_path("effects", "fits_{}.tsv".format(tag)))
    C.write_tsv(coef_rows, C.results_path("effects", "coefficients_{}.tsv".format(tag)))
    np.savez_compressed(C.results_path("effects", "heldout_ll_{}.npz".format(tag)),
                        ids=np.array([t["transcript_id"] for t in recs]), **store)
    C.update_manifest("effects_" + tag, {
        "min_reads": MIN_READS, "prior_sd": PRIOR_SD, "models": {m: L.MODELS[m] for m in models},
        "bins": list(bins), "pos_max": L.POS_MAX, "cds_extension_included": cds,
        "folds": splits.FOLD_FILES[d["species"]], "seconds": round(time.time() - t0)})
    print("[effects] -> {} ({:.0f}s)".format(out, time.time() - t0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(C.DATASETS))
    ap.add_argument("--models", default=",".join(L.MODELS))
    ap.add_argument("--bins", default=",".join(str(b) for b in C.MODEL_BINS))
    ap.add_argument("--cds", action="store_true",
                    help="also model the stored CDS extension (sensitivity)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    C.ensure_layout()
    models = [m for m in args.models.split(",") if m]
    if models[0] != "P":
        models = ["P"] + [m for m in models if m != "P"]
    run(args.dataset, models, [int(b) for b in args.bins.split(",")], cds=args.cds, seed=args.seed)


if __name__ == "__main__":
    main()
