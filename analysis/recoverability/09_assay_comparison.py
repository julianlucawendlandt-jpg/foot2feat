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
NUL = importlib.import_module("analysis.recoverability.04_shuffle_nulls")

FAMILIES = ("SW", "ST", "AUG")
N_BOOT = 2000
OUT_DIR = "missing_eval"


def _stable_tag(s):
    import zlib
    return zlib.crc32(s.encode())


def load(key):
    recs, meta = load_panel(key)
    return {t["transcript_id"]: t for t in recs}, sorted(meta["replicates"])


def thin_dataset(recs_by_id, ids, reps, target, seed, key):
    recs = [recs_by_id[i] for i in ids]
    totals = {r: sum(int(t["counts"][r][:t["msi"] + 3].sum()) for t in recs) for r in reps}
    out = []
    for j, r in enumerate(reps):
        rng = np.random.default_rng([seed, _stable_tag(key), j])
        frac = min(1.0, target / totals[r])
        for i, t in enumerate(recs):
            if j == 0:
                out.append(dict(t, counts={}))
            out[i]["counts"][r] = rng.binomial(t["counts"][r], frac)
    return out


def heldout(ft, y, reads_ok, species, family, b, train_rep, eval_reps, permute=None):
    blocks = L.MODELS[family]
    n_folds = splits.load(species)["n_folds"]
    acc = {B: ([], [], []) for B in eval_reps}
    for k in range(n_folds):
        tr_mask, te_mask = splits.train_test(ft.chrom, species, k)
        tr = np.nonzero(tr_mask & reads_ok)[0]
        te = np.nonzero(te_mask & reads_ok)[0]
        if len(tr) == 0 or len(te) == 0:
            continue
        p0 = L.Problem(ft, tr, b, [])
        f0 = L.fit(p0, p0.bin_counts(y[train_rep]), prior_sd=EFF.PRIOR_SD)
        pt = L.Problem(ft, tr, b, blocks, permute=permute)
        f = L.fit(pt, pt.bin_counts(y[train_rep]), prior_sd=EFF.PRIOR_SD, init=f0.as_dict())
        q0 = L.Problem(ft, te, b, [])
        q1 = L.Problem(ft, te, b, blocks, permute=permute)
        for B in eval_reps:
            ll0, n = L.loglik(f0, q0, q0.bin_counts(y[B]), per_transcript=True)
            ll1, _ = L.loglik(f, q1, q1.bin_counts(y[B]), per_transcript=True)
            acc[B][0].append(te)
            acc[B][1].append(ll1 - ll0)
            acc[B][2].append(n)
    return {B: tuple(np.concatenate(v) for v in vals) for B, vals in acc.items() if vals[0]}


def pooled_dll(parts, T):
    d, n = np.zeros(T), np.zeros(T)
    for tx, dl, nn in parts:
        np.add.at(d, tx, dl)
        np.add.at(n, tx, nn)
    return d, n


def ratio(d, n, idx=None):
    if idx is None:
        return float(d.sum() / n.sum()) if n.sum() > 0 else np.nan
    return float(d[idx].sum() / n[idx].sum())


def run_primary(b, seeds, n_boot):
    SEL, UNS = "gse174329_eif3b", "gse139131_unsel"
    sel_by_id, sel_reps = load(SEL)
    uns_by_id, uns_reps = load(UNS)
    ids = sorted(set(sel_by_id) & set(uns_by_id))
    assert ids == sorted(sel_by_id) == sorted(uns_by_id), \
        "the two human panels must share one annotation (they do by construction)"
    tot = {(SEL, r): sum(int(sel_by_id[i]["counts"][r][:sel_by_id[i]["msi"] + 3].sum()) for i in ids)
           for r in sel_reps}
    tot.update({(UNS, r): sum(int(uns_by_id[i]["counts"][r][:uns_by_id[i]["msi"] + 3].sum())
                              for i in ids) for r in uns_reps})
    target = min(tot.values())
    print("[09] leader totals {}  -> matched target {}".format(tot, target), flush=True)
    rows = []
    modes = [("observed", None)] + [("matched", s) for s in seeds]
    for mode, seed in modes:
        t0 = time.time()
        if mode == "observed":
            sel = [sel_by_id[i] for i in ids]
            uns = [uns_by_id[i] for i in ids]
        else:
            sel = thin_dataset(sel_by_id, ids, sel_reps, target, seed, SEL)
            uns = thin_dataset(uns_by_id, ids, uns_reps, target, seed, UNS)
        rd_sel = {r: EFF.leader_reads(sel, r) for r in sel_reps}
        rd_uns = {r: EFF.leader_reads(uns, r) for r in uns_reps}
        ok_sel = np.min([rd_sel[r] for r in sel_reps], axis=0) >= EFF.MIN_READS
        ok_uns = np.min([rd_uns[r] for r in uns_reps], axis=0) >= EFF.MIN_READS
        common = np.nonzero(ok_sel & ok_uns)[0]
        print("[09] {} seed {}: selected-eligible {}, unselected-eligible {}, common {}".format(
            mode, seed, ok_sel.sum(), ok_uns.sum(), len(common)), flush=True)
        ft = FeatureTable([sel[i] for i in common])
        y = {SEL: {r: ft.counts([sel[i] for i in common], r) for r in sel_reps},
             UNS: {r: ft.counts([uns[i] for i in common], r) for r in uns_reps}}
        allok = np.ones(len(common), bool)
        T = len(common)
        for fam in FAMILIES:
            res = {}
            for key, reps in ((SEL, sel_reps), (UNS, uns_reps)):
                parts = []
                for A in reps:
                    other = [B for B in reps if B != A]
                    h = heldout(ft, y[key], allok, "human", fam, b, A, other)
                    parts += [h[B] for B in other]
                res[key] = pooled_dll(parts, T)
            ds, ns = res[SEL]
            du, nu = res[UNS]
            point = (ratio(ds, ns), ratio(du, nu))
            rng = np.random.default_rng([0 if seed is None else seed, _stable_tag(fam)])
            boot = np.empty(n_boot)
            for k in range(n_boot):
                idx = rng.integers(0, T, T)
                boot[k] = ratio(ds, ns, idx) - ratio(du, nu, idx)
            rows.append({"comparison": "selected_vs_unselected", "depth_mode": mode,
                         "seed": "" if seed is None else seed, "family": fam, "bin_width": b,
                         "selected": SEL, "unselected": UNS,
                         "selected_n": int(ok_sel.sum()), "unselected_n": int(ok_uns.sum()),
                         "common_n": T, "selected_dll": point[0], "unselected_dll": point[1],
                         "difference": point[0] - point[1],
                         "ci_lo": float(np.percentile(boot, 2.5)),
                         "ci_hi": float(np.percentile(boot, 97.5)),
                         "p_difference_le_0": float((boot <= 0).mean()),
                         "matched_leader_target": target if mode == "matched" else "",
                         "selected_reads_common": float(ns.sum() / 2),
                         "unselected_reads_common": float(nu.sum() / 2),
                         "bootstrap_unit": "transcript (paired)", "n_boot": n_boot,
                         "directions": "rep1->rep2 + rep2->rep1 pooled"})
            print("  {:<4} sel {:+.5f}  unsel {:+.5f}  diff {:+.5f} [{:+.5f},{:+.5f}]".format(
                fam, point[0], point[1], point[0] - point[1], rows[-1]["ci_lo"],
                rows[-1]["ci_hi"]), flush=True)
        print("[09] {} seed {} done ({:.0f}s)".format(mode, seed, time.time() - t0), flush=True)
    out = C.results_path(OUT_DIR, "assay_comparison.tsv")
    C.write_tsv(rows, out, extra_first=("comparison", "depth_mode", "seed", "family"))
    print("[09] -> {}".format(out))


def run_same_study(b, seeds, n_perm, n_boot):
    SEL, UNS = "gse139131_eif3b", "gse139131_unsel"
    sel_by_id, sel_reps = load(SEL)
    uns_by_id, uns_reps = load(UNS)
    ids = sorted(set(sel_by_id) & set(uns_by_id))
    tot = [sum(int(sel_by_id[i]["counts"][r][:sel_by_id[i]["msi"] + 3].sum()) for i in ids)
           for r in sel_reps] + \
          [sum(int(uns_by_id[i]["counts"][r][:uns_by_id[i]["msi"] + 3].sum()) for i in ids)
           for r in uns_reps]
    target = min(tot)
    rows = []
    modes = [("observed", None)] + [("matched", s) for s in seeds]
    for mode, seed in modes:
        if mode == "observed":
            sel = [sel_by_id[i] for i in ids]
            uns = [uns_by_id[i] for i in ids]
        else:
            sel = thin_dataset(sel_by_id, ids, sel_reps, target, seed, SEL)
            uns = thin_dataset(uns_by_id, ids, uns_reps, target, seed, UNS)
        rd_sel = {r: EFF.leader_reads(sel, r) for r in sel_reps}
        rd_uns = {r: EFF.leader_reads(uns, r) for r in uns_reps}
        ok_sel = np.min([rd_sel[r] for r in sel_reps], axis=0) >= EFF.MIN_READS
        ok_uns = np.min([rd_uns[r] for r in uns_reps], axis=0) >= EFF.MIN_READS
        common = np.nonzero(ok_sel & ok_uns)[0]
        print("[09s] {} seed {}: eIF3B-eligible {}, unselected-eligible {}, common {}".format(
            mode, seed, ok_sel.sum(), ok_uns.sum(), len(common)), flush=True)
        ft = FeatureTable([sel[i] for i in common])
        y = {SEL: {r: ft.counts([sel[i] for i in common], r) for r in sel_reps},
             UNS: {r: ft.counts([uns[i] for i in common], r) for r in uns_reps}}
        allok = np.ones(len(common), bool)
        T = len(common)
        for fam in FAMILIES:
            dirs = [(SEL, "rep1", "rep1")] + [(UNS, r, r) for r in uns_reps] + \
                   [(UNS, A, B) for A in uns_reps for B in uns_reps if A != B]
            for key, A, B in dirs:
                h = heldout(ft, y[key], allok, "human", fam, b, A, [B])
                tx, dl, nn = h[B]
                d, n = pooled_dll([(tx, dl, nn)], T)
                real = ratio(d, n)
                rng = np.random.default_rng([0 if seed is None else seed, _stable_tag(fam + key + A + B)])
                boot = np.array([ratio(d, n, rng.integers(0, T, T)) for _ in range(n_boot)])
                row = {"comparison": "same_study_selection", "depth_mode": mode,
                       "seed": "" if seed is None else seed, "family": fam, "bin_width": b,
                       "dataset": key, "direction": "{}->{}".format(A, B),
                       "within_library": A == B, "common_n": T,
                       "dll": real, "ci_lo": float(np.percentile(boot, 2.5)),
                       "ci_hi": float(np.percentile(boot, 97.5)),
                       "evidence": ("within-library / dataset-specific" if A == B
                                    else "cross-replicate (reference only)"),
                       "matched_leader_target": target if mode == "matched" else ""}
                if mode == "observed" and A == B and n_perm > 0:
                    blocks, permuted, kind = NUL.NULL_SPECS[fam]
                    nv = []
                    for i in range(n_perm):
                        prng = np.random.default_rng([0, list(NUL.NULL_SPECS).index(fam), i])
                        perm = NUL.PERMUTERS[kind](ft, prng)
                        hp = heldout(ft, y[key], allok, "human", fam, b, A, [B],
                                     permute={blk: perm for blk in permuted})
                        txp, dlp, nnp = hp[B]
                        dp_, np_ = pooled_dll([(txp, dlp, nnp)], T)
                        nv.append(ratio(dp_, np_))
                    nv = np.array(nv)
                    sd = nv.std(ddof=1)
                    row.update(null_mean=float(nv.mean()), null_sd=float(sd), n_perm=n_perm,
                               null_p=float((1 + (nv >= real).sum()) / (1 + n_perm)),
                               null_z=float((real - nv.mean()) / sd) if sd > 0 else np.nan)
                rows.append(row)
                print("  {:<4} {:<16} {}  dLL {:+.5f} [{:+.5f},{:+.5f}]{}".format(
                    fam, key, row["direction"], real, row["ci_lo"], row["ci_hi"],
                    "  null p={:.3f}".format(row["null_p"]) if "null_p" in row else ""),
                    flush=True)
    out = C.results_path(OUT_DIR, "gse139131_same_study_selection.tsv")
    C.write_tsv(rows, out, extra_first=("comparison", "depth_mode", "seed", "family", "dataset",
                                        "direction"))
    print("[09s] -> {}".format(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("primary", "same_study"), required=True)
    ap.add_argument("--bin", type=int, default=10)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-perm", type=int, default=100, help="same_study nulls")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    C.ensure_layout()
    os.makedirs(os.path.join(C.RESULTS, OUT_DIR), exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    if args.mode == "primary":
        run_primary(args.bin, seeds, args.n_boot)
    else:
        run_same_study(args.bin, seeds, args.n_perm, args.n_boot)


if __name__ == "__main__":
    main()
