import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.recoverability import config as C

POWER_OK = 0.8
REFERENCE_SIZE = 1
SYNTH_FAMILY = {"SW": "SW", "ST": "ST", "ST_paired": "ST", "ST_dG": "ST", "AUG": "AUG"}
SYNTH_TRUTH = {"SW": "B", "ST": "C", "AUG": "D"}
EFFECTS = ["S1", "S3", "S5", "SW", "ST_paired", "ST_dG", "ST", "AUG", "UORF"]
COMBINED = ["SW+ST", "SW+AUG", "ST+AUG", "SW+ST+AUG"]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load(path):
    return C.read_tsv(path) if os.path.exists(path) else []


def power_curve(rows, family, truth, size):
    curve = {}
    for r in rows:
        if (r["model"] == family and r["truth"] == truth and r["metric"] == "p_recovered"
                and int(float(r["size_index"])) == size):
            curve[int(float(r["target_median_depth"]))] = _f(r["value"])
    return dict(sorted(curve.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gse174329_eif3b")
    ap.add_argument("--noise", default="dm", help="synthetic noise model used for power")
    ap.add_argument("--power-tag", default="",
                    help="tag of the synthetic power run to read (06 --tag); default untagged")
    ap.add_argument("--out", default=None,
                    help="output table (default summary/recoverability_summary.tsv, the "
                         "frozen primary summary -- pass a new path for other datasets)")
    args = ap.parse_args()
    C.ensure_layout()
    key = args.dataset
    eff = [r for r in load(C.results_path("effects", "nested_effects.tsv"))
           if r.get("panel", r["dataset"]) == key and r["metric"] == "heldout_dll_read_weighted"]
    nul = [r for r in load(C.results_path("nulls", "shuffle_nulls.tsv")) if r["dataset"] == key]
    pw = load(C.results_path("synthetic", "power_curve_{}_{}{}.tsv".format(
        key, args.noise, "_" + args.power_tag if args.power_tag else "")))
    qc = [r for r in load(C.results_path("qc", "qc_summary.tsv"))
          if r["dataset"] == key and r["metric"] == "median_reads_per_leader_ge5"]
    obs_depth = float(np.median([_f(r["value"]) for r in qc])) if qc else np.nan

    rows = []
    for model in EFFECTS:
        verdict_by_b = {}
        for b in C.MODEL_BINS:
            rs = [r for r in eff if r["model"] == model and str(r["bin_width"]) == str(b)]
            if not rs:
                continue
            cross = [r for r in rs if r["replicate"].split("->")[0] != r["replicate"].split("->")[1]]
            within = [r for r in rs if r not in cross]
            cross_ok = bool(cross) and all(_f(r["ci_lo"]) > 0 for r in cross)
            within_ok = any(_f(r["ci_lo"]) > 0 for r in within)
            nr = [r for r in nul if r["model"] == model and str(r["bin_width"]) == str(b)
                  and r["metric"] == "empirical_p" and r.get("cross_replicate") == "True"]
            null_ok = all(_f(r["value"]) <= 0.05 for r in nr) if nr else None
            recov = cross_ok and null_ok is True
            sfam = SYNTH_FAMILY.get(model)
            curve = power_curve(pw, sfam, SYNTH_TRUTH[sfam], REFERENCE_SIZE) if sfam and pw else {}
            p_obs = np.nan
            if curve and np.isfinite(obs_depth):
                ds = np.array(sorted(curve))
                near = ds[np.argmin(np.abs(np.log(ds) - np.log(obs_depth)))]
                p_obs = curve[int(near)]
            if recov:
                status = "recoverable"
            elif within_ok and not cross_ok:
                status = "dataset_specific"
            elif cross_ok and null_ok is None:
                status = "null_pending"
            elif curve and np.isfinite(p_obs):
                if p_obs >= POWER_OK:
                    status = "not_recoverable_at_depth"
                elif any(v >= POWER_OK for d_, v in curve.items() if d_ > obs_depth):
                    status = "depth_limited"
                else:
                    status = "inconclusive"
            elif cross_ok and null_ok is False:
                status = "fails_position_matched_null"
            else:
                status = "inconclusive"
            verdict_by_b[b] = status
            best = max(cross, key=lambda r: _f(r["value"])) if cross else rs[0]
            rows.append({"dataset": key, "replicate": "cross", "split": "chrom_heldout", "seed": 0,
                         "bin_width": b, "model": model, "metric": "status", "value": status,
                         "n_transcripts": best["n_transcripts"], "n_reads": best["n_reads"],
                         "cross_min_dll": min((_f(r["value"]) for r in cross), default=np.nan),
                         "cross_min_ci_lo": min((_f(r["ci_lo"]) for r in cross), default=np.nan),
                         "within_max_ci_lo": max((_f(r["ci_lo"]) for r in within), default=np.nan),
                         "null_ok": null_ok, "power_at_observed_depth": p_obs,
                         "observed_median_depth": obs_depth, "power_threshold": POWER_OK,
                         "reference_size": REFERENCE_SIZE, "noise_model": args.noise})
        if verdict_by_b.get(1) not in (None, "recoverable", "null_pending"):
            for r in rows:
                if r["model"] == model and r["bin_width"] != 1 and r["value"] == "recoverable":
                    r["value"] = "coarse_resolution_recoverable"

    for model in COMBINED:
        for b in C.MODEL_BINS:
            cross = [r for r in eff if r["model"] == model and str(r["bin_width"]) == str(b)
                     and r["replicate"].split("->")[0] != r["replicate"].split("->")[1]]
            if cross:
                rows.append({"dataset": key, "replicate": "cross", "split": "chrom_heldout",
                             "seed": 0, "bin_width": b, "model": model,
                             "metric": "combined_min_cross_dll",
                             "value": min(_f(r["value"]) for r in cross),
                             "cross_min_ci_lo": min(_f(r["ci_lo"]) for r in cross),
                             "n_transcripts": cross[0]["n_transcripts"], "n_reads": ""})

    for r in load(C.results_path("chain", "identifiability.tsv")):
        if r["metric"] == "identifiable" and r.get("reads_per_transcript") == "10000" and r["value"] == "0":
            rows.append({"dataset": "theory", "replicate": "", "split": "", "seed": "",
                         "bin_width": r["bin_width"], "model": "{}:{}".format(r["model"], r["condition"]),
                         "metric": "status", "value": "structurally_unidentifiable",
                         "n_transcripts": "", "n_reads": ""})
    out = args.out or C.results_path("summary", "recoverability_summary.tsv")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    C.write_tsv(rows, out)
    for r in rows:
        print("{:<38} b={:<3} {}".format(r["model"], r["bin_width"], r["value"]))
    print("-> {}".format(out))


if __name__ == "__main__":
    main()
