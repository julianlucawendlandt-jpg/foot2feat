import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.recoverability import config as C
from analysis.recoverability import likelihood as L
from analysis.recoverability.panel import annotation_path, counts_path
from pipeline._common import dataset as preset_of, work_path

OUT = "missing_eval"
UNSEL = "gse139131_unsel"


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _md5(path, limit=2 * 2 ** 30):
    if not os.path.isfile(path):
        return ""
    if os.path.getsize(path) > limit:
        return "skipped (>2 GB)"
    return C.md5sum(path)


def audit():
    rows, fatal = [], []
    eff_path = C.results_path("effects", "nested_effects.tsv")
    eff = C.read_tsv(eff_path) if os.path.exists(eff_path) else []
    qc_path = C.results_path("qc", "qc_summary.tsv")
    qc = C.read_tsv(qc_path) if os.path.exists(qc_path) else []
    models = [m for m in L.MODELS if m != "P"]
    for key in ("gse139131_unsel", "gse174329_eif3b", "gse139131_eif3b", "archer_yeast"):
        d = C.DATASETS[key]
        ds = preset_of(d["preset"])
        reps = list(d["replicates"])
        need = []
        for rep, r in d["replicates"].items():
            need.append(("count_track_" + rep, work_path(ds, r["fraction"], "reads.npz")))
        need += [("panel_annotation", annotation_path(d["species"])),
                 ("panel_counts", counts_path(key)),
                 ("heldout_ll_npz", C.results_path("effects", "heldout_ll_{}.npz".format(key))),
                 ("fits_table", C.results_path("effects", "fits_{}.tsv".format(key))),
                 ("chromosome_folds", C.results_path("{}_chromosome_folds.json".format(d["species"]))),
                 ("nested_effects_table", eff_path), ("qc_summary", qc_path)]
        for name, path in need:
            ok = os.path.exists(path)
            rows.append({"dataset": key, "required_input": name, "path": os.path.relpath(path, C.ROOT),
                         "exists": ok, "checksum": _md5(path) if ok else ""})
            if not ok:
                fatal.append((key, name, path))
        n_dirs = len(reps) ** 2
        have = {(r["model"], r["bin_width"], r["replicate"]) for r in eff
                if r.get("panel", r["dataset"]) == key and r["metric"] == "heldout_dll_read_weighted"}
        expect = {(m, str(b), "{}->{}".format(a, c)) for m in models for b in C.MODEL_BINS
                  for a in reps for c in reps}
        missing = sorted(expect - have)
        rows.append({"dataset": key, "required_input": "nested_effects_rows",
                     "path": os.path.relpath(eff_path, C.ROOT),
                     "exists": not missing,
                     "checksum": "{} of {} expected rows present{}".format(
                         len(expect & have), len(expect),
                         "" if not missing else "; first missing {}".format(missing[0]))})
        if missing:
            fatal.append((key, "nested_effects_rows", "{} missing".format(len(missing))))
        has_qc = any(r["dataset"] == key and r["metric"] == "median_reads_per_leader_ge5" for r in qc)
        rows.append({"dataset": key, "required_input": "qc_rows", "path": os.path.relpath(qc_path, C.ROOT),
                     "exists": has_qc, "checksum": ""})
        if not has_qc:
            fatal.append((key, "qc_rows", qc_path))
        psite = C.measured("psite_offsets", {}).get(key)
        rows.append({"dataset": key, "required_input": "psite_rule", "path": "manifest.json",
                     "exists": psite is not None,
                     "checksum": json.dumps(psite.get("rule")) if psite else ""})
        if psite is None:
            fatal.append((key, "psite_rule", "manifest"))
    for name, path in (("akirtava_supplement", os.path.join(C.ROOT, "eval", "gkaf165_supplemental_files",
                                                          "Akirtava_et_al_2024_Supp_Data_Rev1.xlsx")),
                       ("sample2019_table", os.path.join(C.ROOT, "eval", "sample2019.tsv"))):
        ok = os.path.exists(path)
        rows.append({"dataset": "reporter", "required_input": name, "path": os.path.relpath(path, C.ROOT),
                     "exists": ok, "checksum": _md5(path) if ok else ""})
        if not ok:
            fatal.append(("reporter", name, path))
    out = C.results_path(OUT, "input_audit.tsv")
    C.write_tsv(rows, out, extra_first=("dataset", "required_input", "path", "exists", "checksum"))
    for r in rows:
        print("{:<18} {:<22} {:<6} {}".format(r["dataset"], r["required_input"], str(r["exists"]),
                                             r["path"]))
    if fatal:
        print("\nAUDIT FAILED -- missing or inconsistent frozen inputs (nothing regenerated):")
        for f in fatal:
            print("   ", f)
        sys.exit(1)
    print("-> {}  (all inputs present)".format(out))


def unselected_table(power_tag="n50"):
    eff = [r for r in C.read_tsv(C.results_path("effects", "nested_effects.tsv"))
           if r.get("panel", r["dataset"]) == UNSEL and r["metric"] == "heldout_dll_read_weighted"]
    nul = [r for r in C.read_tsv(C.results_path("nulls", "shuffle_nulls.tsv")) if r["dataset"] == UNSEL]
    summ = C.read_tsv(C.results_path(OUT, "unselected_summary_raw.tsv"))
    rows = []
    for s in summ:
        if s["metric"] != "status" or s["dataset"] != UNSEL:
            continue
        fam, b = s["model"], s["bin_width"]
        row = {"dataset": UNSEL, "family": fam, "bin_width": b}
        for d in ("rep1->rep2", "rep2->rep1"):
            e = next((r for r in eff if r["model"] == fam and r["bin_width"] == b and r["replicate"] == d), None)
            tag = d.replace("->", "_to_")
            row["dll_" + tag] = _f(e["value"]) if e else np.nan
            row["ci_lo_" + tag] = _f(e["ci_lo"]) if e else np.nan
            row["ci_hi_" + tag] = _f(e["ci_hi"]) if e else np.nan
            for m, name in (("empirical_p", "null_p_"), ("z", "null_z_")):
                n = next((r for r in nul if r["model"] == fam and r["bin_width"] == b
                          and r["replicate"] == d and r["metric"] == m), None)
                row[name + tag] = _f(n["value"]) if n else np.nan
        row.update({"observed_median_depth": _f(s.get("observed_median_depth")),
                    "power_reference_size": _f(s.get("power_at_observed_depth")),
                    "power_noise": "Dirichlet-multinomial, alpha calibrated on the unselected pair",
                    "status": s["value"], "n_transcripts": s.get("n_transcripts")})
        rows.append(row)
    out = C.results_path(OUT, "unselected_recoverability_summary.tsv")
    C.write_tsv(rows, out, extra_first=("dataset", "family", "bin_width", "status"))
    for r in rows:
        print("{:<10} b={:<3} {:<30} dLL {:+.4f}/{:+.4f}  null p {}/{}".format(
            r["family"], r["bin_width"], r["status"], r["dll_rep1_to_rep2"], r["dll_rep2_to_rep1"],
            r["null_p_rep1_to_rep2"], r["null_p_rep2_to_rep1"]))
    print("-> {}".format(out))


def _table(path):
    p = C.results_path(*path.split("/"))
    return C.read_tsv(p) if os.path.exists(p) else []


def report():
    L_ = ["# Added evaluations — machine-generated summary", "",
          "Generated from the result files; see run_missing_evaluations.py for commands.", ""]
    u = _table(OUT + "/unselected_recoverability_summary.tsv")
    if u:
        L_ += ["## A. Unselected human 40S (gse139131_unsel)", "",
               "| family | bin | status | dLL r1→r2 | dLL r2→r1 | null p r1→r2 | null p r2→r1 | power |",
               "|---|---|---|---|---|---|---|---|"]
        for r in u:
            L_.append("| {} | {} | {} | {:+.4f} | {:+.4f} | {} | {} | {} |".format(
                r["family"], r["bin_width"], r["status"], _f(r["dll_rep1_to_rep2"]),
                _f(r["dll_rep2_to_rep1"]), r["null_p_rep1_to_rep2"], r["null_p_rep2_to_rep1"],
                r["power_reference_size"]))
        L_.append("")
    a = _table(OUT + "/assay_comparison.tsv")
    if a:
        L_ += ["### Selected vs unselected on common transcripts (10 nt)", "",
               "| depth | seed | family | common n | selected | unselected | difference [95% CI] |",
               "|---|---|---|---|---|---|---|"]
        for r in a:
            L_.append("| {} | {} | {} | {} | {:+.4f} | {:+.4f} | {:+.4f} [{:+.4f}, {:+.4f}] |".format(
                r["depth_mode"], r["seed"], r["family"], r["common_n"], _f(r["selected_dll"]),
                _f(r["unselected_dll"]), _f(r["difference"]), _f(r["ci_lo"]), _f(r["ci_hi"])))
        L_.append("")
    s = _table(OUT + "/gse139131_same_study_selection.tsv")
    if s:
        L_ += ["### Same-study control (within-library / dataset-specific only)", "",
               "| depth | seed | family | dataset | direction | dLL [95% CI] | null p |", "|---|---|---|---|---|---|---|"]
        for r in s:
            L_.append("| {} | {} | {} | {} | {} | {:+.4f} [{:+.4f}, {:+.4f}] | {} |".format(
                r["depth_mode"], r["seed"], r["family"], r["dataset"], r["direction"], _f(r["dll"]),
                _f(r["ci_lo"]), _f(r["ci_hi"]), r.get("null_p", "")))
        L_.append("")
    y = _table(OUT + "/yeast_single_library_summary.tsv")
    if y:
        L_ += ["## B. Yeast (single SSU library — within-library evidence only)", "",
               "| family | bin | dLL [95% CI] | null p | status | optimistic power |", "|---|---|---|---|---|---|"]
        for r in y:
            L_.append("| {} | {} | {:+.4f} [{:+.4f}, {:+.4f}] | {} | {} | {} |".format(
                r["family"], r["bin_width"], _f(r["heldout_dll_read_weighted"]), _f(r["ci_lo"]),
                _f(r["ci_hi"]), r["null_p"], r["status"], r["optimistic_power_at_observed_depth"]))
        L_ += ["", "Yeast has only one independent SSU library; cross-replicate recoverability is "
               "untestable.", ""]
    mc = _table("downstream/model_choice.tsv")
    dsum = _table("downstream/downstream_utility_final.tsv")
    if mc:
        L_ += ["## C. Reporter utility", "", "Model choice (frozen, from TCP-seq results only):", ""]
        for r in mc:
            L_.append("- {}: {} at {} nt ({} = {})".format(r["species"], r["selected_model"],
                                                         r["bin_width"], r["selection_metric"], r["value"]))
        L_.append("")
    if dsum:
        L_ += ["| reporter | target | 40S source | split | bootstrap unit | stratum | model | n | R² | "
               "ΔR² vs seq [95% CI] | P(ΔR²≤0) |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in dsum:
            L_.append("| {} | {} | {} | {} | {} | {} | {} | {} | {:.4f} | {} | {} |".format(
                r["dataset"], r["target"], r["fortyS_source"], r["split_mode"],
                r.get("bootstrap_unit", ""), r["stratum"],
                r["model"], r["n"], _f(r["r2"]),
                "" if r["model"] == "seq" else "{:+.4f} [{:+.4f}, {:+.4f}]".format(
                    _f(r["delta_r2"]), _f(r["delta_ci_lo"]), _f(r["delta_ci_hi"])),
                "" if r["model"] == "seq" else r["p_delta_le_0"]))
        L_.append("")
    chk = _table("downstream/validation_checks.tsv")
    if chk:
        L_ += ["Validation checks:", ""] + ["- [{}] {}".format("x" if r["passed"] == "True" else " ",
                                                            r["check"]) for r in chk] + [""]
    out = C.results_path(OUT, "report.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L_))
    print("-> {}".format(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("audit", "unselected_table", "report"))
    ap.add_argument("--power-tag", default="n50")
    args = ap.parse_args()
    os.makedirs(os.path.join(C.RESULTS, OUT), exist_ok=True)
    if args.command == "audit":
        audit()
    elif args.command == "unselected_table":
        unselected_table(args.power_tag)
    else:
        report()


if __name__ == "__main__":
    main()
