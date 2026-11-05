import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.recoverability import config as C
from pipeline._common import dataset, load_counts_npz, work_path

def _bx():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_bx", os.path.join(C.ROOT, "pipeline", "03_build_examples.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stage_offsets(keys):
    from analysis.recoverability import psite
    bx = _bx()
    rows, offsets = [], dict(C.measured("psite_offsets", {}))
    starts_by_gtf = {}
    for key in keys:
        d = C.DATASETS[key]
        ds = dataset(d["preset"])
        if ds["gtf"] not in starts_by_gtf:
            st, _, _ = bx.parse_start_codons(ds["gtf"])
            starts_by_gtf[ds["gtf"]] = [x for x in st if x[1] not in ("Mito", "chrM")]
        starts = starts_by_gtf[ds["gtf"]]
        per_rep = {}
        for rep, r in d["replicates"].items():
            path = work_path(ds, r["fraction"], "reads.npz")
            if not os.path.exists(path):
                print("[qc] {} {}: no reads.npz yet, skipped".format(key, rep))
                continue
            reads = psite.load_reads(path)
            rule, report, best_all = psite.measure(reads, starts)
            for x in report:
                for p_, v in zip(psite.POS, x["metagene"]):
                    rows.append({"dataset": key, "replicate": rep, "model": x["anchor"],
                                 "metric": "metagene_frac", "length_bin": x["bin"],
                                 "position": int(p_), "value": float(v),
                                 "n_transcripts": x["genes"]})
            five = [x for x in report if x["bin"] == "all" and x["anchor"] == "5p"][0]
            scored = {}
            for name, rl in (("single_5p_offset", psite.single_offset_rule(-five["peak"])),
                             ("length_resolved", rule)):
                mg, used = psite.metagene(psite.apply_rule(reads, rl), starts)
                nb = mg[(psite.POS >= -psite.NB) & (psite.POS < psite.NB)]
                scored[name] = float(mg[psite.POS == 0][0] / max(np.median(nb), 1e-12))
                for p_, v in zip(psite.POS, mg):
                    rows.append({"dataset": key, "replicate": rep, "model": name,
                                 "metric": "psite_metagene_frac", "length_bin": "all",
                                 "position": int(p_), "value": float(v),
                                 "n_transcripts": used})
            span = np.concatenate([s_ for _p, s_ in reads.values()])
            hist = {"{}-{}".format(lo, hi): float(((span >= lo) & (span <= hi)).mean())
                    for lo, hi in psite.LENGTH_BINS}
            per_rep[rep] = {"rule": rule, "single_5p_offset": -five["peak"],
                            "sharpness_at_0": scored, "length_bin_fraction": hist,
                            "bins": [{k: v for k, v in x.items() if k != "metagene"}
                                     for x in report]}
            print("[qc] {} {}: single 5' offset {} (sharpness {:.2f}) vs length-resolved "
                  "(sharpness {:.2f})".format(key, rep, -five["peak"],
                                              scored["single_5p_offset"],
                                              scored["length_resolved"]))
            for lo, hi, anc, off in rule:
                print("      {:>3}-{:<3} nt  {:>4.1%}  {} anchor, offset {}".format(
                    lo, hi, hist["{}-{}".format(lo, hi)], anc, off))
        if not per_rep:
            continue
        ref = "rep1" if "rep1" in per_rep else sorted(per_rep)[0]
        agree = {rep: v["rule"] == per_rep[ref]["rule"] for rep, v in per_rep.items()}
        use_rule = per_rep[ref]["rule"]
        if per_rep[ref]["sharpness_at_0"]["single_5p_offset"] >=                 per_rep[ref]["sharpness_at_0"]["length_resolved"]:
            use_rule = psite.single_offset_rule(per_rep[ref]["single_5p_offset"])
        offsets[key] = {"rule": use_rule, "rule_measured_on": ref,
                        "rule_identical_across_replicates": agree,
                        "replicates": per_rep,
                        "offset": int(per_rep[ref]["single_5p_offset"]),
                        "method": "per length bin, the footprint end (5'/3') with the "
                                  "sharper gene-normalised start-codon metagene; kept "
                                  "only if sharper at 0 than the best single 5' offset"}
    out = C.results_path("qc", "metagene.tsv")
    done = {(r["dataset"], r["replicate"]) for r in rows}
    C.merge_tsv(out, rows, lambda r: (r["dataset"], r["replicate"]) not in done)
    C.update_manifest("psite_offsets", {k: offsets[k] for k in keys if k in offsets}, merge=True)
    print("[qc] P-site rules -> manifest")


LOG_PATTERNS = {
    "raw_reads": r"Total reads processed:\s+([\d,]+)",
    "trimmed_reads": r"Reads written \(passing filters\):\s+([\d,]+)",
}


def parse_align_log(preset):
    name = "align_{}.log".format(preset)
    cands = [os.path.join(C.RESULTS, "logs", name), os.path.join(C.RESULTS, "qc", "logs", name)]
    path = next((p for p in cands if os.path.exists(p)), None)
    if path is None:
        print("[qc] WARNING: no alignment log for {} (looked in {}); raw/trimmed/unique "
              "counts and leader enrichment will be missing".format(preset, ", ".join(cands)))
        return {}
    text = open(path, errors="replace").read()
    runs = re.findall(r"\[align\] cutadapt \| .*?< \S*?([SED]RR\d+)", text)
    raw = [int(x.replace(",", "")) for x in re.findall(LOG_PATTERNS["raw_reads"], text)]
    trim = [int(x.replace(",", "")) for x in re.findall(LOG_PATTERNS["trimmed_reads"], text)]
    rate = [float(x) for x in re.findall(r"([\d.]+)% overall alignment rate", text)]
    uniq = [int(x) for x in re.findall(r"'assigned': (\d+)", text)]
    out = {}
    for i, srr in enumerate(runs):
        r = {}
        if i < len(raw):
            r["raw_reads"] = raw[i]
        if i < len(trim):
            r["trimmed_reads"] = trim[i]
            if i < len(rate):
                r["aligned_reads"] = int(round(rate[i] / 100 * trim[i]))
        if i < len(uniq):
            r["unique_reads"] = uniq[i]
        out[srr] = r
    return out


def region_fractions(track, starts_tx, cds_blocks, utr_blocks):
    tot = {"utr5": 0, "start": 0, "cds": 0, "utr3": 0}
    for tx, chrom, strand, cs, ce in starts_tx:
        cds = sorted(cds_blocks.get(tx, []))
        if not cds:
            continue
        lo, hi = cds[0][0], cds[-1][1]
        a = cs if strand == "+" else ce - 1
        start_iv = (a - 15, a + 16)
        for s, e in utr_blocks.get(tx, []):
            v = int(track.window(chrom, strand, s, e).sum())
            five = (e <= lo) if strand == "+" else (s >= hi)
            tot["utr5" if five else "utr3"] += v
        st = int(track.window(chrom, strand, *start_iv).sum())
        cd = sum(int(track.window(chrom, strand, s, e).sum()) for s, e in cds)
        tot["start"] += st
        tot["cds"] += max(cd - st, 0)
    s = sum(tot.values())
    return {k: v / s for k, v in tot.items()} if s else {}


def stage_tables(keys):
    from analysis.recoverability.panel import load_panel
    rows, per_tx = [], []
    for key in keys:
        d = C.DATASETS[key]
        try:
            recs, meta = load_panel(key)
        except FileNotFoundError:
            print("[qc] {}: no panel yet, skipped".format(key))
            continue
        logs = parse_align_log(d["preset"])
        ds = dataset(d["preset"])
        for rep, r in d["replicates"].items():
            def row(metric, value, **kw):
                x = {"dataset": key, "replicate": rep, "metric": metric, "value": value,
                     "run": r["srr"]}
                x.update(kw)
                rows.append(x)
            lg = logs.get(r["srr"], {})
            for k in ("raw_reads", "trimmed_reads", "aligned_reads", "unique_reads"):
                if k in lg:
                    row(k, lg[k])
            if "raw_reads" in lg:
                row("raw_reads_match_sra", int(lg["raw_reads"] == C.SRA_SPOTS.get(r["srr"])))
            lead = np.array([int(t["counts"][rep][:t["msi"] + 3].sum()) for t in recs])
            L = np.array([t["msi"] + 3 for t in recs])
            row("leader_reads_total", int(lead.sum()), n_transcripts=len(recs))
            if "unique_reads" in lg:
                row("leader_frac_of_unique", lead.sum() / max(lg["unique_reads"], 1))
            row("leaders_gt0", int((lead > 0).sum()))
            for th in C.QC_THRESHOLDS:
                row("leaders_ge_{}".format(th), int((lead >= th).sum()))
            row("median_reads_per_leader", float(np.median(lead)))
            row("mean_reads_per_leader", float(lead.mean()))
            row("median_reads_per_leader_ge5", float(np.median(lead[lead >= 5])) if (lead >= 5).any() else 0)
            row("median_leader_length", float(np.median(L - 3)))
            row("total_leader_positions", int(L.sum()))
            zeros = sum(int((t["counts"][rep][:t["msi"] + 3] == 0).sum()) for t in recs)
            row("zero_position_frac", zeros / L.sum())
            for t, n in zip(recs, lead):
                per_tx.append({"dataset": key, "replicate": rep, "transcript_id": t["transcript_id"],
                               "chrom": t["chrom"], "length": t["msi"], "n_reads": int(n),
                               "zero_frac": float((t["counts"][rep][:t["msi"] + 3] == 0).mean())})
            if d["species"] == "human":
                fr = _human_regions(ds, r)
                for k, v in fr.items():
                    row("region_frac_" + k, v)
        print("[qc] {} done".format(key))
    out = C.results_path("qc", "qc_summary.tsv")
    done = {r["dataset"] for r in rows}
    C.merge_tsv(out, rows, lambda r: r["dataset"] not in done)
    C.write_tsv(per_tx, C.results_path("qc", "reads_per_transcript.tsv"))
    print("[qc] -> {}".format(out))


_HUMAN_ANN = {}


def _human_regions(ds, r):
    if "x" not in _HUMAN_ANN:
        bx = _bx()
        starts, _, cds = bx.parse_start_codons(ds["gtf"])
        utr = {}
        with open(ds["gtf"]) as fh:
            for line in fh:
                f = line.split("\t")
                if len(f) > 8 and f[2] == "UTR":
                    tx = bx._attr(f[8], "transcript_id")
                    utr.setdefault(tx, []).append((int(f[3]) - 1, int(f[4])))
        _HUMAN_ANN["x"] = (starts, cds, utr)
    starts, cds, utr = _HUMAN_ANN["x"]
    track = load_counts_npz(work_path(ds, r["fraction"], "5ends.npz"))
    return region_fractions(track, starts, cds, utr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("offsets", "tables"), required=True)
    ap.add_argument("--datasets", default=",".join(C.DATASETS))
    args = ap.parse_args()
    C.ensure_layout()
    keys = [k for k in args.datasets.split(",") if k]
    if args.stage == "offsets":
        stage_offsets(keys)
    else:
        stage_tables(keys)


if __name__ == "__main__":
    main()
