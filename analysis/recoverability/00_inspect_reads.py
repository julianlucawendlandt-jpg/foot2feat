import argparse
import collections
import gzip
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.recoverability import config as C
from pipeline._common import (BOWTIE2_HINT, WORK_DIR, cutadapt_cmd, dataset,
                              find_cutadapt, find_fastqs, require)

ADAPTERS = {
    "smallRNA_v1.5": "CTGTAGGCACCATCAAT",
    "TruSeq_linker": "AGATCGGAAGAGCACACGTCTGAA",
    "RA3_yeast": "TGGAATTCTCGGGTGCCAAGG",
}
POLY_A = "A" * 10
SCRATCH = os.path.join(WORK_DIR, "inspect")
RRNA_DIR = os.path.join(C.ROOT, "ref", "rrna")

HUMAN_MT_RRNA = [("chrM", 647, 1601), ("chrM", 1670, 3229)]


def runs_in_panel():
    out = []
    for key, d in C.DATASETS.items():
        for rep, r in d["replicates"].items():
            out.append((key, rep, d["preset"], r["fraction"], r["srr"]))
        for frac, r in d.get("auxiliary", {}).items():
            out.append((key, frac, d["preset"], frac, r["srr"]))
    return out


def head_fastq(src, dst, n):
    opener = gzip.open if src.endswith(".gz") else open
    seqs = []
    with opener(src, "rt") as fh, open(dst, "w") as out:
        for i, line in enumerate(fh):
            if i >= 4 * n:
                break
            out.write(line)
            if i % 4 == 1:
                seqs.append(line.strip())
    return seqs


def rrna_index(species, genome_fa):
    prefix = os.path.join(RRNA_DIR, species + "_rrna")
    if os.path.exists(prefix + ".1.bt2"):
        return prefix
    fa = prefix + ".fa"
    with open(fa, "w") as out:
        if species == "yeast":
            out.write(open(os.path.join(RRNA_DIR, "yeast_rdna.fa")).read())
        else:
            out.write(open(os.path.join(RRNA_DIR, "human_rrna_nuclear.fa")).read())
            mt = _fetch_regions(genome_fa, HUMAN_MT_RRNA)
            for (c, s, e), seq in zip(HUMAN_MT_RRNA, mt):
                out.write(">{}_{}_{}\n{}\n".format(c, s, e, seq))
    build = require("bowtie2-build-s", BOWTIE2_HINT, fallbacks=("bowtie2-build",))
    subprocess.run([build, "-q", fa, prefix], check=True)
    return prefix


def _fetch_regions(fa, regions):
    want = {c for c, _, _ in regions}
    seqs, name, buf = {}, None, []
    with open(fa) as fh:
        for line in fh:
            if line.startswith(">"):
                if name in want:
                    seqs[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            elif name in want:
                buf.append(line.strip())
    if name in want:
        seqs[name] = "".join(buf)
    return [seqs[c][s:e].upper() for c, s, e in regions]


def align_stats(bowtie2, index, fq, threads):
    p = subprocess.Popen([bowtie2, "-x", index, "-U", fq, "-L", "18", "-N", "0",
                          "--threads", str(threads)],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    n = aligned = unique = 0
    for line in p.stdout:
        if line.startswith("@"):
            continue
        f = line.split("\t", 6)
        flag = int(f[1])
        if flag & 0x900:
            continue
        n += 1
        if not flag & 0x4:
            aligned += 1
            if int(f[4]) >= 10:
                unique += 1
    p.wait()
    return n, aligned, unique


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--runs", default="", help="comma-separated SRRs; default all")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    C.ensure_layout()
    os.makedirs(SCRATCH, exist_ok=True)
    only = {s for s in args.runs.split(",") if s}
    bowtie2 = require("bowtie2-align-s", BOWTIE2_HINT, fallbacks=("bowtie2",))
    cutadapt = find_cutadapt()

    rows, lens = [], []
    for key, rep, preset, frac, srr in runs_in_panel():
        if only and srr not in only:
            continue
        ds = dataset(preset)
        species = C.DATASETS[key]["species"]
        src = find_fastqs(fractions={frac: srr})[frac]
        print("[inspect] {} {} {} <- {}".format(key, rep, srr, src))
        raw_fq = os.path.join(SCRATCH, srr + ".head.fastq")
        seqs = head_fastq(src, raw_fq, args.n)

        def row(metric, value):
            rows.append({"dataset": key, "replicate": rep, "split": "",
                         "seed": "", "bin_width": "", "model": "",
                         "n_transcripts": "", "n_reads": len(seqs),
                         "metric": metric, "value": value,
                         "run": srr, "sampling": "head"})

        L = [len(s) for s in seqs]
        row("raw_length_median", float(sorted(L)[len(L) // 2]))
        row("raw_length_max", float(max(L)))
        for name, a in ADAPTERS.items():
            row("adapter_frac_" + name, sum(a[:12] in s for s in seqs) / len(seqs))
        row("adapter_frac_polyA10", sum(POLY_A in s for s in seqs) / len(seqs))
        row("preset_adapter", ds["adapter"])

        trimmed = os.path.join(SCRATCH, srr + ".head.trimmed.fastq")
        subprocess.run(cutadapt_cmd(cutadapt, raw_fq, trimmed, threads=2, ds=ds),
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tl = []
        with open(trimmed) as fh:
            for i, line in enumerate(fh):
                if i % 4 == 1:
                    tl.append(len(line.strip()))
        row("trim_kept_frac", len(tl) / len(seqs))
        if tl:
            tl_sorted = sorted(tl)
            row("insert_length_median", float(tl_sorted[len(tl) // 2]))
            row("insert_length_p10", float(tl_sorted[len(tl) // 10]))
            row("insert_length_p90", float(tl_sorted[9 * len(tl) // 10]))
        for k, v in sorted(collections.Counter(tl).items()):
            lens.append({"dataset": key, "replicate": rep, "run": srr,
                         "metric": "insert_length_count", "length": k,
                         "value": v, "n_reads": len(tl)})

        idx = rrna_index(species, ds["genome_fa"])
        n, a, _u = align_stats(bowtie2, idx, trimmed, args.threads)
        row("rrna_frac_of_trimmed", a / max(n, 1))
        n, a, u = align_stats(bowtie2, ds["index"], trimmed, args.threads)
        row("genome_aligned_frac_of_trimmed", a / max(n, 1))
        row("genome_unique_frac_of_trimmed", u / max(n, 1))
        row("genome_unique_frac_of_raw", u / max(len(seqs), 1))
        for p in (raw_fq, trimmed):
            os.remove(p)
        print("  kept {:.3f}  rRNA {:.3f}  unique {:.3f} of trimmed".format(
            len(tl) / len(seqs), rows[-4]["value"], rows[-2]["value"]))

    out = C.results_path("qc", "read_inspection.tsv")
    done = {r["run"] for r in rows}
    C.merge_tsv(out, rows, lambda r: r["run"] not in done)
    lout = C.results_path("qc", "insert_lengths.tsv")
    if os.path.exists(lout):
        lens = [r for r in C.read_tsv(lout) if r["run"] not in done] + lens
    C.write_tsv(lens, lout)
    print("[inspect] -> {}".format(out))


if __name__ == "__main__":
    main()
