import argparse
import gzip
import io
import os
import shutil
import subprocess
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.coverage import accumulate_tracks
from pipeline._common import (BOWTIE2_HINT, WORK_DIR, cutadapt_cmd, dataset,
                              ensure_dirs, find_cutadapt, find_fastqs, require,
                              up_to_date, work_path)


def chrom_lengths_from_fasta(path):
    lengths, name, n = {}, None, 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    lengths[name] = n
                name, n = line[1:].split()[0], 0
            else:
                n += len(line.strip())
    if name is not None:
        lengths[name] = n
    return lengths


def _stored_offset(npz_path):
    try:
        with np.load(npz_path, allow_pickle=True) as z:
            return int(z["_offset"])
    except Exception:
        return None


def _feed(path, dst, max_reads=None):
    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rb") as src:
            if max_reads is None:
                shutil.copyfileobj(src, dst, 1 << 20)
            else:
                for i, line in enumerate(src):
                    if i >= 4 * max_reads:
                        break
                    dst.write(line)
    except BrokenPipeError:
        pass
    finally:
        try:
            dst.close()
        except (OSError, ValueError):
            pass


def align_and_count(bowtie2, fastq, out_npz, chrom_lengths, offset=0,
                    min_mapq=10, threads=4, coverage_npz=None, cutadapt=None,
                    ds=None, max_reads=None):
    have_counts = up_to_date(out_npz, fastq) and _stored_offset(out_npz) == offset
    have_cov = coverage_npz is None or up_to_date(coverage_npz, fastq)
    reads_npz = out_npz.replace("5ends.npz", "reads.npz")
    have_reads = up_to_date(reads_npz, fastq)
    if have_counts and have_cov and have_reads:
        print("[align] {} is up to date (offset {}), skipping".format(out_npz, offset))
        return out_npz

    ds = ds or dataset()
    cmd = [
        bowtie2,
        "-x", ds["index"],
        "-U", "-",
        "--no-unal",
        "-L", "18",
        "-N", "0",
        "--threads", str(threads),
    ]
    print("[align] {}{} < {}".format(
        "cutadapt | " if cutadapt else "", " ".join(cmd), fastq))

    trimmer, pump, scratch = None, None, None
    if cutadapt:
        src = fastq
        if max_reads:
            scratch = out_npz + ".subsample.fastq.gz"
            print("[align] subsampling {} reads -> {}".format(max_reads, scratch))
            with gzip.open(scratch, "wb", compresslevel=1) as fh:
                _feed(fastq, fh, max_reads)
            src = scratch
        trimmer = subprocess.Popen(
            cutadapt_cmd(cutadapt, src, "-", threads=max(1, threads // 2), ds=ds),
            stdout=subprocess.PIPE, bufsize=1 << 20)
        proc = subprocess.Popen(cmd, stdin=trimmer.stdout,
                                stdout=subprocess.PIPE, bufsize=1 << 20)
        trimmer.stdout.close()
    else:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, bufsize=1 << 20)
        pump = threading.Thread(target=_feed, args=(fastq, proc.stdin, max_reads),
                                daemon=True)
        pump.start()

    sam = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace")
    counts, cov, stats, reads = accumulate_tracks(
        sam, chrom_lengths, offset=offset, min_mapq=min_mapq,
        coverage=coverage_npz is not None, keep_reads=True)
    sam.close()
    if proc.wait() != 0:
        raise RuntimeError("bowtie2 exited with code {}".format(proc.returncode))
    if pump is not None:
        pump.join()
    if trimmer is not None and trimmer.wait() != 0:
        raise RuntimeError("cutadapt exited with code {}".format(trimmer.returncode))
    if scratch and os.path.exists(scratch):
        os.remove(scratch)

    print("[align] {}".format(stats))

    if cov is not None:
        tmpc = coverage_npz + ".part.npz"
        np.savez_compressed(tmpc, **cov.to_npz())
        os.replace(tmpc, coverage_npz)
        print("[align] -> {}".format(coverage_npz))

    tmpr = reads_npz + ".part.npz"
    flat = {}
    for (chrom, strand), (pos, span) in reads.items():
        flat["{}|{}|pos".format(chrom, strand)] = pos
        flat["{}|{}|span".format(chrom, strand)] = span
    np.savez_compressed(tmpr, **flat, _offset=np.array(offset))
    os.replace(tmpr, reads_npz)
    print("[align] -> {}".format(reads_npz))

    tmp = out_npz + ".part.npz"
    np.savez_compressed(
        tmp,
        **counts.to_npz(),
        _stats=np.array(sorted(stats.items()), dtype=object),
        _offset=np.array(offset),
    )
    os.replace(tmp, out_npz)
    return out_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=0,
                    help="shift the 5'-end assignment downstream by N nt "
                         "(0 = raw 5' end; 12 = SSU P site)")
    ap.add_argument("--min-mapq", type=int, default=10,
                    help="bowtie2 MAPQ floor; 10 is the usual unique threshold")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dataset", default=None,
                    help="which experiment to process; see DATASETS in "
                         "pipeline/_common.py. Picks the adapter, the minimum "
                         "read length, the reference and the run accessions, "
                         "all of which differ between species.")
    ap.add_argument("--stream", action="store_true",
                    help="trim and align in one pipe, straight from the raw "
                         "FASTQ. No trimmed intermediate is written at all, "
                         "which saves ~13 GB for a 2.5 MB result. Makes "
                         "01_trim.py unnecessary.")
    ap.add_argument("--fastq-dir", help="where the raw FASTQs live (--stream)")
    ap.add_argument("--max-reads", type=int,
                    help="stop after N raw reads per fraction. For measuring "
                         "the alignment rate before committing hours to a "
                         "mammalian run; the counts it writes are a subsample, "
                         "so --out them somewhere disposable.")
    ap.add_argument("--no-coverage", action="store_true",
                    help="skip the Input read-coverage track (used only by "
                         "scripts/call_utr_boundaries.py)")
    args = ap.parse_args()

    ensure_dirs()
    ds = dataset(args.dataset)
    print("[align] dataset {} ({}), adapter {}, min length {}".format(
        ds["name"], ds["genome"], ds["adapter"], ds["min_length"]))
    bowtie2 = require("bowtie2-align-s", BOWTIE2_HINT, fallbacks=("bowtie2",))
    if not os.path.exists(ds["index"] + ".1.bt2"):
        sys.exit("\nNo bowtie2 index at {}.\n"
                 "Run: python scripts/download_reference.py --dataset {}\n"
                 .format(ds["index"], ds["name"]))

    chrom_lengths = chrom_lengths_from_fasta(ds["genome_fa"])
    print("[align] {} sequences, {:.1f} Mb".format(
        len(chrom_lengths), sum(chrom_lengths.values()) / 1e6))

    cutadapt = find_cutadapt() if args.stream else None
    raw = find_fastqs(args.fastq_dir, ds["fractions"]) if args.stream else {}

    for frac in ds["fractions"]:
        if args.stream:
            fastq = raw[frac]
        else:
            cands = [work_path(ds, frac, "trimmed.fastq" + e) for e in (".gz", "")]
            fastq = next((c for c in cands if os.path.exists(c)), None)
            if fastq is None:
                sys.exit("\nMissing {}.\nRun: python pipeline/01_trim.py, or "
                         "pass --stream to skip the intermediate entirely.\n"
                         .format(cands[1]))
        out = work_path(ds, frac,
                        "sample.5ends.npz" if args.max_reads else "5ends.npz")
        cov = (work_path(ds, frac, "coverage.npz")
               if frac == "input" and not args.no_coverage else None)
        align_and_count(bowtie2, fastq, out, chrom_lengths, offset=args.offset,
                        min_mapq=args.min_mapq, threads=args.threads,
                        coverage_npz=cov, cutadapt=cutadapt, ds=ds,
                        max_reads=args.max_reads)
        print("[align] -> {}".format(out))


if __name__ == "__main__":
    main()
