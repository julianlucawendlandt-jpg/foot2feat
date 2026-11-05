import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline._common import CACHE_DIR, GTF, WORK_DIR, ensure_dirs, load_counts_npz

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "pipeline"))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_bx", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pipeline", "03_build_examples.py"))
_bx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bx)


def cds_extents(start_codons, cds_blocks):
    by_key = {}
    for tx, chrom, strand, _cs, _ce in start_codons:
        blocks = cds_blocks.get(tx)
        if not blocks:
            continue
        lo = min(b[0] for b in blocks)
        hi = max(b[1] for b in blocks)
        by_key.setdefault((chrom, strand), []).append((lo, hi))
    for k in by_key:
        by_key[k].sort()
    return by_key


def upstream_limit(extents, chrom, strand, cds_start, cds_end, max_len):
    spans = extents.get((chrom, strand), [])
    if strand == "+":
        best = max((hi for lo, hi in spans if hi <= cds_start), default=None)
        room = cds_start - best if best is not None else max_len
    else:
        best = min((lo for lo, hi in spans if lo >= cds_end), default=None)
        room = best - cds_end if best is not None else max_len
    return max(0, min(max_len, room))


def call_one(cov, chrom, strand, cds_start, cds_end, limit,
             min_gap=10, min_start_cov=2):
    if limit <= 0:
        return None
    arr = cov[(chrom, strand)]
    if strand == "+":
        lo, hi = max(0, cds_start - limit), cds_start
        upstream = arr[lo:hi][::-1]
    else:
        lo, hi = cds_end, min(len(arr), cds_end + limit)
        upstream = arr[lo:hi]
    if upstream.size == 0 or upstream[:3].max() < min_start_cov:
        return None

    gap = 0
    for i, v in enumerate(upstream):
        if v > 0:
            gap = 0
        else:
            gap += 1
            if gap >= min_gap:
                return i + 1 - min_gap
    return int(upstream.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage",
                    default=os.path.join(WORK_DIR, "input.coverage.npz"))
    ap.add_argument("--out", default=os.path.join(CACHE_DIR, "utr_lengths.tsv"))
    ap.add_argument("--min-gap", type=int, default=10,
                    help="consecutive uncovered nt that end the transcript")
    ap.add_argument("--min-start-cov", type=int, default=2,
                    help="coverage required just upstream of the AUG to call at all")
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=300)
    args = ap.parse_args()

    ensure_dirs()
    if not os.path.exists(args.coverage):
        sys.exit(
            "\nNo Input coverage track at {}.\n"
            "Run: python pipeline/02_align_count.py   (it writes this alongside\n"
            "the 5'-end counts; the track is only built for the Input fraction)\n"
            .format(args.coverage))

    cov = load_counts_npz(args.coverage)
    start_codons, _, cds_blocks = _bx.parse_start_codons(GTF)
    extents = cds_extents(start_codons, cds_blocks)

    calls, reasons = {}, {"no_coverage": 0, "blocked_by_neighbour": 0,
                          "too_short": 0, "mitochondrial": 0}
    for tx, chrom, strand, cs, ce in start_codons:
        if chrom == "Mito":
            reasons["mitochondrial"] += 1
            continue
        if (chrom, strand) not in cov:
            reasons["no_coverage"] += 1
            continue
        limit = upstream_limit(extents, chrom, strand, cs, ce, args.max_len)
        if limit <= 0:
            reasons["blocked_by_neighbour"] += 1
            continue
        n = call_one(cov, chrom, strand, cs, ce, limit,
                     min_gap=args.min_gap, min_start_cov=args.min_start_cov)
        if n is None:
            reasons["no_coverage"] += 1
            continue
        if n < args.min_len:
            reasons["too_short"] += 1
            continue
        calls[tx.split("_")[0]] = int(min(n, args.max_len))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("# gene_id\tutr_length -- called from Input read coverage\n")
        for gene in sorted(calls):
            fh.write("{}\t{}\n".format(gene, calls[gene]))

    lens = np.array(sorted(calls.values()))
    print("[utr] called {} genes".format(len(calls)))
    if len(lens):
        print("[utr] length: median {:.0f}  IQR {:.0f}-{:.0f}  min {}  max {}".format(
            np.median(lens), np.percentile(lens, 25), np.percentile(lens, 75),
            lens.min(), lens.max()))
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        if v:
            print("[utr]   no call {:5d}  {}".format(v, k))
    print("[utr] -> {}".format(args.out))
    print("[utr] use it with: python pipeline/03_build_examples.py --utr-table {}"
          .format(args.out))


if __name__ == "__main__":
    main()
