import argparse
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline._common import (WORK_DIR, cutadapt_cmd, dataset, ensure_dirs,
                              find_fastqs, require, run, up_to_date, work_path)


def _stamp_path(fastq_out):
    return fastq_out + ".maxreads"


def _stored_max_reads(fastq_out):
    try:
        return int(pathlib.Path(_stamp_path(fastq_out)).read_text().strip())
    except (OSError, ValueError):
        return None


def trim_one(cutadapt, fastq_in, fastq_out, max_reads=None, ds=None):
    want = int(max_reads) if max_reads else 0
    if up_to_date(fastq_out, fastq_in) and _stored_max_reads(fastq_out) == want:
        print("[trim] {} is up to date (max_reads {}), skipping".format(
            fastq_out, want or "all"))
        return fastq_out

    source = fastq_in
    if max_reads:
        source = fastq_out + ".head.fastq"
        print("[trim] subsampling first {} reads -> {}".format(max_reads, source))
        _head_fastq(fastq_in, source, max_reads)

    tmp_out = fastq_out + (".part.fastq.gz" if fastq_out.endswith(".gz")
                           else ".part")
    run(cutadapt_cmd(cutadapt, source, tmp_out, ds=ds))
    os.replace(tmp_out, fastq_out)
    pathlib.Path(_stamp_path(fastq_out)).write_text(str(want))
    if source != fastq_in:
        os.remove(source)
    return fastq_out


def _head_fastq(src, dest, n_reads):
    with open(src, "r") as fin, open(dest, "w") as fout:
        for i, line in enumerate(fin):
            if i >= n_reads * 4:
                break
            fout.write(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastq-dir")
    ap.add_argument("--max-reads", type=int, default=None,
                    help="only process the first N reads of each FASTQ")
    args = ap.parse_args()

    ensure_dirs()
    fastqs = find_fastqs(args.fastq_dir)

    cutadapt = None
    want = int(args.max_reads) if args.max_reads else 0
    for frac, path in sorted(fastqs.items()):
        print("[trim] {} fraction: {}".format(frac, path))
        out = os.path.join(WORK_DIR, "{}.trimmed.fastq".format(frac))
        stale = not (up_to_date(out, path) and _stored_max_reads(out) == want)
        if stale and cutadapt is None:
            cutadapt = require("cutadapt", "  pip install cutadapt")
        trim_one(cutadapt, path, out, max_reads=args.max_reads)
        print("[trim] -> {}".format(out))


if __name__ == "__main__":
    main()
