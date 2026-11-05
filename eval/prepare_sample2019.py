import argparse
import csv
import gzip
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sample2019.tsv")

MIN_READS = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True,
                    help="GSM4084997_varying_length_25to100.csv.gz")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--sets", default="human",
                    help="comma-separated sets to keep, or 'all'. Default "
                         "'human': the native leaders, the only set that is "
                         "real biology rather than designed sequence.")
    ap.add_argument("--min-reads", type=int, default=MIN_READS)
    args = ap.parse_args()

    if not os.path.exists(args.src):
        sys.exit("\nNot found: {}\nSee the module docstring for the URL.\n".format(args.src))
    keep = None if args.sets == "all" else {s.strip() for s in args.sets.split(",")}

    opener = gzip.open if args.src.endswith(".gz") else open
    kept, stats = [], {"rows": 0, "wrong_set": 0, "low_reads": 0, "bad": 0}
    with opener(args.src, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            stats["rows"] += 1
            st = (r.get("set") or "").strip()
            if keep is not None and st not in keep:
                stats["wrong_set"] += 1
                continue
            seq = (r.get("utr") or "").upper().replace("U", "T")
            try:
                rl = float(r["rl"])
                reads = float(r.get("total_reads") or r.get("total") or 0)
            except (KeyError, TypeError, ValueError):
                stats["bad"] += 1
                continue
            if rl != rl or not seq or set(seq) - set("ACGT"):
                stats["bad"] += 1
                continue
            if reads < args.min_reads:
                stats["low_reads"] += 1
                continue
            kept.append({"id": "{}:{}".format(st or "row", len(kept)),
                         "set": st, "utr_seq": seq, "rl": rl,
                         "length": len(seq), "total_reads": int(reads)})

    if not kept:
        sys.exit("\nNothing kept. Try --sets all or a lower --min-reads.\n")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(kept[0]), delimiter="\t")
        w.writeheader()
        w.writerows(kept)

    import statistics as st_
    rls = sorted(k["rl"] for k in kept)
    lens = sorted(k["length"] for k in kept)
    print("[prep] read {rows} rows ({wrong_set} other sets, {low_reads} under "
          "{m} reads, {bad} unusable)".format(m=args.min_reads, **stats))
    print("[prep] kept {}".format(len(kept)))
    print("[prep] rl      median {:.2f}  p10 {:.2f}  p90 {:.2f}".format(
        rls[len(rls) // 2], rls[len(rls) // 10], rls[9 * len(rls) // 10]))
    print("[prep] length  median {}  p10 {}  p90 {}".format(
        lens[len(lens) // 2], lens[len(lens) // 10], lens[9 * len(lens) // 10]))
    print("[prep] -> {}".format(args.out))


if __name__ == "__main__":
    main()
