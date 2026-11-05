import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline._common import CACHE_DIR, dataset, ensure_dirs

OUT = os.path.join(CACHE_DIR, "human_utr_lengths.tsv")


def _attr(attrs, key):
    marker = key + ' "'
    i = attrs.find(marker)
    if i < 0:
        return None
    i += len(marker)
    return attrs[i:attrs.find('"', i)]


def parse(gtf_path, tag="MANE_Select", biotype="protein_coding"):
    tx = {}
    seen_genes = set()
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] not in ("transcript", "UTR", "start_codon"):
                continue
            a = f[8]
            if '"' + tag + '"' not in a:
                continue
            tid = _attr(a, "transcript_id")
            if tid is None:
                continue
            if f[2] == "transcript":
                if _attr(a, "transcript_type") != biotype:
                    continue
                seen_genes.add(_attr(a, "gene_id"))
                tx[tid] = {"chrom": f[0], "strand": f[6], "utr": [],
                           "start_codon": None,
                           "gene": _attr(a, "gene_name") or _attr(a, "gene_id")}
            elif tid in tx:
                iv = (int(f[3]) - 1, int(f[4]))
                if f[2] == "UTR":
                    tx[tid]["utr"].append(iv)
                else:
                    cur = tx[tid]["start_codon"]
                    tx[tid]["start_codon"] = (
                        iv if cur is None else (min(cur[0], iv[0]), max(cur[1], iv[1])))
    return tx, len(seen_genes)


def five_prime(rec):
    sc, blocks, strand = rec["start_codon"], rec["utr"], rec["strand"]
    if sc is None or not blocks:
        return None
    if strand == "+":
        five = [b for b in blocks if b[1] <= sc[0]]
        if not five:
            return None
        last = max(five, key=lambda b: b[1])
    else:
        five = [b for b in blocks if b[0] >= sc[1]]
        if not five:
            return None
        last = min(five, key=lambda b: b[0])
    return last[1] - last[0], sum(e - s for s, e in five), len(five)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="human-40s")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--tag", default="MANE_Select",
                    help="GTF tag selecting one transcript per gene")
    ap.add_argument("--min-length", type=int, default=20,
                    help="drop leaders whose usable window is shorter than "
                         "this; the chain needs room for the rate net window")
    args = ap.parse_args()

    ensure_dirs()
    ds = dataset(args.dataset)
    if not os.path.exists(ds["gtf"]):
        sys.exit("\nNo annotation at {}.\nRun: python scripts/"
                 "download_reference.py --dataset {} --skip-index\n"
                 .format(ds["gtf"], ds["name"]))

    print("[utr] reading {}".format(ds["gtf"]))
    tx, n_genes = parse(ds["gtf"], tag=args.tag)
    print("[utr] {} {} transcripts over {} genes".format(len(tx), args.tag, n_genes))

    rows, drop = [], collections.Counter()
    spliced_n = 0
    for tid, rec in sorted(tx.items()):
        fp = five_prime(rec)
        if fp is None:
            drop["no_5utr_or_start_codon"] += 1
            continue
        window, spliced, n_ex = fp
        if n_ex > 1:
            spliced_n += 1
        if window < args.min_length:
            drop["window_under_{}nt".format(args.min_length)] += 1
            continue
        rows.append((tid, window, spliced, n_ex, rec["gene"],
                     rec["chrom"], rec["strand"]))

    with open(args.out, "w") as fh:
        fh.write("transcript_id\tutr_window\tutr_spliced\tn_utr_exons\t"
                 "gene\tchrom\tstrand\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")

    if rows:
        w = sorted(r[1] for r in rows)
        s = sorted(r[2] for r in rows)
        lost = sum(r[2] - r[1] for r in rows)
        total = sum(r[2] for r in rows)
        print("[utr] {} transcripts kept".format(len(rows)))
        for k, n in drop.most_common():
            print("[utr]   dropped {:6d}  {}".format(n, k))
        print("[utr] window  median {:4d}  p10 {:3d}  p90 {:4d}".format(
            w[len(w) // 2], w[len(w) // 10], w[9 * len(w) // 10]))
        print("[utr] spliced median {:4d}  p10 {:3d}  p90 {:4d}".format(
            s[len(s) // 2], s[len(s) // 10], s[9 * len(s) // 10]))
        print("[utr] {} of {} leaders are spliced; truncating at the last "
              "intron discards {:.0f}% of total leader sequence".format(
                  spliced_n, len(rows), 100.0 * lost / max(total, 1)))
    print("[utr] -> {}".format(args.out))
    print("[utr] feed this to 03_build_examples.py --utr-table")


if __name__ == "__main__":
    main()
