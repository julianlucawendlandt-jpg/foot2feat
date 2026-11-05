import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(HERE, "gkaf165_supplemental_files", "Akirtava_et_al_2024_Supp_Data_Rev1.xlsx")
OUT = os.path.join(HERE, "akirtava_scer_wt_downstream.tsv")
EXCL = os.path.join(HERE, "akirtava_scer_wt_downstream_exclusions.tsv")
TEFF_MIN_DEPTH = 5000

ID_RE = re.compile(r"^(?P<orf>[^|]+)\|(?P<chrom>[^;]+);(?P<tss>\d+);(?P<atg>\d+)\|(?P<allele>.+)$")


def parse_id(s):
    m = ID_RE.match(str(s).strip())
    if not m:
        return None, "unparsable_id"
    d = m.groupdict()
    if d["chrom"].startswith("Spar"):
        return None, "s_paradoxus"
    if not d["chrom"].startswith("chr"):
        return None, "non_chr_contig"
    if d["allele"].strip().upper() != "WT":
        return None, "non_wt_allele"
    d["chrom"] = d["chrom"][3:]
    d["tss"], d["atg"] = int(d["tss"]), int(d["atg"])
    d["strand"] = "+" if d["tss"] < d["atg"] else "-"
    return d, ""


def num(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def main():
    if not os.path.exists(XLSX):
        sys.exit("\nWorkbook not found at {}\n".format(XLSX))
    import openpyxl
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)

    def sheet(name):
        it = wb[name].iter_rows(values_only=True)
        hdr = [str(h).strip() if h is not None else "" for h in next(it)]
        for r in it:
            yield dict(zip(hdr, r))

    excl = []
    t1, t2, t3 = {}, {}, {}
    for r in sheet("Table_1"):
        d, why = parse_id(r.get("@UTR"))
        if d is None:
            excl.append((r.get("@UTR"), "Table_1", why))
            continue
        t1[r["@UTR"]] = r
    for r in sheet("Table_2"):
        d, why = parse_id(r.get("Construct"))
        if d is None:
            excl.append((r.get("Construct"), "Table_2", why))
            continue
        t2[r["Construct"]] = r
    for r in sheet("Table_3"):
        d, why = parse_id(r.get("Gene"))
        if d is None:
            excl.append((r.get("Gene"), "Table_3", why))
            continue
        t3[r["Gene"]] = r
    wb.close()

    cols = ["id", "orf", "chrom", "strand", "tss", "atg", "tl_length", "uAUG", "utr_seq",
            "seq_source", "MeanYFP", "meanyfp_valid", "MeanRNA", "R1_TEff", "R2_TEff", "TEff",
            "R1_trans", "R1_untrans", "R2_trans", "R2_untrans", "R1_depth", "R2_depth",
            "combined_depth", "teff_primary", "LSMKozakStart"]
    rows = []
    for k in sorted(set(t1) | set(t2) | set(t3)):
        d, _ = parse_id(k)
        a, b, c = t1.get(k), t2.get(k), t3.get(k)
        s1 = str((a or {}).get("UTRseq") or "").strip().upper()
        s3 = str((c or {}).get("UTRseq") or "").strip().upper()
        if s1 and s3 and s1 != s3:
            excl.append((k, "all", "assayed_sequence_differs_between_tables"))
            continue
        seq = s1 or s3
        if not seq:
            excl.append((k, "all", "no_assayed_sequence (PoLib-only construct)"))
            continue
        src = "Table_1" if s1 else "Table_3"
        r1t, r2t = num((b or {}).get("R1-TEff")), num((b or {}).get("R2-TEff"))
        r1d = (num((b or {}).get("R1-trans")) or 0) + (num((b or {}).get("R1-untrans")) or 0) if b else None
        r2d = (num((b or {}).get("R2-trans")) or 0) + (num((b or {}).get("R2-untrans")) or 0) if b else None
        teffs = [v for v in (r1t, r2t) if v is not None]
        teff = sum(teffs) / len(teffs) if teffs else None
        comb = (r1d or 0) + (r2d or 0) if b else None
        yfp3 = num((c or {}).get("MeanYFP"))
        yfp = yfp3 if yfp3 is not None else num((a or {}).get("MeanYFP"))
        row = {"id": k, "orf": d["orf"], "chrom": d["chrom"], "strand": d["strand"], "tss": d["tss"],
               "atg": d["atg"], "tl_length": len(seq),
               "uAUG": sum(1 for i in range(len(seq) - 2) if seq[i:i + 3] == "ATG"),
               "utr_seq": seq, "seq_source": src, "MeanYFP": yfp,
               "meanyfp_valid": int(c is not None and yfp3 is not None),
               "MeanRNA": num((c or a or {}).get("MeanRNA")), "R1_TEff": r1t, "R2_TEff": r2t,
               "TEff": teff, "R1_trans": num((b or {}).get("R1-trans")),
               "R1_untrans": num((b or {}).get("R1-untrans")), "R2_trans": num((b or {}).get("R2-trans")),
               "R2_untrans": num((b or {}).get("R2-untrans")), "R1_depth": r1d, "R2_depth": r2d,
               "combined_depth": comb,
               "teff_primary": int(teff is not None and comb is not None and comb >= TEFF_MIN_DEPTH),
               "LSMKozakStart": num((c or {}).get("LSMKozakStart"))}
        if teff is not None and not row["teff_primary"]:
            excl.append((k, "TEff", "combined_PoLib_depth_below_{}".format(TEFF_MIN_DEPTH)))
        if yfp is not None and not row["meanyfp_valid"]:
            excl.append((k, "MeanYFP", "not_in_published_modelling_table"))
        rows.append(row)

    with open(OUT, "w", newline="") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")
    with open(EXCL, "w", newline="") as fh:
        fh.write("id\tscope\treason\n")
        for k, scope, why in excl:
            fh.write("{}\t{}\t{}\n".format(k, scope, why))
    from collections import Counter
    print("[akirtava] {} S.cer WT constructs with an assayed sequence".format(len(rows)))
    print("[akirtava] TEff present {}, primary (combined depth >= {}) {}".format(
        sum(r["TEff"] is not None for r in rows), TEFF_MIN_DEPTH, sum(r["teff_primary"] for r in rows)))
    print("[akirtava] MeanYFP-valid {}".format(sum(r["meanyfp_valid"] for r in rows)))
    for (scope, why), n in sorted(Counter((s, w) for _k, s, w in excl).items()):
        print("[akirtava]   excluded {:>6}  {:<8} {}".format(n, scope, why))
    print("[akirtava] -> {}\n[akirtava] -> {}".format(OUT, EXCL))


if __name__ == "__main__":
    main()
