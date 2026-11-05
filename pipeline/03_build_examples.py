import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.archer import save_examples
from data.features import _fold_tracks
from data.coverage import (first_cds_blocks, project_onto_blocks, project_onto_utr,
                           utr_interval, utr_sequence)
from pipeline._common import (CACHE_DIR, WORK_DIR, dataset, ensure_dirs,
                              work_path)


def read_fasta(path):
    seqs, name, chunks = {}, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(chunks)
                name, chunks = line[1:].split()[0], []
            else:
                chunks.append(line.strip())
    if name is not None:
        seqs[name] = "".join(chunks)
    return seqs


def _attr(attrs, key):
    marker = key + ' "'
    i = attrs.find(marker)
    if i < 0:
        return None
    i += len(marker)
    j = attrs.find('"', i)
    return attrs[i:j]


def parse_start_codons(gtf_path):
    by_tx = {}
    duplicated = set()
    cds_blocks = {}
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] not in ("start_codon", "CDS"):
                continue
            tx = _attr(f[8], "transcript_id")
            if tx is None:
                continue
            iv = (int(f[3]) - 1, int(f[4]))
            if f[2] == "CDS":
                cds_blocks.setdefault(tx, []).append(iv)
                continue
            if tx in by_tx:
                duplicated.add(tx)
                continue
            by_tx[tx] = (tx, f[0], f[6]) + iv

    for tx in duplicated:
        by_tx.pop(tx, None)
    return list(by_tx.values()), len(duplicated), cds_blocks


def read_utr_table(path):
    table = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                table[parts[0]] = int(float(parts[1]))
            except ValueError:
                continue
    return table


def build(genome, start_codons, counts, utr_length=50, utr_table=None,
          min_ssu_counts=5, min_utr_length=20, cds_length=100, cds_blocks=None,
          context_flank=10, cds_states=0, skip_chroms=("Mito",),
          restrict_to=None, read_offset=0, structure=False, fold_window=40):
    examples = []
    reasons = Counter()
    cds_blocks = cds_blocks or {}

    for tx, chrom, strand, cds_start, cds_end in start_codons:
        try:
            if chrom not in genome:
                reasons["unknown_chrom"] += 1
                continue
            if chrom in skip_chroms:
                reasons["mitochondrial"] += 1
                continue
            if restrict_to is not None and tx not in restrict_to:
                reasons["not_in_restrict_set"] += 1
                continue

            n_up = utr_length
            if utr_table:
                gene = tx.split("_")[0]
                n_up = utr_table.get(gene, utr_table.get(tx, utr_length))
            if n_up < min_utr_length:
                reasons["utr_too_short"] += 1
                continue

            iv = utr_interval(cds_start, cds_end, strand, n_up, len(genome[chrom]))
            if iv is None:
                reasons["window_off_chromosome"] += 1
                continue
            start, end = iv

            seq = utr_sequence(genome, chrom, strand, start, end)
            main_start_index = len(seq) - 3
            if seq[main_start_index:] != "ATG":
                reasons["start_codon_not_ATG"] += 1
                continue
            if set(seq) - set("ACGT"):
                reasons["ambiguous_bases"] += 1
                continue

            ssu = project_onto_utr(counts["ssu"], chrom, strand, start, end,
                                   offset=read_offset)
            assert len(ssu) == len(seq), "projection length mismatch"
            inp = (project_onto_utr(counts["input"], chrom, strand, start, end)
                   if "input" in counts else None)

            if int(ssu.sum()) < min_ssu_counts:
                reasons["low_coverage"] += 1
                continue

            if cds_states > 0:
                want = 3 + cds_states + context_flank
                eb = first_cds_blocks(cds_blocks.get(tx, []), strand, want)
                if sum(e - s2 for s2, e in eb) < want:
                    reasons["cds_too_short_for_states"] += 1
                    continue
                cds_seq = "".join(utr_sequence(genome, chrom, strand, s2, e)
                                  for s2, e in eb)
                cds_ssu = project_onto_blocks(counts["ssu"], chrom, strand, eb,
                                              offset=read_offset)
                tail_seq = cds_seq[3:3 + cds_states]
                seq = seq + tail_seq
                ssu = np.concatenate([ssu, cds_ssu[3:3 + cds_states]])
                if inp is not None:
                    cds_inp = project_onto_blocks(counts["input"], chrom,
                                                  strand, eb)
                    inp = np.concatenate([inp, cds_inp[3:3 + cds_states]])
                cds_tail_context = cds_seq[3:3 + cds_states + context_flank]
            else:
                cds_tail_context = None

            if cds_tail_context is None:
                cstart = max(0, start - context_flank)
                cend = min(len(genome[chrom]), end + context_flank)
                context = utr_sequence(genome, chrom, strand, cstart, cend)
                coff = (start - cstart) if strand == "+" else (cend - end)
            else:
                if strand == "+":
                    cstart = max(0, start - context_flank)
                    head = utr_sequence(genome, chrom, strand, cstart, end)
                    coff = start - cstart
                else:
                    cend = min(len(genome[chrom]), end + context_flank)
                    head = utr_sequence(genome, chrom, strand, start, cend)
                    coff = cend - end
                context = head + cds_tail_context

            if context[coff:coff + len(seq)] != seq:
                reasons["context_misaligned"] += 1
                continue
            if coff < context_flank or (len(context) - coff - len(seq)) < context_flank:
                reasons["flank_off_chromosome"] += 1
                continue
            if set(context) - set("ACGT"):
                reasons["ambiguous_bases_in_flank"] += 1
                continue

            blocks = first_cds_blocks(cds_blocks.get(tx, []), strand, cds_length)
            if not blocks:
                reasons["no_cds_annotation"] += 1
                continue
            rs_cds = (project_onto_blocks(counts["rs"], chrom, strand, blocks)
                      if "rs" in counts else None)
            inp_cds = (project_onto_blocks(counts["input"], chrom, strand, blocks)
                       if "input" in counts else None)

            examples.append({
                "transcript_id": tx,
                "utr_seq": seq,
                "utr_context_seq": context,
                "context_offset": coff,
                "main_start_index": main_start_index,
                "ssu_counts": ssu.astype(np.int64),
                "chrom": chrom,
                "strand": strand,
                "utr_start": start,
                "utr_end": end,
                "cds_blocks": blocks,
            })
            if structure:
                pr, dg = _fold_tracks(context, fold_window)
                examples[-1]["paired"] = pr.astype(np.float32)
                examples[-1]["local_dG"] = dg.astype(np.float32)
            if inp is not None:
                examples[-1]["input_counts"] = inp.astype(np.int64)
            if rs_cds is not None:
                examples[-1]["rs_cds_counts"] = rs_cds.astype(np.int64)
            if inp_cds is not None:
                examples[-1]["input_cds_counts"] = inp_cds.astype(np.int64)
        except Exception as e:
            reasons["error:" + type(e).__name__] += 1

    return examples, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utr-length", type=int, default=50,
                    help="fixed window upstream of the start codon")
    ap.add_argument("--utr-table",
                    help="TSV of gene_id<TAB>utr_length, overriding the window")
    ap.add_argument("--min-ssu-counts", type=int, default=5,
                    help="drop transcripts with fewer SSU 5' ends than this")
    ap.add_argument("--cds-length", type=int, default=100,
                    help="nt of coding region over which 80S reads are summed "
                         "as the commitment readout")
    ap.add_argument("--cds-states", type=int, default=0,
                    help="extend the CHAIN's states this many nt past the main "
                         "AUG into the coding region, so leaked 40S have "
                         "somewhere to go and the SSU counts there constrain "
                         "read_through (0 = stop at the AUG, as before)")
    ap.add_argument("--context-flank", type=int, default=10,
                    help="nt of real genomic sequence added either side of the "
                         "window for the rate net; must be >= the net's window")
    ap.add_argument("--dataset", default=None,
                    help="which experiment to build from; see DATASETS in "
                         "pipeline/_common.py")
    ap.add_argument("--restrict-to", metavar="TSV",
                    help="keep only the transcript ids in column 0 of this "
                         "file. For human, pass the table from "
                         "scripts/human_utrs.py -- without it every GENCODE "
                         "isoform is built as its own example.")
    ap.add_argument("--structure", action="store_true",
                    help="fold every context with ViennaRNA and store the "
                         "base-pairing and local-dG tracks in the cache. Slow "
                         "once (~29 min for the human set), free thereafter; "
                         "without it every run that needs the tracks has to "
                         "fold again.")
    ap.add_argument("--fold-window", type=int, default=40,
                    help="width of the sliding window for local folding "
                         "energy, assigned to the window centre")
    ap.add_argument("--read-offset", type=int, default=0,
                    help="footprint 5' end -> P site shift for the SSU track, "
                         "in nt. Measure it from the start-codon metagene "
                         "(analysis/recoverability/psite.py): 12 for "
                         "the yeast TCP-seq, 13 for the human 40S.")
    ap.add_argument("--pool-ssu", action="store_true",
                    help="sum every fraction whose name starts with 'ssu' "
                         "into one occupancy track, for replicate runs")
    ap.add_argument("--out", default=os.path.join(CACHE_DIR, "archer_examples.npz"))
    args = ap.parse_args()

    ensure_dirs()
    ds = dataset(args.dataset)
    print("[build] dataset {} ({})".format(ds["name"], ds["genome"]))

    counts_paths = {f: work_path(ds, f, "5ends.npz") for f in ds["fractions"]}
    for frac, p in counts_paths.items():
        if not os.path.exists(p):
            sys.exit("\nMissing {} counts at {}.\n"
                     "Run: python pipeline/02_align_count.py --dataset {}\n"
                     .format(frac, p, ds["name"]))

    from pipeline._common import load_counts_npz
    counts = {frac: load_counts_npz(p) for frac, p in counts_paths.items()}

    reps = [f for f in counts if f.startswith("ssu") and f != "ssu"]
    if args.pool_ssu and reps:
        for frac in reps:
            counts["ssu"].merge(counts.pop(frac))
        print("[build] pooled {} into ssu".format(", ".join(reps)))

    restrict = None
    if args.restrict_to:
        with open(args.restrict_to) as fh:
            restrict = {ln.split("\t")[0].strip() for ln in fh
                        if ln.strip() and not ln.startswith("#")}
        print("[build] restricted to {} transcripts from {}".format(
            len(restrict), args.restrict_to))

    print("[build] reading genome and annotation")
    genome = read_fasta(ds["genome_fa"])
    start_codons, n_split, cds_blocks = parse_start_codons(ds["gtf"])
    print("[build] {} start codons ({} dropped as intron-split)".format(
        len(start_codons), n_split))

    utr_table = read_utr_table(args.utr_table) if args.utr_table else None
    if utr_table:
        print("[build] UTR table: {} genes from {}".format(
            len(utr_table), args.utr_table))
    else:
        print("[build] fixed {} nt window upstream of each start codon".format(
            args.utr_length))

    examples, reasons = build(
        genome, start_codons, counts,
        utr_length=args.utr_length, utr_table=utr_table,
        min_ssu_counts=args.min_ssu_counts, cds_length=args.cds_length,
        cds_blocks=cds_blocks, context_flank=args.context_flank,
        cds_states=args.cds_states, restrict_to=restrict,
        skip_chroms=("Mito", "chrM"), read_offset=args.read_offset,
        structure=args.structure, fold_window=args.fold_window)

    print("[build] {} transcripts pass".format(len(examples)))
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print("[build]   dropped {:6d}  {}".format(n, reason))

    if not examples:
        sys.exit("\nNo transcripts passed.  Check that the counts files are "
                 "non-empty and that --min-ssu-counts is not too strict.\n")

    save_examples(examples, args.out)
    print("[build] -> {}".format(args.out))


if __name__ == "__main__":
    main()
