import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.recoverability import config as C
from data.coverage import first_cds_blocks, project_onto_blocks, project_onto_utr, utr_interval, utr_sequence
from data.features import _fold_tracks, start_positions, uorf_returns
from pipeline._common import dataset, load_counts_npz, work_path

STRUCTURE_METHOD = ("ViennaRNA {ver}: MFE pairing of the stored sequence (leader "
                    "window + main AUG + {ext} nt exon-aware CDS), no genomic "
                    "flank; local dG in {w}-nt windows centred on each position, "
                    "clipped to the stored sequence")


def _load_03():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_bx", os.path.join(C.ROOT, "pipeline", "03_build_examples.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def annotation_path(species):
    return os.path.join(C.PANEL_DIR, "annotation_{}.npz".format(species))


def counts_path(key):
    return os.path.join(C.PANEL_DIR, "counts_{}.npz".format(key))


def _save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, data=np.array([obj], dtype=object))
    os.replace(tmp, path)


def _load(path):
    with np.load(path, allow_pickle=True) as z:
        return z["data"][0]


def human_windows(ds):
    from scripts.human_utrs import five_prime, parse
    tx, _ = parse(ds["gtf"])
    out = {}
    for tid, rec in tx.items():
        fp = five_prime(rec)
        if fp is None:
            continue
        window, spliced, n_ex = fp
        out[tid] = (window, {"gene_id": rec["gene"], "utr_spliced": spliced,
                             "n_utr_exons": n_ex})
    return out


def yeast_windows(start_codons, cds_blocks, chrom_len):
    from scripts.call_utr_boundaries import call_one, cds_extents, upstream_limit
    cov_path = work_path(dataset("archer-yeast"), "input", "coverage.npz")
    if not os.path.exists(cov_path):
        sys.exit("\nNo Input coverage at {} -- align archer-yeast first.\n".format(cov_path))
    cov = load_counts_npz(cov_path)
    dense = {}
    for (chrom, strand), (p, v) in cov.data.items():
        a = np.zeros(chrom_len[chrom] + 1, dtype=np.int32)
        a[p] = v
        dense[(chrom, strand)] = a
    extents = cds_extents(start_codons, cds_blocks)
    out = {}
    for tx, chrom, strand, cs, ce in start_codons:
        if chrom == "Mito" or (chrom, strand) not in dense:
            continue
        limit = upstream_limit(extents, chrom, strand, cs, ce, 300)
        if limit <= 0:
            continue
        n = call_one(dense, chrom, strand, cs, ce, limit, min_gap=10, min_start_cov=2)
        if n is None or n < C.MIN_LEADER_LEN:
            continue
        out[tx] = (int(min(n, 300)), {"gene_id": tx.split("_")[0]})
    return out


def _fold_one(args):
    seq, w = args
    return _fold_tracks(seq, w)


def build_annotation(species, processes=4):
    bx = _load_03()
    preset = {"human": "human-eif3b-replicated", "yeast": "archer-yeast"}[species]
    ds = dataset(preset)
    print("[panel] reading genome {}".format(ds["genome_fa"]))
    genome = bx.read_fasta(ds["genome_fa"])
    start_codons, n_split, cds_blocks = bx.parse_start_codons(ds["gtf"])
    windows = (human_windows(ds) if species == "human" else
               yeast_windows(start_codons, cds_blocks, {c: len(s) for c, s in genome.items()}))
    print("[panel] {} start codons ({} intron-split dropped), {} leader windows".format(
        len(start_codons), n_split, len(windows)))

    recs, reasons = [], {}

    def drop(r):
        reasons[r] = reasons.get(r, 0) + 1

    for tx, chrom, strand, cs, ce in start_codons:
        if tx not in windows:
            drop("no_leader_window")
            continue
        if chrom in ("Mito", "chrM"):
            drop("mitochondrial")
            continue
        n_up, meta = windows[tx]
        if n_up < C.MIN_LEADER_LEN:
            drop("leader_too_short")
            continue
        iv = utr_interval(cs, ce, strand, n_up, len(genome[chrom]))
        if iv is None:
            drop("window_off_chromosome")
            continue
        start, end = iv
        lead = utr_sequence(genome, chrom, strand, start, end)
        if lead[-3:] != "ATG":
            drop("start_codon_not_ATG")
            continue
        eb = first_cds_blocks(cds_blocks.get(tx, []), strand, 3 + C.CDS_EXTENSION)
        if sum(e - s for s, e in eb) < 3 + C.CDS_EXTENSION:
            drop("cds_shorter_than_extension")
            continue
        cds = "".join(utr_sequence(genome, chrom, strand, s, e) for s, e in eb)
        seq = lead + cds[3:3 + C.CDS_EXTENSION]
        if set(seq) - set("ACGT"):
            drop("ambiguous_bases")
            continue
        msi = len(lead) - 3
        rec = {"transcript_id": tx, "gene_id": meta.get("gene_id", tx), "chrom": chrom,
               "strand": strand, "utr_start": start, "utr_end": end, "cds_blocks": eb,
               "seq": seq, "msi": msi, "length": msi}
        rec.update({k: v for k, v in meta.items() if k != "gene_id"})
        sm = start_positions(seq)
        rec["aug_mask"] = sm
        rec["near_cognate_mask"] = start_positions(seq, near_cognate=True) & ~sm
        ret, ulen, gap = uorf_returns(seq[:msi + 3], sm[:msi + 3], msi)
        rec["uorf_return"], rec["uorf_len"], rec["uorf_gap"] = ret, ulen, gap
        recs.append(rec)

    print("[panel] {} transcripts; folding with {} processes".format(len(recs), processes))
    t0 = time.time()
    with mp.Pool(processes) as pool:
        folded = pool.map(_fold_one, [(r["seq"], C.FOLD_WINDOW) for r in recs], chunksize=20)
    for r, (pr, dg) in zip(recs, folded):
        r["paired"], r["local_dG"] = pr.astype(np.float32), dg.astype(np.float32)
    print("[panel] folded in {:.0f}s".format(time.time() - t0))

    import RNA
    meta = {"species": species, "n_transcripts": len(recs), "dropped": reasons,
            "intron_split_start_codons": n_split, "annotation": C.ANNOTATION[species],
            "cds_extension": C.CDS_EXTENSION, "min_leader_len": C.MIN_LEADER_LEN,
            "structure": STRUCTURE_METHOD.format(ver=RNA.__version__, ext=C.CDS_EXTENSION,
                                                 w=C.FOLD_WINDOW),
            "leader_definition": ("last 5'UTR exon of the MANE Select transcript"
                                  if species == "human" else
                                  "called from Input read coverage (min_gap 10, "
                                  "min_start_cov 2, max 300 nt, clipped at the "
                                  "nearest same-strand upstream CDS); genes with no "
                                  "call are excluded, not given a fixed window")}
    _save({"meta": meta, "records": recs}, annotation_path(species))
    C.update_manifest("panels", {"annotation_" + species: meta}, merge=True)
    print("[panel] -> {}".format(annotation_path(species)))
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print("[panel]   dropped {:6d}  {}".format(v, k))
    return recs


def build_counts(key):
    d = C.DATASETS[key]
    ds = dataset(d["preset"])
    offsets = C.measured("psite_offsets", {})
    if key not in offsets:
        sys.exit("\nNo measured P-site offset for {}.  Run:\n"
                 "  python -m analysis.recoverability.01_qc --stage offsets\n".format(key))
    from analysis.recoverability import psite
    rule = [tuple(x) for x in offsets[key]["rule"]]
    ann = _load(annotation_path(d["species"]))
    counts, totals = {}, {}
    for rep, r in d["replicates"].items():
        reads = psite.load_reads(work_path(ds, r["fraction"], "reads.npz"))
        totals[rep] = int(sum(len(p_) for p_, _s in reads.values()))
        track = psite.apply_rule(reads, rule)
        off = 0
        arrs = []
        for t in ann["records"]:
            lead = project_onto_utr(track, t["chrom"], t["strand"], t["utr_start"],
                                    t["utr_end"], offset=off)
            ext = project_onto_blocks(track, t["chrom"], t["strand"], t["cds_blocks"],
                                      offset=off)[3:3 + C.CDS_EXTENSION]
            v = np.concatenate([lead, ext]).astype(np.int32)
            assert len(v) == len(t["seq"]), t["transcript_id"]
            arrs.append(v)
        counts[rep] = arrs
        print("[panel] {} {}: {} reads on {} panel transcripts (of {} unique 5' ends)"
              .format(key, rep, sum(int(a[:t["msi"] + 3].sum()) for a, t in
                                    zip(arrs, ann["records"])),
                      len(arrs), totals[rep]))
    meta = {"dataset": key, "psite_rule": rule, "replicates": list(counts),
            "unique_5ends_total": totals, "annotation": annotation_path(d["species"]),
            "count_tracks": {rep: work_path(ds, r["fraction"], "reads.npz")
                             for rep, r in d["replicates"].items()}}
    _save({"meta": meta, "counts": counts,
           "ids": [t["transcript_id"] for t in ann["records"]]}, counts_path(key))
    C.update_manifest("panels", {"counts_" + key: meta}, merge=True)
    return counts


def load_panel(key):
    d = C.DATASETS[key]
    ann = _load(annotation_path(d["species"]))
    cnt = _load(counts_path(key))
    assert cnt["ids"] == [t["transcript_id"] for t in ann["records"]], \
        "counts were built on a different annotation; rebuild them"
    recs = []
    for i, t in enumerate(ann["records"]):
        r = dict(t)
        r["counts"] = {rep: cnt["counts"][rep][i] for rep in cnt["counts"]}
        recs.append(r)
    meta = dict(ann["meta"])
    meta.update(cnt["meta"])
    return recs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", choices=("human", "yeast"))
    ap.add_argument("--dataset", choices=sorted(C.DATASETS))
    ap.add_argument("--processes", type=int, default=4)
    args = ap.parse_args()
    C.ensure_layout()
    if args.species:
        build_annotation(args.species, processes=args.processes)
    if args.dataset:
        build_counts(args.dataset)


if __name__ == "__main__":
    main()
