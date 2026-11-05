import argparse
import gzip
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

ENSEMBL_RELEASE = 115
ENSEMBL = "https://ftp.ensembl.org/pub/release-{r}".format(r=ENSEMBL_RELEASE)
GENCODE_RELEASE = 49
GENCODE = ("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
           "release_{r}".format(r=GENCODE_RELEASE))

REFERENCES = {
    "sacCer3": {
        "genome_url": (ENSEMBL + "/fasta/saccharomyces_cerevisiae/dna/"
                       "Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa.gz"),
        "gtf_url": (ENSEMBL + "/gtf/saccharomyces_cerevisiae/"
                    "Saccharomyces_cerevisiae.R64-1-1.{r}.gtf.gz"
                    .format(r=ENSEMBL_RELEASE)),
        "note": "12 Mb; the index builds in about a minute",
    },
    "GRCh38": {
        "genome_url": GENCODE + "/GRCh38.primary_assembly.genome.fa.gz",
        "gtf_url": (GENCODE + "/gencode.v{r}.annotation.gtf.gz"
                    .format(r=GENCODE_RELEASE)),
        "note": "3.1 Gb; the index needs a few hours, several GB of RAM, "
                "and ~4.3 GB of disk",
    },
}


def reference_for(dataset_name):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline._common import dataset
    g = dataset(dataset_name)["genome"]
    if g not in REFERENCES:
        sys.exit("\nNo reference URLs for genome {!r}. Add it to REFERENCES "
                 "in {}.\n".format(g, os.path.basename(__file__)))
    return g, REFERENCES[g]


def _blocked(url, err):
    sys.exit(
        "\nCould not download:\n  {url}\n  {err}\n\n"
        "If this host is blocked, fetch the file by hand, decompress it, and\n"
        "drop it in the reference directory under the name this script\n"
        "expects (<genome>.fa / <genome>.gtf), then re-run -- it will skip the\n"
        "download and go straight to building the index.\n".format(
            url=url, err=err)
    )


def fetch_and_gunzip(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("[ref] have {} ({:.1f} MB), skipping".format(
            dest, os.path.getsize(dest) / 1e6))
        return dest

    tmp = dest + ".gz.part"
    print("[ref] downloading {}".format(url))
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as out:
            shutil.copyfileobj(r, out)
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        _blocked(url, e)

    print("[ref] decompressing -> {}".format(dest))
    with gzip.open(tmp, "rb") as fin, open(dest, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    os.remove(tmp)
    return dest


def build_index(ref_dir, fasta, prefix, threads=1):
    first = os.path.join(ref_dir, prefix + ".1.bt2")
    if os.path.exists(first):
        print("[ref] bowtie2 index present, skipping")
        return

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline._common import BOWTIE2_HINT, require
    exe = require("bowtie2-build-s", BOWTIE2_HINT, fallbacks=("bowtie2-build",))

    size = os.path.getsize(fasta) / 1e9
    print("[ref] building bowtie2 index for {:.1f} Gb -- this is the slow step"
          .format(size))
    if size > 1.0:
        print("[ref] a genome this size takes a few hours and several GB of "
              "RAM; --threads helps if your bowtie2-build supports it")
    cmd = [exe]
    if threads > 1:
        cmd += ["--threads", str(threads)]
    cmd += [fasta, os.path.join(ref_dir, prefix)]
    print("[ref] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", default="ref")
    ap.add_argument("--dataset", default=None,
                    help="which dataset's reference to fetch; see DATASETS in "
                         "pipeline/_common.py (default: the yeast one)")
    ap.add_argument("--threads", type=int, default=1,
                    help="passed to bowtie2-build; the dominant cost for a "
                         "mammalian genome")
    ap.add_argument("--skip-index", action="store_true",
                    help="download only; do not call bowtie2-build")
    args = ap.parse_args()

    genome, spec = reference_for(args.dataset)
    print("[ref] {} -- {}".format(genome, spec["note"]))

    os.makedirs(args.ref_dir, exist_ok=True)
    fasta = fetch_and_gunzip(spec["genome_url"],
                             os.path.join(args.ref_dir, genome + ".fa"))
    fetch_and_gunzip(spec["gtf_url"], os.path.join(args.ref_dir, genome + ".gtf"))

    if not args.skip_index:
        build_index(args.ref_dir, fasta, genome, threads=args.threads)

    print("[ref] done -> {}".format(args.ref_dir))


if __name__ == "__main__":
    main()
