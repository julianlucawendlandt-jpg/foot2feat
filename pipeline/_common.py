import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REF_DIR = os.path.join(ROOT, "ref")
WORK_DIR = os.path.join(ROOT, "work")
CACHE_DIR = os.path.join(ROOT, "cache")

DATASETS = {
    "archer-yeast": {
        "genome": "sacCer3",
        "adapter": "TGGAATTCTCGGGTGCCAAGG",
        "poly_a": True,
        "min_length": 18,
        "fractions": {
            "ssu": "SRR3458591",
            "rs": "SRR3458592",
            "input": "SRR3458593",
        },
    },
    "human-40s": {
        "genome": "GRCh38",
        "adapter": "CTGTAGGCACCATCAAT",
        "poly_a": False,
        "min_length": 25,
        "fractions": {
            "ssu": "SRR10315320",
            "ssu_rep2": "SRR10315321",
        },
    },
    "human-eif3b": {
        "genome": "GRCh38",
        "adapter": "CTGTAGGCACCATCAAT",
        "poly_a": False,
        "min_length": 25,
        "fractions": {"ssu": "SRR10315322"},
    },
    "human-eif3b-replicated": {
        "genome": "GRCh38",
        "adapter": "AGATCGGAAGAGCACACGTCTGAA",
        "poly_a": False,
        "min_length": 25,
        "fractions": {
            "ssu_rep1": "SRR14510018",
            "ssu_rep2": "SRR14510019",
        },
    },
}

DEFAULT_DATASET = "archer-yeast"

BOWTIE2_INDEX = os.path.join(REF_DIR, "sacCer3")
GENOME_FA = os.path.join(REF_DIR, "sacCer3.fa")
GTF = os.path.join(REF_DIR, "sacCer3.gtf")
FRACTIONS = DATASETS[DEFAULT_DATASET]["fractions"]


def dataset(name=None):
    name = name or DEFAULT_DATASET
    if name not in DATASETS:
        sys.exit("\nUnknown dataset {!r}. Known: {}\n".format(
            name, ", ".join(sorted(DATASETS))))
    d = dict(DATASETS[name])
    g = d["genome"]
    d.update(
        name=name,
        index=os.path.join(REF_DIR, g),
        genome_fa=os.path.join(REF_DIR, g + ".fa"),
        gtf=os.path.join(REF_DIR, g + ".gtf"),
        prefix="" if name == DEFAULT_DATASET else name + ".",
    )
    return d


def work_path(ds, frac, suffix):
    return os.path.join(WORK_DIR, "{}{}.{}".format(ds["prefix"], frac, suffix))

FASTQ_SEARCH_DIRS = [
    os.path.join(os.path.dirname(ROOT), "fastq"),
    os.path.join(ROOT, "fastq"),
    os.path.dirname(ROOT),
]


ADAPTER = "TGGAATTCTCGGGTGCCAAGG"
MIN_LENGTH = 18


def find_cutadapt():
    exe = _which("cutadapt")
    if exe:
        return [exe]
    try:
        import cutadapt
    except ImportError:
        sys.exit("\ncutadapt not found on PATH, and not importable either.\n"
                 "  pip install cutadapt\n")
    return [sys.executable, "-m", "cutadapt"]


def cutadapt_cmd(cutadapt, src, dst, threads=0, ds=None):
    ds = ds or dataset()
    prefix = list(cutadapt) if isinstance(cutadapt, (list, tuple)) else [cutadapt]
    cmd = prefix + ["-a", ds["adapter"]]
    if ds.get("poly_a"):
        cmd += ["-a", "A{20}", "--times", "2"]
    cmd += [
        "-m", str(ds["min_length"]),
        "-q", "20",
        "--max-n", "0",
        "-j", str(threads),
        "-o", dst,
        src,
    ]
    return cmd


def find_fastqs(extra_dir=None, fractions=None):
    dirs = ([extra_dir] if extra_dir else []) + FASTQ_SEARCH_DIRS
    found = {}
    for frac, acc in (fractions or FRACTIONS).items():
        for d in dirs:
            if not d or not os.path.isdir(d):
                continue
            for name in (acc + ".fastq", acc + ".fastq.gz",
                         acc + "_1.fastq", acc + "_1.fastq.gz"):
                p = os.path.join(d, name)
                if os.path.exists(p):
                    found[frac] = p
                    break
            if frac in found:
                break
        if frac not in found:
            sys.exit(
                "\nCould not find the {} FASTQ ({}).\n"
                "Looked in:\n  {}\n"
                "Pass --fastq-dir to point at it.\n".format(
                    frac, acc, "\n  ".join(d for d in dirs if d))
            )
    return found


BIN_DIR = os.path.join(ROOT, "bin")

BOWTIE2_HINT = """  Windows: download bowtie2-<ver>-mingw-x86_64.zip from
           https://github.com/BenLangmead/bowtie2/releases and unzip it under
           utr-scan/bin/  (bioconda has no Windows packages)
  Linux/macOS/WSL: conda install -c bioconda bowtie2"""


def _which(name):
    if os.path.isdir(BIN_DIR):
        for d, _, _ in os.walk(BIN_DIR):
            hit = shutil.which(name, path=d)
            if hit:
                return hit
    return shutil.which(name)


def require(binary, install_hint, fallbacks=()):
    for name in (binary,) + tuple(fallbacks):
        exe = _which(name)
        if exe:
            return exe
    sys.exit(
        "\n{} not found on PATH or under {}.\n{}\n"
        "See README.md for the full list of required binaries.\n".format(
            binary, BIN_DIR, install_hint)
    )


def up_to_date(output, *inputs):
    if not os.path.exists(output) or os.path.getsize(output) == 0:
        return False
    out_mtime = os.path.getmtime(output)
    return all(
        os.path.getmtime(i) <= out_mtime
        for i in inputs if i and os.path.exists(i)
    )


def run(cmd, **kwargs):
    print("[run] " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kwargs)


def ensure_dirs():
    for d in (WORK_DIR, CACHE_DIR, REF_DIR):
        os.makedirs(d, exist_ok=True)


def load_counts_npz(path):
    import numpy as np
    from data.coverage import CountTracks
    with np.load(path, allow_pickle=True) as z:
        return CountTracks.from_npz(z)
