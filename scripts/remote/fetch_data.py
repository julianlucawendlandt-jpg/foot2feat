import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = os.path.join(ROOT, "ref")
DL = os.path.join(ROOT, "downloads")

RUNS = {
    "SRR14510018": "GSE174329 eIF3B Sel-TCP-seq rep1 (GSM5291862)",
    "SRR14510019": "GSE174329 eIF3B Sel-TCP-seq rep2 (GSM5291863)",
    "SRR10315320": "GSE139131 unselected 40S rep1 (GSM4131410)",
    "SRR10315321": "GSE139131 unselected 40S rep2 (GSM4131411)",
    "SRR10315322": "GSE139131 eIF3B Sel-TCP-seq (GSM4131412)",
    "SRR3458591": "Archer yeast SSU (SRX1734358)",
    "SRR3458592": "Archer yeast 80S (SRX1734359)",
    "SRR3458593": "Archer yeast Input (SRX1734360)",
}
GENCODE = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49"
HUMAN = {
    "index_zip": "https://genome-idx.s3.amazonaws.com/bt/GRCh38_noalt_as.zip",
    "fasta_gz": GENCODE + "/GRCh38.primary_assembly.genome.fa.gz",
    "gtf_gz": GENCODE + "/gencode.v49.annotation.gtf.gz",
}
NCBI_RRNA = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore"
             "&id=NR_046235.3,NR_023363.1&rettype=fasta&retmode=text")
GB = 2 ** 30


def log(msg):
    print("[fetch] " + msg, flush=True)


def md5sum(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def http_get(url, timeout=60):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read().decode()
        except Exception as e:
            if attempt == 4:
                raise
            log("retry {} ({})".format(url, e))
            time.sleep(5 * (attempt + 1))


def download(url, dest, expected_bytes=None):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    if shutil.which("curl"):
        for attempt in range(8):
            r = subprocess.run(["curl", "-fL", "--retry", "5", "--retry-delay", "10",
                                "-C", "-", "-o", part, url])
            if r.returncode in (0, 33):
                break
            log("curl exit {} for {}, retrying".format(r.returncode, url))
            time.sleep(15)
        else:
            sys.exit("\nDownload failed repeatedly: {}\n".format(url))
    else:
        start = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers={"Range": "bytes={}-".format(start)} if start else {})
        with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if start else "wb") as out:
            shutil.copyfileobj(r, out, 1 << 22)
    if expected_bytes and os.path.getsize(part) != int(expected_bytes):
        sys.exit("\n{} is {} bytes, expected {}.  Delete it and re-run.\n".format(
            part, os.path.getsize(part), expected_bytes))
    os.replace(part, dest)


def ena_report(run):
    url = ("https://www.ebi.ac.uk/ena/portal/api/filereport?accession={}&result=read_run"
           "&fields=run_accession,fastq_ftp,fastq_md5,fastq_bytes,read_count".format(run))
    lines = http_get(url).strip().splitlines()
    hdr, row = lines[0].split("\t"), lines[1].split("\t")
    d = dict(zip(hdr, row))
    files = [f for f in d["fastq_ftp"].split(";") if f]
    md5s, sizes = d["fastq_md5"].split(";"), d["fastq_bytes"].split(";")
    i = 0 if len(files) == 1 else next(k for k, f in enumerate(files) if f.endswith("_1.fastq.gz"))
    return {"url": "https://" + files[i], "md5": md5s[i], "bytes": int(sizes[i]),
            "read_count": int(d["read_count"]), "name": os.path.basename(files[i])}


def check_disk(missing_bytes, min_margin_gb):
    free = shutil.disk_usage(ROOT).free / GB
    need = missing_bytes / GB + min_margin_gb
    log("free disk {:.0f} GB, need about {:.0f} GB more".format(free, need))
    if free < need:
        sys.exit("\nNot enough disk: {:.0f} GB free, about {:.0f} GB needed (downloads still "
                 "missing + references + alignment/analysis intermediates).\nFree space, or "
                 "lower the margin with --margin-gb.\n".format(free, need))


def fetch_fastqs(fastq_dir, manifest, margin_gb):
    os.makedirs(fastq_dir, exist_ok=True)
    total = 0
    reports = {}
    for run in RUNS:
        reports[run] = ena_report(run)
        total += reports[run]["bytes"]
    log("{} FASTQs, {:.1f} GB in total".format(len(RUNS), total / GB))
    missing = sum(r["bytes"] for r in reports.values()
                  if not os.path.exists(os.path.join(fastq_dir, r["name"])))
    check_disk(missing, margin_gb)
    for run, rep in reports.items():
        dest = os.path.join(fastq_dir, rep["name"])
        if os.path.exists(dest) and os.path.getsize(dest) == rep["bytes"]:
            if md5sum(dest) == rep["md5"]:
                log("ok      {} ({})".format(rep["name"], RUNS[run]))
                manifest["fastq"][run] = dict(rep, path=dest, verified=True)
                continue
            log("md5 mismatch on existing {}, re-downloading".format(dest))
            os.remove(dest)
        for attempt in range(2):
            log("get     {} ({:.2f} GB) {}".format(rep["name"], rep["bytes"] / GB, RUNS[run]))
            download(rep["url"], dest, rep["bytes"])
            if md5sum(dest) == rep["md5"]:
                break
            log("md5 mismatch after download, retrying from scratch")
            os.remove(dest)
        else:
            sys.exit("\n{} failed its ENA MD5 twice.\n".format(rep["name"]))
        manifest["fastq"][run] = dict(rep, path=dest, verified=True)


def fetch_yeast():
    if all(os.path.exists(os.path.join(REF, "sacCer3" + e)) for e in (".fa", ".gtf", ".1.bt2")):
        log("ok      yeast reference")
        return
    log("yeast reference (Ensembl R64-1-1, builds bowtie2 index)")
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "download_reference.py")],
                   check=True, cwd=ROOT)


def fetch_human(manifest):
    os.makedirs(REF, exist_ok=True)
    idx = [os.path.join(REF, "GRCh38" + s) for s in
           (".1.bt2", ".2.bt2", ".3.bt2", ".4.bt2", ".rev.1.bt2", ".rev.2.bt2")]
    if not all(os.path.exists(p) for p in idx):
        z = os.path.join(DL, "GRCh38_noalt_as.zip")
        if not os.path.exists(z):
            log("get     human bowtie2 index (~3.7 GB)")
            download(HUMAN["index_zip"], z)
        log("unpack  human bowtie2 index")
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                base = os.path.basename(info.filename)
                if not base.endswith(".bt2"):
                    continue
                dest = os.path.join(REF, base.replace("GRCh38_noalt_as", "GRCh38"))
                with zf.open(info) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 22)
        os.remove(z)
    else:
        log("ok      human bowtie2 index")
    fa = os.path.join(REF, "GRCh38.fa")
    if not os.path.exists(fa):
        g = os.path.join(DL, "GRCh38.primary_assembly.genome.fa.gz")
        if not os.path.exists(g):
            log("get     GENCODE v49 genome FASTA (~0.85 GB)")
            download(HUMAN["fasta_gz"], g)
        log("unpack  genome FASTA")
        with gzip.open(g, "rb") as src, open(fa + ".part", "wb") as out:
            shutil.copyfileobj(src, out, 1 << 22)
        os.replace(fa + ".part", fa)
        os.remove(g)
    else:
        log("ok      human genome FASTA")
    gtf = os.path.join(REF, "GRCh38.gtf")
    if not os.path.exists(gtf):
        g = os.path.join(DL, "gencode.v49.annotation.gtf.gz")
        if not os.path.exists(g):
            log("get     GENCODE v49 GTF (~0.1 GB)")
            download(HUMAN["gtf_gz"], g)
        log("filter  GTF to MANE_Select records")
        n = 0
        with gzip.open(g, "rt") as src, open(gtf + ".part", "w") as out:
            for line in src:
                if "MANE_Select" in line:
                    out.write(line)
                    n += 1
        os.replace(gtf + ".part", gtf)
        log("        {} MANE_Select records".format(n))
    else:
        log("ok      human GTF (MANE_Select)")
    manifest["human"] = dict(HUMAN, gtf_filter="lines containing MANE_Select")


def fetch_rrna():
    d = os.path.join(REF, "rrna")
    os.makedirs(d, exist_ok=True)
    h = os.path.join(d, "human_rrna_nuclear.fa")
    if not os.path.exists(h) or os.path.getsize(h) < 1000:
        log("get     human rRNA (NR_046235.3, NR_023363.1)")
        with open(h, "w") as fh:
            fh.write(http_get(NCBI_RRNA))
    y = os.path.join(d, "yeast_rdna.fa")
    if not os.path.exists(y):
        seq, on = [], False
        with open(os.path.join(REF, "sacCer3.fa")) as fh:
            for line in fh:
                if line.startswith(">"):
                    on = line[1:].split()[0] == "XII"
                    continue
                if on:
                    seq.append(line.strip())
        with open(y, "w") as fh:
            fh.write(">XII_450000_470000\n" + "".join(seq)[450000:470000] + "\n")
    log("ok      rRNA references")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastq-dir", default=os.path.join(ROOT, "fastq"))
    ap.add_argument("--skip-fastq", action="store_true")
    ap.add_argument("--margin-gb", type=float, default=None,
                    help="working space required on top of missing downloads "
                         "(default: 20 GB, +12 GB if the human reference is absent)")
    args = ap.parse_args()
    margin = args.margin_gb
    if margin is None:
        margin = 20.0 + (0.0 if os.path.exists(os.path.join(REF, "GRCh38.fa")) else 12.0)
    manifest = {"fastq": {}}
    if not args.skip_fastq:
        fetch_fastqs(args.fastq_dir, manifest, margin)
    else:
        check_disk(0, margin)
    fetch_yeast()
    fetch_human(manifest)
    fetch_rrna()
    with open(os.path.join(REF, "download_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    log("all inputs present and verified")


if __name__ == "__main__":
    main()
