import argparse
import datetime
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.recoverability import config as C

M = "analysis.recoverability."
P = "gse174329_eif3b"
Y = "archer_yeast"
ALL = ["gse174329_eif3b", "gse139131_unsel", "gse139131_eif3b", "archer_yeast"]
ALIGN = {"human-eif3b-replicated": ["SRR14510018", "SRR14510019"],
         "archer-yeast": ["SRR3458591", "SRR3458592", "SRR3458593"],
         "human-40s": ["SRR10315320", "SRR10315321"],
         "human-eif3b": ["SRR10315322"]}
ALL_FAMILIES = "SW,ST,AUG,S1,S3,S5,ST_paired,ST_dG,UORF"
DONE = os.path.join(C.RESULTS, ".done")
LOGS = os.path.join(C.RESULTS, "logs")


def plan(res):
    W = str(C.default_workers())
    cpus = res["cpus"]
    n_align = max(1, min(len(ALIGN), int(res["ram_gb"] // 6), cpus // 4 or 1))
    align_threads = str(max(1, cpus // n_align))
    py = [sys.executable, "-u"]
    pym = py + ["-m"]

    def m(mod, *a):
        return pym + [M + mod] + list(a)

    g = []
    g.append(("align", [("align_" + ds, py + ["pipeline/02_align_count.py", "--dataset", ds,
                                             "--stream", "--threads", align_threads])
                        for ds in ALIGN]))
    g.append(("inspect", [("inspect", m("00_inspect_reads", "--threads", str(min(cpus, 16))))]))
    g.append(("offsets", [("offsets", m("01_qc", "--stage", "offsets"))]))
    g.append(("annotation", [("annotation_human", m("panel", "--species", "human",
                                                    "--processes", str(max(1, cpus - 2)))),
                             ("annotation_yeast", m("panel", "--species", "yeast",
                                                    "--processes", "2"))]))
    g.append(("counts", [("counts_" + k, m("panel", "--dataset", k)) for k in ALL]))
    g.append(("qc_tables", [("qc_tables", m("01_qc", "--stage", "tables"))]))
    g.append(("resolution", [("resolution", m("02_replicate_resolution", "--datasets",
                                              "gse174329_eif3b,gse139131_unsel"))]))
    g.append(("effects", [("effects_" + k, m("03_nested_effects", "--dataset", k)) for k in ALL]))
    nulls = [("nulls_P_b10", P, ALL_FAMILIES, "10"),
             ("nulls_P_b1", P, "SW,ST,AUG,S3", "1"),
             ("nulls_P_b5", P, "SW,ST,AUG", "5"),
             ("nulls_P_b20", P, "ST,AUG", "20")]
    nulls += [("nulls_Y_b{}".format(b), Y, ALL_FAMILIES, b) for b in ("1", "5", "10")]
    for name, key, fam, b in nulls:
        g.append((name, [(name, m("04_shuffle_nulls", "--dataset", key, "--features", fam,
                                  "--bin", b, "--workers", W))]))
    g.append(("identifiability", [("identifiability", m("06b_identifiability"))]))
    g.append(("synthetic_P", [("synthetic_P_dm", m("06_synthetic_power", "--dataset", P,
                                                   "--noise", "dm", "--workers", W))]))
    g.append(("synthetic_Y", [("synthetic_Y_multinomial", m("06_synthetic_power", "--dataset", Y,
                                                           "--noise", "multinomial",
                                                           "--workers", W))]))
    g.append(("chain", [("chain_Y", m("07_chain_ablation", "--dataset", Y, "--bins", "10",
                                      "--seeds", "0,1,2")),
                        ("chain_P", m("07_chain_ablation", "--dataset", P, "--bins", "10",
                                      "--seeds", "0,1,2"))]))
    g.append(("depth", [("depth_P", m("05_depth_curve", "--dataset", P, "--bin", "10"))]))
    g.append(("depth_matched", [("depth_matched", m("05_depth_curve", "--matched",
                                                    "gse174329_eif3b,gse139131_unsel",
                                                    "--bin", "10", "--seeds", "0,1,2"))]))
    g.append(("summary", [("summary", m("summary", "--dataset", P, "--noise", "dm"))]))
    g.append(("figures", [("figures", m("make_figures"))]))
    g.append(("downstream", [("downstream", m("08_downstream_utility"))]))
    return g


def preflight(steps_to_run):
    problems = []
    need_align = any(s.startswith("align_") for s in steps_to_run)
    for tool in (["bowtie2-align-s", "cutadapt"] if need_align else []):
        if not shutil.which(tool):
            problems.append("missing tool on PATH: {} (activate the utrscan environment, "
                            "or run scripts/remote/bootstrap.sh)".format(tool))
    try:
        import RNA
    except ImportError:
        problems.append("ViennaRNA (python module RNA) not importable")
    for f in ("ref/sacCer3.fa", "ref/sacCer3.gtf", "ref/sacCer3.1.bt2",
              "ref/GRCh38.fa", "ref/GRCh38.gtf", "ref/GRCh38.1.bt2",
              "ref/rrna/human_rrna_nuclear.fa", "ref/rrna/yeast_rdna.fa"):
        if not os.path.exists(os.path.join(C.ROOT, f)):
            problems.append("missing reference {} (run scripts/remote/fetch_data.py)".format(f))
    if need_align:
        fq = os.path.join(C.ROOT, "fastq")
        have = os.listdir(fq) if os.path.isdir(fq) else []
        for runs in ALIGN.values():
            for r in runs:
                if not any(n.startswith(r) and n.endswith((".fastq.gz", ".fastq")) for n in have):
                    problems.append("missing FASTQ for {} in fastq/ "
                                    "(run scripts/remote/fetch_data.py)".format(r))
    if problems:
        sys.exit("\nPreflight failed:\n  " + "\n  ".join(problems) + "\n")


def say(msg, fh=None):
    line = "[{}] {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def run_group(steps, force, logfh, env):
    procs = []
    for name, argv in steps:
        marker = os.path.join(DONE, name)
        if os.path.exists(marker) and not force:
            say("skip    {} (done)".format(name), logfh)
            continue
        log = open(os.path.join(LOGS, name + ".log"), "w")
        say("start   {}   log: {}".format(name, os.path.relpath(log.name, C.ROOT)), logfh)
        p = subprocess.Popen(argv, cwd=C.ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        procs.append((name, p, log, time.time()))
    ok = True
    for name, p, log, t0 in procs:
        rc = p.wait()
        log.close()
        mins = (time.time() - t0) / 60
        if rc == 0:
            open(os.path.join(DONE, name), "w").write(datetime.datetime.now().isoformat())
            say("done    {} ({:.1f} min)".format(name, mins), logfh)
        else:
            ok = False
            say("FAILED  {} (exit {}, {:.1f} min) -- see {}".format(
                name, rc, mins, os.path.relpath(log.name, C.ROOT)), logfh)
            try:
                with open(log.name, errors="replace") as fh:
                    tail = fh.readlines()[-15:]
                print("".join("        | " + l for l in tail))
            except OSError:
                pass
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--from", dest="start", help="start at this step or group")
    ap.add_argument("--only", help="comma-separated steps or groups")
    ap.add_argument("--force", action="store_true", help="re-run steps already marked done")
    ap.add_argument("--skip-align", action="store_true",
                    help="alignment outputs already exist (work/*.npz); skip it")
    args = ap.parse_args()
    C.require_env()
    C.ensure_layout()
    os.makedirs(DONE, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    res = C.resources()
    groups = plan(res)
    names = [(gname, [s for s, _ in steps]) for gname, steps in groups]
    if args.skip_align:
        groups = [g for g in groups if g[0] != "align"]
    if args.start:
        idx = next((i for i, (gn, ss) in enumerate(names) if args.start == gn or args.start in ss), None)
        if idx is None:
            sys.exit("unknown step or group {!r}; see --list".format(args.start))
        keep = {gn for gn, _ in names[idx:]}
        groups = [g for g in groups if g[0] in keep]
    if args.only:
        want = set(args.only.split(","))
        groups = [(gn, [s for s in steps if gn in want or s[0] in want]) for gn, steps in groups]
        groups = [g for g in groups if g[1]]
    if args.list:
        print("hardware: {}  backend: {}  workers: {}".format(res, C.backend(), C.default_workers()))
        for gn, steps in groups:
            print("{:<16} {}".format(gn, "  ".join(
                s + (" [done]" if os.path.exists(os.path.join(DONE, s)) else "") for s, _ in steps)))
        return
    todo = [s for _, steps in groups for s, _ in steps]
    preflight(todo)
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env["PYTHONPATH"] = C.ROOT + os.pathsep + env.get("PYTHONPATH", "")
    with open(os.path.join(LOGS, "run_all.log"), "a") as logfh:
        say("run_all: {} groups, hardware {}, backend {}, workers {}".format(
            len(groups), res, C.backend(), C.default_workers()), logfh)
        t0 = time.time()
        for gname, steps in groups:
            say("== {}".format(gname), logfh)
            if not run_group(steps, args.force, logfh, env):
                say("stopped after a failure.  Fix it, then resume with:\n"
                    "    python -m analysis.recoverability.run_all --from {}".format(gname), logfh)
                sys.exit(1)
        say("ALL DONE in {:.1f} h.  Results: {}   Figures: {}".format(
            (time.time() - t0) / 3600, os.path.relpath(C.RESULTS, C.ROOT),
            os.path.relpath(os.path.join(C.RESULTS, "figures"), C.ROOT)), logfh)


if __name__ == "__main__":
    main()
