import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.recoverability import config as C

M = "analysis.recoverability."
UNS, Y = "gse139131_unsel", "archer_yeast"
OUT = os.path.join(C.RESULTS, "missing_eval")
DONE, LOGS = os.path.join(OUT, ".done"), os.path.join(OUT, "logs")
MANIFEST = os.path.join(OUT, "manifest.json")

FULL = {"n_perm": 100, "null_seed": 0, "n_sim": 50, "n_null_sim": 100, "n_boot": 2000,
        "seeds": "0,1,2", "power_tag": "n50", "same_study_n_perm": 100, "power_grid": []}
SMOKE = {"n_perm": 3, "null_seed": 0, "n_sim": 2, "n_null_sim": 3, "n_boot": 50,
         "seeds": "0", "power_tag": "smoke", "same_study_n_perm": 2,
         "power_grid": ["--depths", "20,200", "--sizes", "1", "--folds", "0"]}

NULL_PLAN = [(UNS, "1", "SW,ST,AUG,S3"), (UNS, "5", "SW,ST,AUG"),
             (UNS, "10", "SW,ST,AUG,S1,S3,S5,ST_paired,ST_dG,UORF"), (UNS, "20", "ST,AUG"),
             (Y, "20", "ST,AUG")]


def plan(p, workers):
    py = [sys.executable, "-u"]

    def m(mod, *a):
        return py + ["-m", M + mod] + [str(x) for x in a]

    steps = [("audit_inputs", [m("missing_eval", "audit")], None)]
    for key, b, fams in NULL_PLAN[:4]:
        steps.append(("nulls_unselected_b" + b,
                      [m("04_shuffle_nulls", "--dataset", key, "--features", fams, "--bin", b,
                         "--n-perm", p["n_perm"], "--seed", p["null_seed"], "--workers", workers)],
                      (key, b, fams)))
    steps += [
        ("power_unselected_dm", [m("06_synthetic_power", "--dataset", UNS, "--noise", "dm", "--bin", 10,
                                   "--n-sim", p["n_sim"], "--n-null-sim", p["n_null_sim"],
                                   "--tag", p["power_tag"], "--workers", workers,
                                   *p["power_grid"])], None),
        ("summary_unselected", [m("summary", "--dataset", UNS, "--noise", "dm",
                                  "--power-tag", p["power_tag"],
                                  "--out", os.path.join(OUT, "unselected_summary_raw.tsv")),
                                m("missing_eval", "unselected_table", "--power-tag", p["power_tag"])],
         None),
        ("assay_comparison", [m("09_assay_comparison", "--mode", "primary", "--bin", 10,
                                "--seeds", p["seeds"], "--n-boot", p["n_boot"])], None),
        ("same_study_selection", [m("09_assay_comparison", "--mode", "same_study", "--bin", 10,
                                    "--seeds", p["seeds"], "--n-perm", p["same_study_n_perm"],
                                    "--n-boot", p["n_boot"])], None),
    ]
    key, b, fams = NULL_PLAN[4]
    steps.append(("nulls_yeast_b20",
                  [m("04_shuffle_nulls", "--dataset", key, "--features", fams, "--bin", b,
                     "--n-perm", p["n_perm"], "--seed", p["null_seed"], "--workers", workers)],
                  (key, b, fams)))
    steps += [
        ("power_yeast_multinomial", [m("06_synthetic_power", "--dataset", Y, "--noise", "multinomial",
                                       "--bin", 10, "--n-sim", p["n_sim"], "--n-null-sim",
                                       p["n_null_sim"], "--tag", p["power_tag"],
                                       "--workers", workers, *p["power_grid"])], None),
        ("summary_yeast_single_library", [m("10_yeast_single_library_summary",
                                            "--power-tag", p["power_tag"])], None),
        ("freeze_downstream_model_choice", [m("08_downstream_utility", "--stage", "choose")], None),
        ("prepare_reporter_tables", [py + [os.path.join(C.ROOT, "eval", "prepare_akirtava_downstream.py")]],
         None),
        ("downstream_reporter_utility", [m("08_downstream_utility", "--stage", "run",
                                           "--n-boot", p["n_boot"])], None),
        ("missing_eval_figures", [m("make_missing_figures")], None),
        ("missing_eval_report", [m("missing_eval", "report")], None),
    ]
    return steps


def nulls_present(key, b, fams):
    path = C.results_path("nulls", "shuffle_nulls.tsv")
    if not os.path.exists(path):
        return False
    have = {(r["dataset"], r["bin_width"], r["model"]) for r in C.read_tsv(path)}
    return all((key, b, f_) in have for f_ in fams.split(","))


def versions():
    v = {"python": platform.python_version(), "platform": platform.platform(),
         "git_commit": C.git_commit()}
    for mod in ("numpy", "scipy", "sklearn", "torch", "RNA", "matplotlib"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception as e:
            v[mod] = "unavailable ({})".format(type(e).__name__)
    return v


def record(entry):
    m = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {"steps": []}
    m["steps"].append(entry)
    with open(MANIFEST, "w") as fh:
        json.dump(m, fh, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--from", dest="start")
    ap.add_argument("--only")
    ap.add_argument("--force", action="store_true", help="re-run steps already marked done")
    ap.add_argument("--smoke", action="store_true", help="tiny counts, for testing the wiring")
    args = ap.parse_args()
    C.require_env()
    if not os.path.exists(os.path.join(C.RESULTS, "effects", "nested_effects.tsv")):
        sys.exit("\nNo frozen results at {} -- this runner only adds to a completed run.\n".format(C.RESULTS))
    os.makedirs(DONE, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    p = SMOKE if args.smoke else FULL
    workers = C.default_workers()
    steps = plan(p, workers)
    names = [s[0] for s in steps]
    if args.start:
        if args.start not in names:
            sys.exit("unknown step {!r}; see --list".format(args.start))
        steps = steps[names.index(args.start):]
    if args.only:
        want = set(args.only.split(","))
        steps = [s for s in steps if s[0] in want]
    if args.list:
        print("parameters: {}  workers: {}  backend: {}".format(p, workers, C.backend()))
        for name, cmds, _ in steps:
            print("{:<32} {}".format(name, "[done]" if os.path.exists(os.path.join(DONE, name)) else ""))
        return
    m = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {"steps": []}
    m.update({"parameters": p, "smoke": args.smoke, "versions": versions(),
              "hardware": C.resources(), "workers": workers, "backend": C.backend(),
              "results_dir": C.RESULTS, "started": datetime.datetime.now().isoformat(timespec="seconds")})
    with open(MANIFEST, "w") as fh:
        json.dump(m, fh, indent=2, default=str)
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONPATH"] = C.ROOT + os.pathsep + env.get("PYTHONPATH", "")
    for name, cmds, null_spec in steps:
        marker = os.path.join(DONE, name)
        if os.path.exists(marker) and not args.force:
            print("skip    {} (done)".format(name), flush=True)
            continue
        if null_spec and nulls_present(*null_spec) and not args.force:
            print("skip    {} (null rows already present; not re-running a completed job)".format(name),
                  flush=True)
            open(marker, "w").write("pre-existing")
            continue
        t0 = time.time()
        log = os.path.join(LOGS, name + ".log")
        print("start   {}   log: {}".format(name, os.path.relpath(log, C.ROOT)), flush=True)
        rc = 0
        with open(log, "w") as fh:
            for cmd in cmds:
                fh.write("$ " + " ".join(cmd) + "\n")
                fh.flush()
                rc = subprocess.run(cmd, cwd=C.ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
                if rc != 0:
                    break
        record({"step": name, "commands": [" ".join(c) for c in cmds], "exit": rc,
                "minutes": round((time.time() - t0) / 60, 2),
                "finished": datetime.datetime.now().isoformat(timespec="seconds")})
        if rc != 0:
            print("FAILED  {} (exit {}) -- see {}".format(name, rc, os.path.relpath(log, C.ROOT)))
            with open(log, errors="replace") as fh:
                print("".join("        | " + l_ for l_ in fh.readlines()[-15:]))
            print("Resume with: python -m analysis.recoverability.run_missing_evaluations --from {}".format(name))
            sys.exit(1)
        open(marker, "w").write(datetime.datetime.now().isoformat())
        print("done    {} ({:.1f} min)".format(name, (time.time() - t0) / 60), flush=True)
    print("ALL ADDED EVALUATIONS DONE.  Report: {}".format(
        os.path.relpath(os.path.join(OUT, "report.md"), C.ROOT)))


if __name__ == "__main__":
    main()
