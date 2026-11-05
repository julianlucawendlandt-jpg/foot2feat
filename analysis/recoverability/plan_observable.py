import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DE = importlib.import_module("analysis.recoverability.12_experiment_design")

ALIAS = {"80S_known": "80S_known_scale", "80S_unknown": "80S_unknown_scale", "cds30": "cds30"}


def observable(mech, cand):
    if cand == "baseline":
        return "occupancy"
    if cand == "output":
        return "main_AUG_output_only"
    if cand in ("res3", "res5", "res10"):
        return "occupancy@{}nt".format(cand[3:])
    parts = [ALIAS.get(p, p) for p in cand.split("+")]
    return "occupancy+" + "+".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--mechanism", required=True, choices=sorted(DE.MECHANISMS))
    ap.add_argument("--candidates", default="")
    ap.add_argument("--reads", type=float, default=500, help="reads per transcript")
    a = ap.parse_args()
    cands = ["baseline"] + [c for c in a.candidates.split(",") if c]
    obs = [observable(a.mechanism, c) for c in cands]
    rows = DE.score_observables(a.mechanism, obs, N=a.reads)
    base = rows[0]
    print("\n{}  (target: {}; {:.0f} reads per transcript)".format(a.mechanism, base["target"], a.reads))
    print("  {:<38} {:>12} {:>12} {:>15} {:>10}".format("observation", "identifiable", "target SE",
                                                       "min eig / read", "log det"))
    for c, r in zip(cands, rows):
        print("  {:<38} {:>12} {:>12} {:>15.2e} {:>10.1f}".format(
            r["observable"], str(r["identifiable"]), "{:.3g}".format(r["target_se"]) if r["identifiable"] else "inf",
            r["min_eig_per_read"], r["logdet"]))
    ident = [r for r in rows if r["identifiable"]]
    if not ident:
        print("\n  none of these makes {} identifiable".format(base["target"]))
    else:
        best = min(ident, key=lambda r: r["target_se"])
        gain = ("" if not base["identifiable"] else
                " ({:.0%} lower target SE than the baseline)".format(1 - best["target_se"] / base["target_se"]))
        print("\n  recommended: {}{}".format(best["observable"], gain))
    print("  status: validated -- reproduces 06b's frozen Fisher-information results "
          "(model-based identifiability, not an experimental measurement)")


if __name__ == "__main__":
    main()
