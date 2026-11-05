import argparse
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DE = importlib.import_module("analysis.recoverability.12_experiment_design")
C, EFF, E = DE.C, DE.EFF, DE.E


def resolve(name):
    hits = [k for k in C.DATASETS if k == name or k.startswith(name)]
    if len(hits) != 1:
        sys.exit("--pilot {!r} matches {}; choose one of {}".format(name, hits or "nothing", sorted(C.DATASETS)))
    return hits[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--effect", default="SW", choices=DE.FAMS)
    ap.add_argument("--effect-size", type=float, default=None)
    ap.add_argument("--bin", type=int, default=10)
    ap.add_argument("--budget-reads", type=float, default=1e8, help="total sequenced reads, all libraries")
    ap.add_argument("--replicates", default="2,3,4,5,6")
    ap.add_argument("--target", type=float, default=DE.POWER_OK)
    ap.add_argument("--kappa", type=float, default=None, help="override the pilot's overdispersion")
    ap.add_argument("--cost-library", type=float, default=None,
                    help="EUR per library (ILLUSTRATIVE overlay; default {:.0f} if only the other is set)".format(DE.COST_LIBRARY))
    ap.add_argument("--cost-per-million", type=float, default=None,
                    help="EUR per million reads (ILLUSTRATIVE overlay; default {:.0f})".format(DE.COST_PER_MILLION))
    a = ap.parse_args(argv)
    P = DE.Pilot(resolve(a.pilot), b=a.bin, kappa=a.kappa)
    info = E.Info(P.key, a.effect, a.bin)
    fam, Rs = a.effect, [int(x) for x in a.replicates.split(",")]

    if a.effect_size is None:
        o = DE.observed_size(P, info, fam)
        size, cv2 = o["size"], o["cv2"]
        check = "PASS" if o["self_check"] else "FAIL (pilot too shallow: predictions unreliable)"
        eff = "observed in the pilot: held-out dLL {:+.4f}/read (planted-size equivalent {:.3f})".format(
            o["observed_dll"], size * (SP_ref(fam)))
    else:
        size, cv2 = DE.size_units(fam, a.effect_size), P.cv2[fam]
        check = "not applicable (hypothesised effect)"
        eff = "hypothesised: {} of size {} (06 units)".format(fam, a.effect_size)
    pilot_design = E.loo_design(info, info.med, 2, P.kappa, size=size, cv2=cv2)

    print("\nPILOT")
    print("  dataset = {}   bin width = {} nt  [{}]".format(P.key, a.bin, DE.bin_status(a.bin)))
    print("  kappa = {:.0f}  [{}]".format(P.kappa, P.kappa_source))
    print("  eligible transcripts (pilot depth, 2 libraries) = {:.0f}".format(pilot_design["eligible_transcripts"]))
    print("  leader reads per sequenced read = {:.3%}; median {:.0f} reads/leader".format(P.yield_, info.med))
    print("  effect: {}".format(eff))
    print("  pilot self-check = {}".format(check))

    print("\nREQUIRED DEPTH WITH 2 LIBRARIES (target power {:.2f})  [validated design]".format(a.target))
    for rule, lab in (("power_pooled", "pooled LOO (primary)"), ("power_every", "both directions pass")):
        d = DE.depth_for_power(P, info, size, cv2, rule, R=2, target=a.target)
        if np.isfinite(d):
            print("  {:<24} ~{:.0f} median leader reads per library (~{:.2g} sequenced reads per library)".format(
                lab, d, P.sequenced_from_depth(d)))
        else:
            print("  {:<24} not reached at any depth with 2 libraries".format(lab))

    print("\nDESIGNS AT {:.3g} TOTAL SEQUENCED READS".format(a.budget_reads))
    hdr = ("R", "reads/lib", "leader/lib", "eligible", "T_eff", "DM infl", "pooled", "every", "(indep)", "LOO corr", "status")
    print("  {:>3} {:>10} {:>11} {:>9} {:>7} {:>8} {:>7} {:>7} {:>8} {:>9}  {}".format(*hdr))
    for R in Rs:
        r = DE.design_row(P, info, size, R, a.budget_reads, cv2)
        if r.get("eligible_transcripts", 0) <= 0:
            print("  {:>3} {:>10.3g}   no leader reaches {} reads".format(R, r["sequenced_reads_per_library"], EFF.MIN_READS))
            continue
        print("  {:>3} {:>10.3g} {:>11.3g} {:>9.0f} {:>7.0f} {:>8.2f} {:>7.2f} {:>7.2f} {:>8.2f} {:>9.3f}  {}".format(
            R, r["sequenced_reads_per_library"], r["leader_reads_per_library"], r["eligible_transcripts"],
            r["T_eff"], r["dm_inflation"], r["power_pooled"], r["power_every"], r["power_every_independent"],
            r["loo_corr_mean"], r["validation_status"]))
    print("  (indep) = every-direction power if the LOO directions were independent (P_dir^R), for contrast only")

    print("\nTOTAL SEQUENCING NEEDED FOR {:.0%} POOLED-LOO POWER (cost-free)".format(a.target))
    need = {}
    for R in Rs:
        need[R] = DE.reads_for_power(P, info, size, R, cv2, "power_pooled", a.target)
        print("  R = {:<3} {:>12}  [{}]".format(R, "{:.3g}".format(need[R]) if np.isfinite(need[R]) else "not reached",
                                             DE.status(R)))
    ok = {R: B for R, B in need.items() if np.isfinite(B)}
    if ok:
        Rmin = min(ok, key=ok.get)
        print("  fewest total reads: R = {} ({:.3g} reads){}".format(
            Rmin, ok[Rmin], "" if Rmin == 2 else "  [extrapolated]"))
    else:
        print("  no replicate count in range reaches the target")

    if a.cost_library is not None or a.cost_per_million is not None:
        cl = DE.COST_LIBRARY if a.cost_library is None else a.cost_library
        cm = DE.COST_PER_MILLION if a.cost_per_million is None else a.cost_per_million
        rows, best = DE.cost_overlay(need, cl, cm)
        print("\nILLUSTRATIVE COST OVERLAY  (prices are assumptions, not measurements)")
        print("  library cost = EUR {:.0f}; sequencing cost = EUR {:.2f} per million reads".format(cl, cm))
        for r in rows:
            print("  R = {:<3} total cost {}".format(r["R"], "EUR {:,.0f}".format(r["cost_eur"]) if np.isfinite(r["cost_eur"]) else "-"))
        if best:
            print("  Under illustrative costs of EUR {:.0f}/library and EUR {:.2f} per million reads, the minimum-cost "
                  "design reaching {:.0%} pooled power is R = {} (EUR {:,.0f}).".format(cl, cm, a.target, best["R"], best["cost_eur"]))

    print("\nSTATUS: R = 2 validated (06 simulations, retrospective pilots); R > 2 extrapolated (analytic, no R > 2 "
          "data); {}; costs illustrative.".format(DE.bin_status(a.bin)))


def SP_ref(fam):
    return DE.SP.SIZE_K[1] if fam == "AUG" else DE.SP.SIZE_S[1]


if __name__ == "__main__":
    main()
