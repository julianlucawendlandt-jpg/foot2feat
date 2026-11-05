import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.recoverability import config as C
from analysis.recoverability.make_figures import (BLUE, DATASET_COLOR, DATASET_LABEL, GREY,
                                                  INK, INK2, ORANGE, AQUA, f, save, table)

SEL, UNS = "gse174329_eif3b", "gse139131_unsel"


def d1():
    res = table("resolution", "replicate_resolution.tsv")
    eff = table("effects", "nested_effects.tsv")
    nul = table("nulls", "shuffle_nulls.tsv") or []
    cmp_ = table("missing_eval", "assay_comparison.tsv")
    if not (res and eff and cmp_):
        print("[mfig] d1 skipped: inputs missing")
        return
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.subplots_adjust(wspace=0.32, bottom=0.3)
    for k in (SEL, UNS):
        pts = sorted((int(r["bin_width"]), f(r["value"]), f(r["ci_lo"]), f(r["ci_hi"])) for r in res
                     if r["dataset"] == k and r["metric"] == "pooled_posadj_r" and r["depth_stratum"] == "all")
        if pts:
            b, v, lo, hi = map(np.array, zip(*pts))
            ax[0].plot(b, v, marker="o", ms=4, color=DATASET_COLOR[k], label=DATASET_LABEL[k])
            ax[0].fill_between(b, lo, hi, color=DATASET_COLOR[k], alpha=0.15, lw=0)
    ax[0].set_xscale("log")
    ax[0].set_xticks(C.RESOLUTION_BINS)
    ax[0].set_xticklabels(C.RESOLUTION_BINS)
    ax[0].set_xlabel("bin width (nt)")
    ax[0].set_ylabel("position-adjusted replicate r")
    ax[0].set_title("a  replicate agreement", loc="left", color=INK)
    ax[0].legend(fontsize=7, loc="lower right")
    fams = ["SW", "ST", "AUG", "UORF"]
    for i, fam in enumerate(fams):
        for j, (k, off) in enumerate(((SEL, -0.16), (UNS, 0.16))):
            rs = [r for r in eff if r.get("panel", r["dataset"]) == k and r["model"] == fam
                  and r["bin_width"] == "10" and r["metric"] == "heldout_dll_read_weighted"
                  and r["replicate"] in ("rep1->rep2", "rep2->rep1")]
            for q, r in enumerate(rs):
                y = i + off + (q - 0.5) * 0.1
                ax[1].plot([f(r["ci_lo"]), f(r["ci_hi"])], [y, y], color=DATASET_COLOR[k], lw=1.5)
                ax[1].plot(f(r["value"]), y, "o", ms=4, color=DATASET_COLOR[k],
                           label=DATASET_LABEL[k] if (i == 0 and q == 0) else None)
            nm = [f(r["value"]) for r in nul if r["dataset"] == k and r["model"] == fam
                  and r["bin_width"] == "10" and r["metric"] == "null_mean" and r.get("cross_replicate") == "True"]
            if nm:
                ax[1].plot(np.mean(nm), i + off, "x", ms=7, mew=2, color=INK2,
                           label="matched-null mean" if (i == 0 and j == 0) else None)
    ax[1].axvline(0, color=INK2, lw=0.8)
    ax[1].set_yticks(range(len(fams)))
    ax[1].set_yticklabels(fams)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("cross-replicate held-out ΔLL / read (10 nt)")
    ax[1].set_title("b  feature effects, all transcripts", loc="left", color=INK)
    ax[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    core = ["SW", "ST", "AUG"]
    for i, fam in enumerate(core):
        for mode, off, c_, lab in (("observed", -0.15, GREY, "observed depth"),
                                   ("matched", 0.15, BLUE, "matched leader depth")):
            rs = [r for r in cmp_ if r["family"] == fam and r["depth_mode"] == mode]
            for q, r in enumerate(rs):
                y = i + off + (q - (len(rs) - 1) / 2) * 0.08
                ax[2].plot([f(r["ci_lo"]), f(r["ci_hi"])], [y, y], color=c_, lw=1.5)
                ax[2].plot(f(r["difference"]), y, "o", ms=4, color=c_,
                           label=lab if (i == 0 and q == 0) else None)
    ax[2].axvline(0, color=INK2, lw=0.8)
    ax[2].set_yticks(range(len(core)))
    ax[2].set_yticklabels(core)
    ax[2].invert_yaxis()
    ax[2].set_xlabel("selected − unselected ΔLL / read")
    n = cmp_[0]["common_n"] if cmp_ else ""
    ax[2].set_title("c  same {} transcripts, same folds".format(n), loc="left", color=INK)
    ax[2].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    save(fig, "missing_eval_selected_vs_unselected")


def d2():
    mg = table("qc", "metagene.tsv")
    ys = table("missing_eval", "yeast_single_library_summary.tsv")
    pw = None
    for tag in ("_n50", ""):
        pw = table("synthetic", "power_curve_archer_yeast_multinomial{}.tsv".format(tag))
        if pw:
            break
    if not (mg and ys):
        print("[mfig] d2 skipped: inputs missing")
        return
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
    fig.subplots_adjust(wspace=0.38, bottom=0.3)
    fig.suptitle("Yeast TCP-seq: single SSU library — within-library evidence only",
                 color=ORANGE, fontsize=10, y=1.02)
    for k in ("archer_yeast", SEL):
        pts = sorted((int(r["position"]), f(r["value"])) for r in mg if r["dataset"] == k
                     and r["replicate"] == "rep1" and r["metric"] == "psite_metagene_frac"
                     and r["model"] == "length_resolved")
        if pts:
            x, y = zip(*pts)
            ax[0].plot(x, np.array(y) / max(y), color=DATASET_COLOR[k] if k != "archer_yeast" else ORANGE,
                       label=DATASET_LABEL[k])
    ax[0].set_xlim(-60, 40)
    ax[0].axvline(0, color=INK2, lw=0.8, ls=":")
    ax[0].set_xlabel("P-site position relative to main AUG (nt)")
    ax[0].set_ylabel("metagene (scaled to peak)")
    ax[0].set_title("a  start-codon metagene", loc="left", color=INK)
    ax[0].legend(fontsize=7, loc="upper left")
    fams = ["SW", "ST", "AUG", "UORF", "SW+ST+AUG"]
    bins = [1, 5, 10, 20]
    for i, fam in enumerate(fams):
        for j, b in enumerate(bins):
            r = next((x for x in ys if x["family"] == fam and int(x["bin_width"]) == b), None)
            if r is None:
                continue
            y = i + (j - 1.5) * 0.17
            col = {1: BLUE, 5: AQUA, 10: ORANGE, 20: GREY}[b]
            ax[1].plot([f(r["ci_lo"]), f(r["ci_hi"])], [y, y], color=col, lw=1.5)
            ax[1].plot(f(r["heldout_dll_read_weighted"]), y, "o", ms=4, color=col,
                       label="{} nt".format(b) if i == 0 else None)
            if np.isfinite(f(r["null_mean"])):
                ax[1].plot(f(r["null_mean"]), y, "x", ms=5, mew=1.5, color=INK2,
                           label="matched-null mean" if (i == 0 and j == 0) else None)
    ax[1].axvline(0, color=INK2, lw=0.8)
    ax[1].set_yticks(range(len(fams)))
    ax[1].set_yticklabels(fams)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("held-out chromosome ΔLL / read (rep1→rep1)")
    ax[1].set_title("b  within-library effects", loc="left", color=INK)
    ax[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=5,
                 frameon=False, columnspacing=1.0, handletextpad=0.3)
    if pw:
        styles = {"SW": ("-", "o", 3), "ST": ("--", "s", 5), "AUG": (":", "o", 3)}
        for fam, truth, col in (("SW", "B", AQUA), ("ST", "C", BLUE), ("AUG", "D", ORANGE)):
            pts = sorted((int(f(r["target_median_depth"])), f(r["value"]), f(r.get("p_lo")), f(r.get("p_hi")))
                         for r in pw if r["model"] == fam and r["truth"] == truth
                         and r["metric"] == "p_recovered_within" and int(f(r["size_index"])) == 1)
            if pts:
                x, v, lo, hi = map(np.array, zip(*pts))
                ls, mk, ms = styles[fam]
                ax[2].plot(x, v, ls=ls, marker=mk, ms=ms, color=col, label=fam,
                           mfc="none" if fam == "ST" else col, zorder=3 if fam == "ST" else 2)
                if np.isfinite(lo).all():
                    ax[2].fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
        d = [f(r["observed_median_depth"]) for r in ys]
        if d and np.isfinite(d[0]):
            ax[2].axvline(d[0], color=INK2, ls=":", lw=1, label="observed median depth")
        ax[2].axhline(0.8, color=INK2, lw=0.6, ls="--")
        ax[2].set_xscale("log")
        ax[2].set_ylim(-0.03, 1.05)
        ax[2].legend(fontsize=7, loc="center left")
    ax[2].set_xlabel("median reads per leader")
    ax[2].set_ylabel("P(recovered), within library")
    ax[2].set_title("c  multinomial (sampling-only) power", loc="left", color=INK)
    save(fig, "missing_eval_yeast_single_library")


def d3():
    s = table("downstream", "downstream_utility_final.tsv")
    if not s:
        print("[mfig] d3 skipped: inputs missing")
        return
    panels = [("akirtava_yeast", "TEff", "rep1", "Akirtava TEff"),
              ("akirtava_yeast", "MeanYFP", "rep1", "Akirtava MeanYFP"),
              ("sample2019_human", "rl", "rep1", "Sample MRL (40S rep1)"),
              ("sample2019_human", "rl", "rep2", "Sample MRL (40S rep2)")]
    models = ["seq", "40S", "seq+40S", "late_fusion", "residual"]
    fig, ax = plt.subplots(2, len(panels), figsize=(14, 6.2))
    fig.subplots_adjust(hspace=0.55, wspace=0.25)
    combos = ["seq+40S", "late_fusion", "residual"]
    for j, (ds, tgt, src, title) in enumerate(panels):
        split = "chrom_folds" if ds.startswith("akirtava") else "random_balanced"
        unit = "gene" if ds.startswith("akirtava") else "leader"
        rs = {r["model"]: r for r in s if r["dataset"] == ds and r["target"] == tgt
              and r["fortyS_source"] == src and r["split_mode"] == split
              and r["bootstrap_unit"] == unit and r["stratum"] == "all"}
        if not rs:
            continue
        v = [f(rs[m]["r2"]) for m in models]
        ax[0, j].bar(range(len(models)), v, color=[GREY, ORANGE, BLUE, AQUA, AQUA], width=0.7)
        for i, val in enumerate(v):
            ax[0, j].text(i, val + 0.005, "{:.3f}".format(val), ha="center", fontsize=6.5, color=INK2)
        ax[0, j].set_xticks(range(len(models)))
        ax[0, j].set_xticklabels(models, fontsize=7, rotation=30, ha="right")
        ax[0, j].set_title("{}  n={}".format(title, rs["seq"]["n"]), loc="left", color=INK, fontsize=8.5)
        ax[0, j].set_ylim(0, max(v) * 1.12)
        for i, m in enumerate(combos):
            r = rs[m]
            ax[1, j].plot([f(r["delta_ci_lo"]), f(r["delta_ci_hi"])], [i, i], color=BLUE, lw=1.6)
            ax[1, j].plot(f(r["delta_r2"]), i, "o", color=BLUE, ms=4)
        ax[1, j].axvline(0, color=INK2, lw=0.8)
        ax[1, j].set_yticks(range(len(combos)))
        ax[1, j].set_yticklabels(combos if j == 0 else [], fontsize=7)
        ax[1, j].set_ylim(len(combos) - 0.5, -0.5)
        ax[1, j].set_xlim(-0.012, 0.03)
        ax[1, j].set_xlabel("ΔR² vs sequence (95% CI)", fontsize=8)
        ax[1, j].set_title("40S alone: ΔR² {:+.3f}".format(f(rs["40S"]["delta_r2"])),
                           loc="left", color=INK2, fontsize=7.5)
    ax[0, 0].set_ylabel("held-out R²")
    save(fig, "missing_eval_reporter_utility")


def fig9():
    s = table("downstream", "downstream_utility_final.tsv")
    if not s:
        print("[mfig] fig9 skipped: inputs missing")
        return
    fig, ax = plt.subplots(figsize=(2.5, 1.75))
    rows = [("akirtava_yeast", "TEff", "rep1", "chrom_folds", "gene", "Akirtava TEff"),
            ("akirtava_yeast", "MeanYFP", "rep1", "chrom_folds", "gene", "Akirtava MeanYFP"),
            ("sample2019_human", "rl", "rep1", "random_balanced", "leader", "Sample MRL (40S rep1)"),
            ("sample2019_human", "rl", "rep2", "random_balanced", "leader", "Sample MRL (40S rep2)")]
    labels = []
    for i, (ds, tgt, src, split, unit, lab) in enumerate(rows):
        r = next((x for x in s if x["dataset"] == ds and x["target"] == tgt
                  and x["fortyS_source"] == src and x["split_mode"] == split
                  and x["bootstrap_unit"] == unit and x["stratum"] == "all"
                  and x["model"] == "seq+40S"), None)
        if r is None:
            labels.append(lab)
            continue
        d, lo, hi = f(r["delta_r2"]), f(r["delta_ci_lo"]), f(r["delta_ci_hi"])
        ax.plot([lo, hi], [i, i], color=BLUE, lw=1.6)
        ax.plot(d, i, "o", ms=4.5, color=BLUE)
        labels.append(lab)
        ax.annotate("{:+.4f} [{:.4f}, {:.4f}]".format(d, lo, hi) if abs(d) < 0.005
                    else "{:+.3f} [{:.3f}, {:.3f}]".format(d, lo, hi),
                    xy=(1.02, i), xycoords=("axes fraction", "data"),
                    va="center", fontsize=7, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(-0.005, 0.03)
    ax.set_xticks([0, 0.01, 0.02, 0.03])
    ax.set_xlabel(r"$\Delta R^2 = R^2_{\mathrm{seq+40S}} - R^2_{\mathrm{seq}}$")
    save(fig, "fig9_reporter_utility")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="d1,d2,d3,fig9")
    args = ap.parse_args()
    for k in args.only.split(","):
        {"d1": d1, "d2": d2, "d3": d3, "fig9": fig9}[k]()


if __name__ == "__main__":
    main()
