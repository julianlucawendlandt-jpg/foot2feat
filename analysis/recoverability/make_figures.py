import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.recoverability import config as C

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
DATASET_COLOR = {"gse174329_eif3b": BLUE, "gse139131_unsel": ORANGE,
                 "gse139131_eif3b": AQUA, "archer_yeast": GREY}
DATASET_LABEL = {"gse174329_eif3b": "eIF3B Sel-TCP (GSE174329)",
                 "gse139131_unsel": "unselected 40S (GSE139131)",
                 "gse139131_eif3b": "eIF3B Sel-TCP (GSE139131)",
                 "archer_yeast": "yeast TCP-seq (Archer)"}
FAMILY_COLOR = {"ST": BLUE, "AUG": ORANGE, "SW": AQUA}
INF_SYMBOL = "∞"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 150, "savefig.bbox": "tight",
})


def table(*parts):
    p = C.results_path(*parts)
    return C.read_tsv(p) if os.path.exists(p) else None


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(C.results_path("figures", "{}.{}".format(name, ext)))
    plt.close(fig)
    print("[fig] {}".format(name))


def fig1_qc():
    qc, per_tx, mg = (table("qc", "qc_summary.tsv"), table("qc", "reads_per_transcript.tsv"),
                      table("qc", "metagene.tsv"))
    if not (qc and per_tx and mg):
        print("[fig] 1 skipped: QC tables missing")
        return
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    keys = ["gse174329_eif3b", "gse139131_unsel", "gse139131_eif3b", "archer_yeast"]
    for k in keys[:3]:
        v = np.array([f(r["n_reads"]) for r in per_tx
                      if r["dataset"] == k and r["replicate"] == "rep1"])
        if len(v) == 0:
            continue
        xs = np.logspace(0, 3.5, 60)
        ax[0].plot(xs, [(v >= x).sum() for x in xs], color=DATASET_COLOR[k], label=DATASET_LABEL[k])
    ax[0].set_xscale("log")
    ax[0].set_xlabel("leader reads (rep1)")
    ax[0].set_ylabel("leaders with at least x reads")
    ax[0].legend(fontsize=7)
    ax[0].set_title("a  depth per leader", loc="left", color=INK)
    rows = [(k, r["replicate"], f(r["value"])) for k in keys for r in qc
            if r["dataset"] == k and r["metric"] == "leader_frac_of_unique"]
    labels = ["{}\n{}".format(DATASET_LABEL[k].split(" (")[0], rep) for k, rep, _ in rows]
    ax[1].bar(range(len(rows)), [v for *_, v in rows], color=[DATASET_COLOR[k] for k, *_ in rows],
              width=0.7)
    for i, (_, _, v) in enumerate(rows):
        ax[1].text(i, v + 0.01, "{:.0%}".format(v), ha="center", fontsize=7, color=INK2)
    ax[1].set_xticks(range(len(rows)))
    ax[1].set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax[1].set_ylabel("unique footprints in leaders")
    ax[1].set_title("b  leader enrichment", loc="left", color=INK)
    for k in keys:
        pts = sorted((int(r["position"]), f(r["value"])) for r in mg if r["dataset"] == k
                     and r["replicate"] == "rep1" and r["metric"] == "psite_metagene_frac"
                     and r["model"] == "length_resolved")
        if pts:
            x, y = zip(*pts)
            ax[2].plot(x, np.array(y) / max(y), color=DATASET_COLOR[k], label=DATASET_LABEL[k])
    ax[2].set_xlim(-60, 40)
    ax[2].axvline(0, color=INK2, lw=0.8, ls=":")
    ax[2].set_xlabel("P-site position relative to main AUG (nt)")
    ax[2].set_ylabel("metagene (scaled to peak)")
    ax[2].set_title("c  start-codon metagene", loc="left", color=INK)
    ax[2].legend(fontsize=7, loc="upper left")
    save(fig, "fig1_qc")


def fig2_resolution():
    res = table("resolution", "replicate_resolution.tsv")
    if not res:
        print("[fig] 2 skipped")
        return
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for i, (metric, title) in enumerate((("pearson_mean", "a  per-transcript Pearson r"),
                                         ("pooled_posadj_r", "b  position-adjusted pooled r"))):
        for k in ("gse174329_eif3b", "gse139131_unsel"):
            for variant, ls, lab in (("", "-", ""), ("_ceiling", "--", " (multinomial ceiling)")):
                pts = sorted((int(r["bin_width"]), f(r["value"]), f(r.get("ci_lo")),
                              f(r.get("ci_hi")))
                             for r in res if r["dataset"] == k and r["metric"] == metric + variant
                             and r["depth_stratum"] == "all")
                if not pts:
                    continue
                b, v, lo, hi = map(np.array, zip(*pts))
                ax[i].plot(b, v, ls=ls, marker=None if variant else "o", ms=4,
                           color=DATASET_COLOR[k], label=DATASET_LABEL[k] + lab)
                if not variant and np.isfinite(lo).all():
                    ax[i].fill_between(b, lo, hi, color=DATASET_COLOR[k], alpha=0.15, lw=0)
        ax[i].set_xscale("log")
        ax[i].set_xticks(C.RESOLUTION_BINS)
        ax[i].set_xticklabels(C.RESOLUTION_BINS)
        ax[i].set_xlabel("bin width (nt)")
        ax[i].set_title(title, loc="left", color=INK)
    ax[0].set_ylabel("replicate agreement")
    ax[1].legend(fontsize=7, loc="lower right")
    save(fig, "fig2_resolution")
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    strata = ("20-49", "50-99", "100-199", "200-inf")
    shades = ("#a9c8ef", "#6fa3e3", "#2a78d6", "#1a4f93")
    for s_, c_ in zip(strata, shades):
        pts = sorted((int(r["bin_width"]), f(r["value"])) for r in res
                     if r["dataset"] == "gse174329_eif3b" and r["metric"] == "pooled_posadj_r"
                     and r["depth_stratum"] == s_)
        if pts:
            b, v = zip(*pts)
            ax.plot(b, v, marker="o", ms=3, color=c_,
                    label="{} reads".format(s_.replace("-inf", "+")))
    ax.set_xscale("log")
    ax.set_xticks(C.RESOLUTION_BINS)
    ax.set_xticklabels(C.RESOLUTION_BINS)
    ax.set_xlabel("bin width (nt)")
    ax.set_ylabel("position-adjusted pooled r")
    ax.set_title("eIF3B replicates by depth (min of the two)", loc="left", color=INK, fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "fig2b_resolution_by_depth")


def fig3_effects(key="gse174329_eif3b"):
    eff = table("effects", "nested_effects.tsv")
    if not eff:
        print("[fig] 3 skipped")
        return
    rows = [r for r in eff if r.get("panel", r["dataset"]) == key
            and r["metric"] == "heldout_dll_read_weighted"]
    if not rows:
        print("[fig] 3 skipped: no rows for {}".format(key))
        return
    models = ["SW", "ST", "AUG", "UORF", "SW+ST+AUG", "S5"]
    bins = [1, 5, 10, 20]
    fig, ax = plt.subplots(1, len(bins), figsize=(12, 3.6), sharey=True)
    for j, b in enumerate(bins):
        for i, m in enumerate(models):
            for dk, (dirs, c_, off) in {
                    "within": (("rep1->rep1", "rep2->rep2"), GREY, -0.15),
                    "cross": (("rep1->rep2", "rep2->rep1"), BLUE, 0.15)}.items():
                rs = [r for r in rows if r["model"] == m and int(r["bin_width"]) == b
                      and r["replicate"] in dirs]
                for k_, r in enumerate(rs):
                    v, lo, hi = f(r["value"]), f(r["ci_lo"]), f(r["ci_hi"])
                    y = i + off + (k_ - 0.5) * 0.12
                    ax[j].plot([lo, hi], [y, y], color=c_, lw=1.5)
                    ax[j].plot(v, y, "o", color=c_, ms=4,
                               label=("cross-replicate" if dk == "cross" else "within-replicate")
                               if (i == 0 and k_ == 0 and j == 0) else None)
        nul = table("nulls", "shuffle_nulls.tsv") or []
        for i, m in enumerate(models):
            nm = [f(r["value"]) for r in nul if r["dataset"] == key and r["model"] == m
                  and int(f(r["bin_width"])) == b and r["metric"] == "null_mean"
                  and r.get("cross_replicate") == "True"]
            if nm:
                ax[j].plot(np.mean(nm), i + 0.15, "x", color=ORANGE, ms=7, mew=2,
                           label="matched-null mean" if (j == 0 or not ax[j].get_legend_handles_labels()[0]) and i == 0 else None)
        ax[j].axvline(0, color=INK2, lw=0.8)
        ax[j].set_title("{} nt".format(b), loc="left", color=INK)
        ax[j].set_xlabel("held-out ΔLL / read")
    ax[0].set_yticks(range(len(models)))
    ax[0].set_yticklabels(models)
    ax[0].invert_yaxis()
    handles, labels = [], []
    for a in ax:
        for h, l_ in zip(*a.get_legend_handles_labels()):
            if l_ not in labels:
                handles.append(h)
                labels.append(l_)
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.12))
    save(fig, "fig3_effects_{}".format(key))


def fig4_power(key="gse174329_eif3b", noise="dm", tag="", theory=False):
    stem = "{}_{}{}".format(key, noise, "_" + tag if tag else "")
    pw = table("synthetic", "power_curve_{}.tsv".format(stem))
    qc = table("qc", "qc_summary.tsv")
    if not pw:
        print("[fig] 4 skipped")
        return
    grid = table("theory", "theory_power_grid.tsv") if theory else None
    val = table("theory", "theory_validation.tsv") if theory else None
    if theory and not grid:
        print("[fig] 4 theory overlay skipped: run 11_recoverability_theory first")
        return
    fams = [("SW", "B"), ("ST", "C"), ("AUG", "D")]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    shades = {0: 0.35, 1: 0.65, 2: 1.0}
    for j, (fam, truth) in enumerate(fams):
        for size in (0, 1, 2):
            pts = sorted((int(f(r["target_median_depth"])), f(r["value"])) for r in pw
                         if r["model"] == fam and r["truth"] == truth
                         and r["metric"] == "p_recovered" and int(f(r["size_index"])) == size)
            if pts:
                x, y = zip(*pts)
                ax[j].plot(x, y, marker="o", ms=3, color=FAMILY_COLOR[fam], alpha=shades[size],
                           label="size {}".format(size))
            if grid:
                lo, hi = (min(x), max(x)) if pts else (5, 500)
                th = sorted((f(r["depth"]), f(r["p_det"])) for r in grid if r["dataset"] == key
                            and r["family"] == fam and r["truth"] == truth
                            and int(f(r["size_index"])) == size and lo <= f(r["depth"]) <= hi)
                if th:
                    ax[j].plot(*zip(*th), ls="--", lw=1.4, color=FAMILY_COLOR[fam], alpha=shades[size])
        if val:
            e = [abs(f(r["pred_power"]) - f(r["sim_power"])) for r in val if r["dataset"] == key
                 and r["family"] == fam and r["truth"] == truth and r["direction"] == "rep1->rep2"]
            if e:
                ax[j].text(1.0, 1.02, "analytic MAE {:.3f}".format(np.mean(e)), ha="right",
                           va="bottom", transform=ax[j].transAxes, fontsize=7, color=INK2)
        if qc:
            for k, ls in (("gse174329_eif3b", "-"), ("gse139131_unsel", ":")):
                d = [f(r["value"]) for r in qc if r["dataset"] == k
                     and r["metric"] == "median_reads_per_leader_ge5"]
                if d:
                    ax[j].axvline(np.median(d), color=DATASET_COLOR[k], ls=ls, lw=1,
                                  label="observed: " + DATASET_LABEL[k].split(" (")[0]
                                  if j == 0 else None)
        allp = [f(r["value"]) for r in pw if r["model"] == fam and r["truth"] == truth
                and r["metric"] == "p_recovered"]
        if allp and min(allp) >= 1.0:
            ax[j].text(0.5, 0.5, "{}all sizes and depths: P = 1".format("simulated, " if grid else ""),
                       transform=ax[j].transAxes,
                       ha="center", fontsize=8, color=INK2)
        ax[j].set_xscale("log")
        ax[j].axhline(0.8, color=INK2, lw=0.6, ls="--")
        ax[j].set_title({"SW": "a  local sequence", "ST": "b  structure",
                         "AUG": "c  start recognition"}[fam], loc="left", color=INK)
        ax[j].set_xlabel("median reads per leader")
    ax[0].set_ylabel("P(recovered)")
    if grid:
        from matplotlib.lines import Line2D
        h, _ = ax[0].get_legend_handles_labels()
        nsim = int(f(pw[0]["n_sim"]))
        h += [Line2D([], [], color=INK2, marker="o", ms=3, label="simulated ({} per point)".format(nsim)),
              Line2D([], [], color=INK2, ls="--", lw=1.4, label="analytic prediction")]
        ax[0].legend(handles=h, fontsize=7, loc="center right")
    else:
        ax[0].legend(fontsize=7, loc="center right")
    save(fig, "fig4_power_{}{}".format(stem, "_theory" if theory else ""))


def _se_series(ax, pts, color, label, y_inf, offset=1.0):
    x = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    inf = ~np.isfinite(y) | (y > 1e6)
    if inf.all():
        ax.plot(x, np.full(len(x), y_inf * offset), marker="x", ms=5, ls=":", color=color,
                label="{}  (SE = {})".format(label, INF_SYMBOL))
    else:
        ax.plot(x[~inf], y[~inf], marker="o", ms=3, color=color, label=label)


def _inf_band(ax):
    ax.axhspan(8, 40, color=GRID, alpha=0.6, lw=0)
    ax.text(25, 30, "not identifiable at any depth", fontsize=7, color=INK2)


def fig5_identifiability():
    idt = table("chain", "identifiability.tsv")
    if not idt:
        print("[fig] 5 skipped")
        return
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    for (cond, c_, lab), off in zip((("occupancy_only", ORANGE, "40S occupancy only"),
                                     ("occupancy+80S_rho_known", BLUE, "+ 80S, known scale"),
                                     ("occupancy+80S_rho_unknown", GREY, "+ 80S, unknown scale")),
                                    (1.0, 1.0, 0.55)):
        pts = sorted((int(f(r["reads_per_transcript"])), f(r["value"])) for r in idt
                     if r["model"] == "E1_commit_vs_drop" and r.get("condition") == cond
                     and r["metric"] == "se_share")
        if pts:
            _se_series(ax[0], pts, c_, lab, 20.0, off)
    _inf_band(ax[0])
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("reads per transcript")
    ax[0].set_ylabel("SE of commit share (logit)")
    ax[0].set_title("a  commit vs drop-off", loc="left", color=INK)
    ax[0].legend(fontsize=7, loc="lower left")
    for cond, c_, lab in (("window_ends_AUG+2", ORANGE, "window ends at AUG+2"),
                          ("30_CDS_states", BLUE, "+30 CDS states")):
        pts = sorted((int(f(r["bin_width"])), abs(f(r["value"]))) for r in idt
                     if r["model"] == "E2_blur_vs_leak" and r.get("condition") == cond
                     and r["metric"].startswith("corr") and r.get("reads_per_transcript") == "500")
        if pts:
            x, y = zip(*pts)
            ax[1].plot(x, y, marker="o", ms=4, color=c_, label=lab)
    ax[1].set_ylim(0, 1.02)
    ax[1].set_xlabel("bin width (nt)")
    ax[1].set_ylabel("|corr(leak, blur)| of estimates")
    ax[1].set_title("b  blur vs leaky scanning", loc="left", color=INK)
    ax[1].legend(fontsize=7)
    for cond, b_, c_, lab in (("occupancy", "1", BLUE, "occupancy, 1 nt"),
                              ("occupancy", "20", AQUA, "occupancy, 20 nt"),
                              ("main_AUG_output_only", "", ORANGE, "main-AUG output only")):
        pts = sorted((int(f(r["reads_per_transcript"])), f(r["value"])) for r in idt
                     if r["model"] == "E3_reinit_vs_direct" and r.get("condition") == cond
                     and r["metric"] == "se_reinit_logit" and r.get("bin_width", "") == b_)
        if pts:
            _se_series(ax[2], pts, c_, lab, 20.0)
    _inf_band(ax[2])
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("reads per transcript")
    ax[2].set_ylabel("SE of reinitiation logit")
    ax[2].set_title("c  reinitiation vs direct", loc="left", color=INK)
    ax[2].legend(fontsize=7, loc="lower left")
    save(fig, "fig5_identifiability")


def fig6_complexity():
    ch = table("chain", "chain_ablation.tsv")
    eff = table("effects", "nested_effects.tsv") or []
    if not ch:
        print("[fig] 6 skipped")
        return
    keys = [k for k in ("archer_yeast", "gse174329_eif3b") if any(r["dataset"] == k for r in ch)]
    fig, ax = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 3.4), sharey=False)
    ax = np.atleast_1d(ax)
    b = 10
    for j, key in enumerate(keys):
        heldout_metric = lambda r: (r["metric"] == "heldout_dll_vs_P" and int(f(r["bin_width"])) == b
                                    and r["dataset"] == key)
        cross = [r for r in ch if heldout_metric(r) and r.get("cross_replicate") == "True"]
        held = cross or [r for r in ch if heldout_metric(r)]
        for i, m in enumerate(("B", "C")):
            tr = [f(r["value"]) for r in ch if r["dataset"] == key and r["model"] == m
                  and r["metric"] == "train_dll_per_read" and int(f(r["bin_width"])) == b]
            x0 = i * 3
            if tr:
                ax[j].plot([x0] * len(tr), tr, "o", color=GREY, ms=3, alpha=0.6)
                ax[j].plot([x0 - 0.25, x0 + 0.25], [np.mean(tr)] * 2, color=GREY, lw=2,
                           label="training (per fold)" if i == 0 else None)
            hs = [r for r in held if r["model"] == m]
            for k_, r in enumerate(hs):
                x = x0 + 1 + (k_ - (len(hs) - 1) / 2) * 0.12
                ax[j].plot([x, x], [f(r["ci_lo"]), f(r["ci_hi"])], color=BLUE, lw=1.5)
                ax[j].plot(x, f(r["value"]), "o", color=BLUE, ms=4,
                           label="held-out chromosomes (per seed, 95% CI)" if i == 0 and k_ == 0 else None)
        add = [f(r["value"]) for r in eff if r.get("panel", r["dataset"]) == key
               and r["model"] == "SW+ST+AUG" and int(r["bin_width"]) == b
               and r["metric"] == "heldout_dll_read_weighted"
               and (r["replicate"].split("->")[0] != r["replicate"].split("->")[1] or not cross)]
        if add:
            ax[j].axhline(np.mean(add), color=ORANGE, ls="--", lw=1.2,
                          label="additive SW+ST+AUG (held-out)")
        ax[j].axhline(0, color=INK2, lw=0.8)
        ax[j].set_xticks([0, 1, 3, 4])
        ax[j].set_xticklabels(["B train", "B held-out", "C train", "C held-out"], fontsize=7)
        ax[j].set_title("{}  {} ({} nt){}".format("ab"[j], DATASET_LABEL[key], b,
                                                    "" if cross else ", within-library"),
                        loc="left", color=INK, fontsize=9)
        ax[j].legend(fontsize=6.5, loc="upper left")
    ax[0].set_ylabel("ΔLL / read vs position-only")
    save(fig, "fig6_complexity")


FIGS = {1: fig1_qc, 2: fig2_resolution, 3: fig3_effects, 4: fig4_power,
        5: fig5_identifiability, 6: fig6_complexity}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="1,2,3,4,5,6")
    args = ap.parse_args()
    C.ensure_layout()
    for i in [int(x) for x in args.only.split(",")]:
        FIGS[i]()


if __name__ == "__main__":
    main()
