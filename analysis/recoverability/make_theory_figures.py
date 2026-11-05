import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.recoverability.make_figures import (BLUE, GREY, INK, INK2, ORANGE, AQUA, f, fig4_power,
                                                  save, table)

DS = {"archer_yeast": ("yeast, multinomial", AQUA, "o"),
      "gse139131_unsel": ("unselected human, DM κ=69", ORANGE, "s"),
      "gse174329_eif3b": ("eIF3B Sel-TCP, DM κ=125", BLUE, "^")}


def main():
    val = table("theory", "theory_validation.tsv")
    d80 = table("theory", "theory_required_depth.tsv")
    grid = table("theory", "theory_power_grid.tsv")
    if not (val and d80 and grid):
        print("[tfig] inputs missing")
        return
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    fig.subplots_adjust(wspace=0.32, bottom=0.26)

    R = [r for r in val if r["truth"] != "A" and r["direction"] == "rep1->rep2"]
    for key, (lab, col, mk) in DS.items():
        rs = [r for r in R if r["dataset"] == key]
        if not rs:
            continue
        x = np.array([f(r["pred_power"]) for r in rs])
        y = np.array([f(r["sim_power"]) for r in rs])
        lo = np.array([f(r["sim_power_lo"]) for r in rs])
        hi = np.array([f(r["sim_power_hi"]) for r in rs])
        ax[0].errorbar(x, y, yerr=[y - lo, hi - y], fmt=mk, ms=4, color=col, alpha=0.75,
                       elinewidth=0.6, capsize=0, label=lab)
    ax[0].plot([0, 1], [0, 1], color=INK2, lw=0.8)
    e = [abs(f(r["pred_power"]) - f(r["sim_power"])) for r in R]
    ax[0].text(0.03, 0.95, "MAE {:.3f}  (n={})".format(np.mean(e), len(e)), transform=ax[0].transAxes,
               va="top", fontsize=8, color=INK2)
    ax[0].set_xlabel("predicted P(recovered)")
    ax[0].set_ylabel("simulated P(recovered)  (Wilson 95% CI)")
    ax[0].set_title("a  power, every cell", loc="left", color=INK)
    ax[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False)

    for key, (lab, col, mk) in DS.items():
        for r in d80:
            if r["dataset"] != key:
                continue
            x, y = f(r["pred_d80"]), f(r["sim_d80"])
            if r["sim_d80_op"] == ">":
                ax[1].annotate("", xy=(x, y * 1.6), xytext=(x, y), arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
            if r["sim_d80_op"] == "<=":
                ax[1].annotate("", xy=(x, y / 1.6), xytext=(x, y), arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
            ax[1].plot(x, y, mk, ms=5, color=col, mfc=col if r["sim_d80_op"] == "=" else "none")
    lim = [3, 8e4]
    ax[1].plot(lim, lim, color=INK2, lw=0.8)
    ax[1].axhline(500, color=GREY, lw=0.6, ls=":")
    ax[1].axhline(5, color=GREY, lw=0.6, ls=":")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlim(lim)
    ax[1].set_ylim(2, 2000)
    ax[1].set_xlabel("predicted depth for 80% power (reads/leader)")
    ax[1].set_ylabel("simulated depth for 80% power")
    ax[1].set_title("b  required depth", loc="left", color=INK)
    ax[1].text(0.03, 0.97, "open / arrows: outside the\nsimulated range 5–500", transform=ax[1].transAxes,
               va="top", fontsize=7, color=INK2)

    sim = table("synthetic", "power_curve_archer_yeast_multinomial_n50.tsv") or []
    simu = table("synthetic", "power_curve_gse139131_unsel_dm_n50.tsv") or []
    styles = {0: ":", 1: "-", 2: "--"}
    for key, src in (("archer_yeast", sim), ("gse139131_unsel", simu)):
        lab, col, mk = DS[key]
        for size in (0, 1, 2):
            g = sorted((f(r["depth"]), f(r["p_det"])) for r in grid if r["dataset"] == key
                       and r["family"] == "SW" and r["truth"] == "B" and int(f(r["size_index"])) == size)
            if g:
                ax[2].plot(*zip(*g), ls=styles[size], color=col, lw=1.4)
            pts = sorted((f(r["target_median_depth"]), f(r["value"])) for r in src if r["model"] == "SW"
                         and r["truth"] == "B" and int(f(r["size_index"])) == size
                         and r["metric"] == "p_recovered")
            if pts:
                ax[2].plot(*zip(*pts), mk, ms=4, color=col, mfc="none" if size == 0 else col)
    ax[2].set_xscale("log")
    ax[2].set_xlim(4, 5e4)
    ax[2].axvline(500, color=GREY, lw=0.6, ls=":")
    ax[2].set_ylim(-0.03, 1.03)
    ax[2].set_xlabel("median reads per leader")
    ax[2].set_ylabel("P(recovered)")
    ax[2].set_title("c  SW: theory (lines) vs simulation", loc="left", color=INK)
    from matplotlib.lines import Line2D
    hand = [Line2D([], [], color=DS[k][1], marker=DS[k][2], label=DS[k][0]) for k in ("archer_yeast", "gse139131_unsel")]
    hand += [Line2D([], [], color=INK2, ls=styles[s], label="SW s = {}".format((0.1, 0.25, 0.5)[s])) for s in (0, 1, 2)]
    ax[2].legend(handles=hand, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)

    pred = table("theory", "theory_predictions.tsv")
    for key in ("archer_yeast", "gse139131_unsel"):
        lab, col, mk = DS[key]
        P = sorted((f(r["depth"]), -f(r["mu"])) for r in pred if r["dataset"] == key and r["truth"] == "A"
                   and r["family"] == "SW")
        ax[3].plot(*zip(*P), color=col, lw=1.4)
        S = sorted((f(r["depth"]), -f(r["sim_mean"])) for r in val if r["dataset"] == key and r["truth"] == "A"
                   and r["family"] == "SW" and r["direction"] == "rep1->rep2")
        ax[3].plot(*zip(*S), mk, ms=4, color=col)
    ax[3].set_xscale("log")
    ax[3].set_yscale("log")
    ax[3].set_xlabel("median reads per leader")
    ax[3].set_ylabel("− mean held-out ΔLL/read, no effect (SW)")
    ax[3].set_title("d  estimation penalty: 1/depth vs floor", loc="left", color=INK)
    save(fig, "theory_validation")
    for key, noise, tag in (("gse174329_eif3b", "dm", ""), ("gse139131_unsel", "dm", "n50"),
                            ("archer_yeast", "multinomial", "n50")):
        fig4_power(key, noise, tag, theory=True)


if __name__ == "__main__":
    main()
