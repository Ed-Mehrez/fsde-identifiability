"""Figure for the Gate 1 fBm pilot: three small multiples against the true H.

    python pilot/plot_gate1.py [results/pilot_gate1]
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "results/pilot_gate1"
cells = pd.read_csv(f"{out}/cells.csv")

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STYLE = {  # fixed categorical order; marker shape is the secondary encoding
    "chronos-2": ("#2a78d6", "o"),
    "timesfm-2.5": ("#eb6834", "s"),
    "chronos-bolt-small": ("#1baf7a", "^"),
}
PANELS = [
    ("H_fc", "Forecast-implied H", "oracle_H_fc", "oracle (= true H)"),
    ("cov80_long", "80% interval coverage, horizons 33–64", 0.8, "target 0.80"),
    ("rel_pinball_long", "Quantile loss ÷ oracle, horizons 33–64", 1.0, "oracle = 1"),
]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), facecolor=SURF)
for ax, (col, title, ref, ref_label) in zip(axes, PANELS):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    orc = cells[cells.model == "oracle"].sort_values("H")
    yref = orc[ref] if isinstance(ref, str) else [ref] * len(orc)
    ax.plot(orc.H, yref, ls="--", lw=1.5, color=INK2, label=ref_label, zorder=2)
    for m, (c, mk) in STYLE.items():
        g = cells[cells.model == m].sort_values("H")
        if g.empty:
            continue
        ax.plot(g.H, g[col], color=c, lw=2, marker=mk, ms=6, mec=SURF, mew=1.2, label=m, zorder=3)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.set_xlabel("true H")
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.text(0.01, -0.03, "fBm, context 512, horizon 64, 500 series per H. Oracle = exact Gaussian "
         "conditional forecast.", color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(f"{out}/gate1_fbm.png", dpi=200, bbox_inches="tight", facecolor=SURF)
print(f"wrote {out}/gate1_fbm.png")
