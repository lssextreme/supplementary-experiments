# -*- coding: utf-8 -*-
"""生成回复信配图 1（逐 seed IoU）与 2（覆盖率分带分析）。"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT = r"D:\a_firstpaper\compare\work_figs"
DATA = r"D:\data1"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "lines.linewidth": 0.8,
    "axes.linewidth": 0.6,
    "savefig.dpi": 300,
})

# ---------------- Fig 1: per-seed IoU ----------------
with open(r"D:\学业相关\HXJ大论文\lss_BsiNet\revise_exp\outputs\results\multiseed_summary.json",
          encoding="utf-8") as f:
    ms = json.load(f)["full"]

datasets = ["shandong", "netherlands"]
names = {"shandong": "Shandong", "netherlands": "Netherlands", "sichuan": "Sichuan"}
n_test = {"shandong": 73, "netherlands": 66, "sichuan": 20}
colors = ["#2a82aa", "#6aa84f", "#b45f06"]

fig, axes = plt.subplots(1, 2, figsize=(4.4, 1.75), sharey=True)
for ax, ds in zip(axes, datasets):
    vals = [ms[ds]["42"]["IoU"], ms[ds]["123"]["IoU"], ms[ds]["2024"]["IoU"]]
    x = np.arange(3)
    ax.bar(x, vals, width=0.55, color=colors, edgecolor="none")
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.6, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["seed 42", "seed 123", "seed 2024"], rotation=0)
    ax.set_title(f"{names[ds]}  (n = {n_test[ds]})")
    ax.set_ylim(45, 92)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    ax.set_ylabel("IoU (%)")
axes[0].text(1.0, 89.5, f"mean {np.mean([ms['shandong'][s]['IoU'] for s in ms['shandong']]):.2f} \u00b1 {np.std([ms['shandong'][s]['IoU'] for s in ms['shandong']], ddof=0):.2f}",
             ha="center", fontsize=6.5)
axes[1].text(1.0, 89.5, f"mean {np.mean([ms['netherlands'][s]['IoU'] for s in ms['netherlands']]):.2f} \u00b1 {np.std([ms['netherlands'][s]['IoU'] for s in ms['netherlands']], ddof=0):.2f}",
             ha="center", fontsize=6.5)
fig.subplots_adjust(wspace=0.35, left=0.07, right=0.99, top=0.9, bottom=0.13)
fig.savefig(os.path.join(OUT, "fig_per_seed.pdf"))
plt.close(fig)
print("fig_per_seed.pdf done")

# ---------------- Fig 2: coverage-band analysis ----------------
with open(os.path.join(OUT, "band_split_results.json"), encoding="utf-8") as f:
    band = json.load(f)

mask_dirs = {
    "shandong": os.path.join(DATA, "Shangdong", "test", "mask"),
    "netherlands": os.path.join(DATA, "Netherlands", "test", "mask"),
    "sichuan": os.path.join(DATA, "Sichuan", "test", "mask"),
}


def coverage_list(d):
    out = []
    for f in sorted(os.listdir(d)):
        if f.startswith("."):
            continue
        a = np.array(Image.open(os.path.join(d, f)))
        if a.ndim == 3:
            a = a[..., 0]
        out.append(float((a > 127).mean()) * 100.0)
    return np.array(out)


fig, axes = plt.subplots(2, 2, figsize=(4.4, 3.3))
bands = ["all", ">=40%", "20-40%", "<20%"]
labels = ["all", "\u226540%", "20\u201340%", "<20%"]
bar_colors = ["#2a82aa", "#6aa84f", "#e69138", "#c0504d"]

for j, ds in enumerate(datasets):
    ax = axes[0, j]
    vals = [band[ds][b]["metrics"]["IoU"] for b in bands]
    ns = [band[ds][b]["n"] for b in bands]
    xs = np.arange(4)
    for xi, v, c in zip(xs, vals, bar_colors):
        if v > 0:
            ax.bar(xi, v, width=0.6, color=c)
            ax.text(xi, v + 0.7, f"{v:.2f}", ha="center", fontsize=6.5)
    for xi, n in zip(xs, ns):
        if n > 0:
            ax.text(xi, 30, f"n={n}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.7, 3.7)
    ax.set_title(f"{names[ds]}: IoU by coverage band", fontsize=7.5)
    ax.set_ylim(0, 96)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    if j == 0:
        ax.set_ylabel("IoU (%)")

    axh = axes[1, j]
    cov = coverage_list(mask_dirs[ds])
    axh.hist(cov, bins=12, range=(0, 100), color="#7f8c8d", edgecolor="white", lw=0.3)
    axh.axvline(40, color="#c0504d", lw=0.8, ls="--")
    axh.text(41.5, axh.get_ylim()[1] * 0.86, "40%", color="#c0504d", fontsize=6.5)
    axh.set_xlabel("farmland coverage per patch (%)")
    if j == 0:
        axh.set_ylabel("number of test patches")
    axh.spines["top"].set_visible(False)
    axh.spines["right"].set_visible(False)

fig.subplots_adjust(wspace=0.3, hspace=0.55, left=0.07, right=0.99, top=0.93, bottom=0.1)
fig.savefig(os.path.join(OUT, "fig_coverage.pdf"))
plt.close(fig)
print("fig_coverage.pdf done")
