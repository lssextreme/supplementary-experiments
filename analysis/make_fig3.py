# -*- coding: utf-8 -*-
"""回复信配图 3：Sichuan 最差 3 张失败样例（输入/GT/预测/叠加），带 patch ID 与 IoU。"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

_REPO = os.path.join(r"D:\学业相关\HXJ大论文\lss_BsiNet", "revise_exp")
_PARENT = os.path.dirname(_REPO)
for _p in (_REPO, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from common import set_seed, metrics_from_preds
from harnet_model import build_model
import evaluate

OUT = r"D:\a_firstpaper\compare\work_figs"

set_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ckpt = os.path.join(config.CKPT_DIR, "full_sichuan_seed42_w1_1_1", "best.pth")
net = build_model("full", device=str(device))
net.load_state_dict(torch.load(ckpt, map_location=device))
net.to(device)
net.eval()

ds = config.DATASETS["sichuan"]
test_img = ds["test"]["img"]
test_mask = ds["test"]["mask"]
files = sorted(f for f in os.listdir(test_img) if not f.startswith("."))

# 全部 20 张推理，取 IoU 最低 3 张
records = []
with torch.no_grad():
    for f in files:
        mp = os.path.join(test_mask, f)
        if not os.path.exists(mp):
            continue
        x = evaluate.load_rgb_tensor(os.path.join(test_img, f))
        x, H, W = evaluate.pad_to_multiple(x)
        x = x.to(device, dtype=torch.float32)
        outs = net(x)
        prob = torch.sigmoid(outs[1]).squeeze(0).squeeze(0)[:H, :W]
        pred = (prob > config.MASK_THRESHOLD).cpu().numpy().astype(np.uint8)
        gt = evaluate.load_mask(mp)
        m = metrics_from_preds(pred, gt)
        records.append((f, m["IoU"], pred, gt))

records.sort(key=lambda t: t[1])
top3 = records[:3]
print("top3:", [(f, round(i, 2)) for f, i, _, _ in top3])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                     "axes.titlesize": 7.5, "savefig.dpi": 300})

fig, axes = plt.subplots(3, 4, figsize=(6.3, 4.6))
col_titles = ["Input image", "Ground truth", "Prediction",
              "Overlay (red: GT,\nlime: pred.)"]
for j, t in enumerate(col_titles):
    axes[0, j].set_title(t)

for i, (f, iou, pred, gt) in enumerate(top3):
    rgb = np.array(Image.open(os.path.join(test_img, f)).convert("RGB"))
    if rgb.shape[:2] != pred.shape:
        rgb = np.array(Image.open(os.path.join(test_img, f)).convert("RGB")
                       .resize((pred.shape[1], pred.shape[0])))
    axes[i, 0].imshow(rgb)
    axes[i, 1].imshow(gt, cmap="gray")
    axes[i, 2].imshow(pred, cmap="gray")
    axes[i, 3].imshow(rgb)
    axes[i, 3].contour(gt, colors="red", linewidths=0.8)
    axes[i, 3].contour(pred, colors="lime", linewidths=0.8)
    for ax in axes[i]:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)

# Row labels ("Example N / IoU = XX%") drawn inside the left margin of the
# figure canvas, right-aligned against the first column, so they can never
# be clipped at the page edge.
fig.subplots_adjust(left=0.16, right=0.995, top=0.92, bottom=0.02,
                    wspace=0.03, hspace=0.1)
for i, (f, iou, pred, gt) in enumerate(top3):
    pos = axes[i, 0].get_position()
    fig.text(pos.x0 - 0.012, pos.y0 + pos.height / 2.0,
             f"Example {i+1}\nIoU = {iou:.2f}%",
             ha="right", va="center", fontsize=7, linespacing=1.2)
fig.savefig(os.path.join(OUT, "fig_failure.pdf"))
plt.close(fig)
print("fig_failure.pdf done")
