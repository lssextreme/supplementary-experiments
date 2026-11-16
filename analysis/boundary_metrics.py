# -*- coding: utf-8 -*-
"""边界头在全测试集上的质量（不同阈值），供 R2 回复引用。"""
import os
import sys

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(r"D:\学业相关\HXJ大论文\lss_BsiNet", "revise_exp")
_PARENT = os.path.dirname(_REPO)
for _p in (_REPO, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from common import set_seed, accumulate_confusion, metrics_from_confusion
from harnet_model import build_model


def load_rgb_tensor(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)[None])


def load_gray(path):
    return (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)


def pad_to_multiple(x, mult=32):
    _, _, h, w = x.shape
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph))
    return x, h, w


set_seed(42)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
ckpt = os.path.join(config.CKPT_DIR, "full_shandong_seed42_w1.0_1.0_1.0", "best.pth")
net = build_model("full", device=device)
net.load_state_dict(torch.load(ckpt, map_location=device))
net.to(device)
net.eval()

ds = config.DATASETS["shandong"]
bnd_dir = os.path.join(config.DATA_ROOT, "Shangdong", "test", "boundary")
files = sorted(f for f in os.listdir(ds["test"]["img"]) if not f.startswith("."))
files = [f for f in files if os.path.exists(os.path.join(bnd_dir, f))]

# 收集全部概率图
probs = []
gts = []
with torch.no_grad():
    for f in files:
        x = load_rgb_tensor(os.path.join(ds["test"]["img"], f))
        x, H, W = pad_to_multiple(x)
        x = x.to(device, dtype=torch.float32)
        outs = net(x)
        pb = torch.sigmoid(outs[0]).squeeze(0).squeeze(0)[:H, :W].cpu().numpy()
        probs.append(pb)
        gts.append(load_gray(os.path.join(bnd_dir, f)))

all_p = np.concatenate([p.ravel() for p in probs])
all_g = np.concatenate([g.ravel() for g in gts])
print(f"n={len(files)}  边界预测 max={all_p.max():.3f} mean={all_p.mean():.4f}")

print("阈值 → (Acc, F1, Recall, Prec, IoU)")
for thr in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
    pred = (all_p > thr).astype(np.uint8)
    a, b, c, d = accumulate_confusion(pred, all_g)
    m = metrics_from_confusion(a, b, c, d)
    print(f"thr={thr:.2f}: Acc={m['Acc']:.2f} F1={m['F1']:.2f} Recall={m['Recall']:.2f} "
          f"Prec={m['Prec']:.2f} IoU={m['IoU']:.2f}")

# 最优 F1 阈值扫描
best = None
for thr in np.arange(0.02, 0.6, 0.01):
    pred = (all_p > thr).astype(np.uint8)
    a, b, c, d = accumulate_confusion(pred, all_g)
    m = metrics_from_confusion(a, b, c, d)
    if best is None or m["F1"] > best[1]:
        best = (thr, m["F1"], m)
print(f"best-F1: thr={best[0]:.2f} F1={best[1]:.2f} Recall={best[2]['Recall']:.2f} Prec={best[2]['Prec']:.2f} IoU={best[2]['IoU']:.2f}")
