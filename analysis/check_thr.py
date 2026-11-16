# -*- coding: utf-8 -*-
"""检查不同 mask 阈值下 Sichuan 各 seed 的 IoU，看论文 63.18 是否来自阈值差异。"""
import os
import sys

import numpy as np
import torch

_REPO = os.path.join(r"D:\学业相关\HXJ大论文\lss_BsiNet", "revise_exp")
_PARENT = os.path.dirname(_REPO)
for _p in (_REPO, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from common import set_seed, accumulate_confusion, metrics_from_confusion
from harnet_model import build_model
import evaluate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ds = config.DATASETS["sichuan"]
files = sorted(f for f in os.listdir(ds["test"]["img"]) if not f.startswith("."))
files = [f for f in files if os.path.exists(os.path.join(ds["test"]["mask"], f))]

for seed in ["42", "123", "2024"]:
    set_seed(42)
    ckpt = os.path.join(config.CKPT_DIR, f"full_sichuan_seed{seed}_w1_1_1", "best.pth")
    net = build_model("full", device=str(device))
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.to(device)
    net.eval()

    probs = []
    gts = []
    with torch.no_grad():
        for f in files:
            x = evaluate.load_rgb_tensor(os.path.join(ds["test"]["img"], f))
            x, H, W = evaluate.pad_to_multiple(x)
            x = x.to(device, dtype=torch.float32)
            outs = net(x)
            prob = torch.sigmoid(outs[1]).squeeze(0).squeeze(0)[:H, :W].cpu().numpy()
            probs.append(prob)
            gts.append(evaluate.load_mask(os.path.join(ds["test"]["mask"], f)))
    p = np.concatenate([x.ravel() for x in probs])
    g = np.concatenate([x.ravel() for x in gts])
    print(f"seed {seed}:", end=" ")
    for thr in [0.3, 0.4, 0.5, 0.6]:
        pred = (p > thr).astype(np.uint8)
        a, b, c, d = accumulate_confusion(pred, g)
        m = metrics_from_confusion(a, b, c, d)
        print(f"thr{thr}: IoU={m['IoU']:.2f} F1={m['F1']:.2f} R={m['Recall']:.2f} P={m['Prec']:.2f} Acc={m['Acc']:.2f} |", end=" ")
    print()
