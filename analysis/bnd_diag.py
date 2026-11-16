# -*- coding: utf-8 -*-
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
from common import set_seed
from harnet_model import build_model


def load_rgb_tensor(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)[None])


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
files = sorted(f for f in os.listdir(ds["test"]["img"]) if not f.startswith("."))[:6]
with torch.no_grad():
    for f in files:
        x = load_rgb_tensor(os.path.join(ds["test"]["img"], f))
        x, H, W = pad_to_multiple(x)
        x = x.to(device, dtype=torch.float32)
        outs = net(x)
        pb = torch.sigmoid(outs[0]).squeeze(0).squeeze(0)[:H, :W].cpu().numpy()
        pm = torch.sigmoid(outs[1]).squeeze(0).squeeze(0)[:H, :W].cpu().numpy()
        pd = outs[2].squeeze(0).squeeze(0)[:H, :W].cpu().numpy()
        gt_b = (np.array(Image.open(os.path.join(config.DATA_ROOT, "Shangdong", "test", "boundary", f))) > 127)
        print(f"{f}: bnd_pred max={pb.max():.3f} mean={pb.mean():.4f} >0.5占比={ (pb>0.5).mean()*100:.2f}% "
              f"| mask_pred max={pm.max():.3f} >0.5占比={(pm>0.5).mean()*100:.1f}% "
              f"| GT bnd占比={gt_b.mean()*100:.2f}%")
        # 边界预测降低阈值后与 GT 的匹配
        for thr in [0.05, 0.1, 0.2]:
            pred = (pb > thr)
            inter = (pred & gt_b).sum()
            rec = inter / max(gt_b.sum(), 1)
            prec = inter / max(pred.sum(), 1)
            print(f"    thr={thr}: pred占比={pred.mean()*100:.2f}% recall={rec:.3f} prec={prec:.3f}")
