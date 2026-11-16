# -*- coding: utf-8 -*-
"""R5 覆盖率分组评估（无需训练的小实验）:
固定已有 seed42 checkpoint，在三个数据集的测试集上按农田覆盖率分组统计指标：
全部 / >=40% / [20,40) / <20%（近空背景），直接检验 <40% 过滤的选择偏差问题。
"""
import argparse
import json
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

CHECKPOINTS = {
    "shandong": os.path.join(config.CKPT_DIR, "full_shandong_seed42_w1_1_1", "best.pth"),
    "netherlands": os.path.join(config.CKPT_DIR, "full_netherlands_seed42_w1_1_1", "best.pth"),
    "sichuan": os.path.join(config.CKPT_DIR, "full_sichuan_seed42_w1_1_1", "best.pth"),
}


def load_rgb_tensor(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None]
    return torch.from_numpy(arr)


def load_mask(path):
    img = Image.open(path).convert("L")
    return (np.array(img) > 0).astype(np.uint8)


def pad_to_multiple(x, mult=32):
    _, _, h, w = x.shape
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph))
    return x, h, w


def coverage_of(gt):
    return float(gt.mean())


def band_of(cov):
    if cov < 0.20:
        return "<20%"
    if cov < 0.40:
        return "20-40%"
    return ">=40%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=os.path.join(r"D:\a_firstpaper\compare\work_figs", "band_split_results.json"))
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = {}
    for dataset in ["shandong", "netherlands", "sichuan"]:
        set_seed(42)
        ckpt = CHECKPOINTS[dataset]
        if not os.path.exists(ckpt):
            alt = os.path.join(config.CKPT_DIR, f"full_{dataset}_seed42_w1_1_1", "best.pth")
            ckpt = alt if os.path.exists(alt) else ckpt
        print(f"[{dataset}] checkpoint: {ckpt}")
        net = build_model("full", device=str(device))
        state = torch.load(ckpt, map_location=device)
        net.load_state_dict(state)
        net.to(device)
        net.eval()

        ds = config.DATASETS[dataset]
        test_img = ds["test"]["img"]
        test_mask = ds["test"]["mask"]
        files = sorted(f for f in os.listdir(test_img) if not f.startswith("."))

        bands = {"all": [0, 0, 0, 0], ">=40%": [0, 0, 0, 0],
                 "20-40%": [0, 0, 0, 0], "<20%": [0, 0, 0, 0]}
        per_img = []
        with torch.no_grad():
            for f in files:
                img_path = os.path.join(test_img, f)
                mask_path = os.path.join(test_mask, f)
                if not os.path.exists(mask_path):
                    continue
                x = load_rgb_tensor(img_path)
                x, H, W = pad_to_multiple(x)
                x = x.to(device, dtype=torch.float32)
                outs = net(x)
                prob = torch.sigmoid(outs[1]).squeeze(0).squeeze(0)[:H, :W]
                pred = (prob > 0.5).cpu().numpy().astype(np.uint8)
                gt = load_mask(mask_path)
                cov = coverage_of(gt)
                a, b, c, d = accumulate_confusion(pred, gt)
                for key in ["all", band_of(cov)]:
                    bands[key][0] += a; bands[key][1] += b; bands[key][2] += c; bands[key][3] += d
                per_img.append({"file": f, "coverage": round(cov, 4), "IoU": None})
                m = metrics_from_confusion(a, b, c, d)
                per_img[-1]["IoU"] = m["IoU"]

        res = {}
        for key, (a, b, c, d) in bands.items():
            res[key] = {"n": None, "metrics": metrics_from_confusion(a, b, c, d)}
        # 统计各分组的样本数
        for key in [">=40%", "20-40%", "<20%"]:
            res[key]["n"] = sum(1 for x in per_img if band_of(x["coverage"]) == key)
        res["all"]["n"] = len(per_img)
        out[dataset] = res
        print(json.dumps(res, ensure_ascii=False, indent=1))

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
