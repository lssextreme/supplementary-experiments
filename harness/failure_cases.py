# -*- coding: utf-8 -*-
"""C6 · 失败样例可视化（最低 IoU 的 K 张图）。

在指定数据集的测试集上推理，按每图 IoU 升序取 top-K，保存
「输入 / GT / 预测 / 叠加(红=GT, 绿=Pred)」四联图，用于论文失败案例分析。
用法：
    python revise_exp/failure_cases.py --dataset sichuan --checkpoint <best.pth> --topk 20
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config
from common import set_seed, metrics_from_preds
from harnet_model import build_model
import evaluate  # 复用其 load_rgb_tensor / load_mask / pad_to_multiple


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sichuan", choices=list(config.DATASETS))
    p.add_argument("--checkpoint", required=True, help=".pth 权重路径")
    p.add_argument("--model", default="full", choices=list(config.MODEL_REGISTRY))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    net = build_model(args.model, device=str(device))
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.to(device)
    net.eval()

    ds = config.DATASETS[args.dataset]
    test_img = ds["test"]["img"]
    test_mask = ds["test"]["mask"]
    files = sorted(f for f in os.listdir(test_img) if not f.startswith("."))

    out_dir = os.path.join(config.OUT_DIR, "failure_cases", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    records = []
    with torch.no_grad():
        for f in files:
            img_path = os.path.join(test_img, f)
            mask_path = os.path.join(test_mask, f)
            if not os.path.exists(mask_path):
                continue
            x = evaluate.load_rgb_tensor(img_path)
            x, H, W = evaluate.pad_to_multiple(x)
            x = x.to(device, dtype=torch.float32)
            outs = net(x)
            prob = torch.sigmoid(outs[1]).squeeze(0).squeeze(0)[:H, :W]
            pred = (prob > config.MASK_THRESHOLD).cpu().numpy().astype(np.uint8)
            gt = evaluate.load_mask(mask_path)
            if pred.shape != gt.shape:
                continue
            m = metrics_from_preds(pred, gt)
            records.append((f, m["IoU"], pred, gt, img_path))

    records.sort(key=lambda t: t[1])  # 升序：最差在前
    top = records[: args.topk]

    for i, (f, iou, pred, gt, img_path) in enumerate(top):
        rgb = np.array(Image.open(img_path).convert("RGB"))
        if rgb.shape[:2] != pred.shape:
            rgb = np.array(Image.open(img_path).convert("RGB")
                           .resize((pred.shape[1], pred.shape[0])))

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(rgb); axes[0].set_title("Input"); axes[0].axis("off")
        axes[1].imshow(gt, cmap="gray"); axes[1].set_title("GT"); axes[1].axis("off")
        axes[2].imshow(pred, cmap="gray"); axes[2].set_title("Pred"); axes[2].axis("off")
        axes[3].imshow(rgb)
        axes[3].contour(gt, colors="red", linewidths=1)
        axes[3].contour(pred, colors="lime", linewidths=1)
        axes[3].set_title(f"IoU={iou:.1f}%"); axes[3].axis("off")
        fig.suptitle(f"{f}  (IoU {iou:.2f}%)")

        out_png = os.path.join(out_dir, f"{i:02d}_{os.path.splitext(f)[0]}.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[{i+1}/{len(top)}] IoU={iou:.2f}% -> {out_png}")

    print(f"\n[failure_cases] 已保存 {len(top)} 张最差样本到 {out_dir}")


if __name__ == "__main__":
    main()
