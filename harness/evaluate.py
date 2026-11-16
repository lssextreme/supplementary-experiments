# -*- coding: utf-8 -*-
"""HARNet 测试集推理 + 5 指标评估（Acc / F1 / Recall / Prec / IoU）。

用法：
    python revise_exp/evaluate.py --dataset shandong --model full --checkpoint <best.pth>
输出：
    outputs/results/<tag>.json  （全局像素级指标 + 每图 IoU/F1 的 mean±std）
    可选 --save-preds 时保存二值掩码到 outputs/predictions/<tag>/
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_PARENT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from common import set_seed, accumulate_confusion, metrics_from_confusion
from harnet_model import build_model


def load_rgb_tensor(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None]            # 1,C,H,W
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="full", choices=list(config.MODEL_REGISTRY))
    p.add_argument("--dataset", required=True, choices=list(config.DATASETS))
    p.add_argument("--checkpoint", required=True, help=".pth 权重路径")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save-preds", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tag = args.tag or f"{args.model}_{args.dataset}"

    net = build_model(args.model, device=str(device))
    state = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(state)
    net.to(device)
    net.eval()

    ds = config.DATASETS[args.dataset]
    test_img = ds["test"]["img"]
    test_mask = ds["test"]["mask"]
    files = sorted(f for f in os.listdir(test_img) if not f.startswith("."))

    pred_dir = None
    if args.save_preds:
        pred_dir = os.path.join(config.PRED_DIR, tag)
        os.makedirs(pred_dir, exist_ok=True)

    tp = fp = fn = tn = 0
    per_img = []
    with torch.no_grad():
        for f in tqdm(files, desc=f"eval {args.dataset}"):
            img_path = os.path.join(test_img, f)
            mask_path = os.path.join(test_mask, f)
            if not os.path.exists(mask_path):
                print(f"[skip] no mask for {f}")
                continue

            x = load_rgb_tensor(img_path)
            x, H, W = pad_to_multiple(x)
            x = x.to(device, dtype=torch.float32)
            outs = net(x)
            pred_mask = outs[1]                       # mask 头（三任务顺序里 index=1）
            prob = torch.sigmoid(pred_mask).squeeze(0).squeeze(0)[:H, :W]
            pred = (prob > config.MASK_THRESHOLD).cpu().numpy().astype(np.uint8)

            gt = load_mask(mask_path)
            if pred.shape != gt.shape:
                print(f"[skip] shape mismatch {f}: {pred.shape} vs {gt.shape}")
                continue

            a, b, c, d = accumulate_confusion(pred, gt)
            tp += a; fp += b; fn += c; tn += d
            m = metrics_from_confusion(a, b, c, d)
            per_img.append({"file": f, "IoU": m["IoU"], "F1": m["F1"]})

            if pred_dir is not None:
                Image.fromarray((pred * 255).astype(np.uint8)).save(
                    os.path.join(pred_dir, os.path.splitext(f)[0] + ".png"))

    global_metrics = metrics_from_confusion(tp, fp, fn, tn)
    ious = [x["IoU"] for x in per_img]
    f1s = [x["F1"] for x in per_img]
    result = {
        "model": args.model,
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "n_images": len(per_img),
        "global": global_metrics,
        "per_image_mean_std": {
            "IoU": [float(np.mean(ious)) if ious else 0.0,
                    float(np.std(ious)) if ious else 0.0],
            "F1":  [float(np.mean(f1s)) if f1s else 0.0,
                    float(np.std(f1s)) if f1s else 0.0],
        },
    }
    out_path = os.path.join(config.RESULT_DIR, tag + ".json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n===== 全局像素级指标 (%) =====")
    for k in ["Acc", "F1", "Recall", "Prec", "IoU"]:
        print(f"  {k:8s}: {global_metrics[k]:.2f}")
    print(f"  每图 IoU mean±std: {np.mean(ious):.2f} ± {np.std(ious):.2f}")
    print(f"[eval] result -> {out_path}")


if __name__ == "__main__":
    main()
