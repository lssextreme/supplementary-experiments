# -*- coding: utf-8 -*-
"""HARNet 统一训练脚本（三任务；支持多种子 / 权重 / 消融 / mask-only）。

用法示例（在 rs-torch 环境下，工作目录 = lss_BsiNet）：
    python revise_exp/train.py --dataset shandong --model full --seed 42 --weights 1,1,1
    python revise_exp/train.py --dataset shandong --model full --seed 42 --mask-only
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # lss_BsiNet
_HERE = os.path.dirname(os.path.abspath(__file__))                      # revise_exp
for _p in (_PARENT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from common import set_seed
from harnet_model import build_model
from dataset import BasicDataset
from loss_yb import MultiTaskLoss, BCEDiceLoss


class MaskOnlyLoss(nn.Module):
    """mask-only 单任务消融（C2）：只用 mask 的 BCE+Dice。"""
    def __init__(self):
        super().__init__()
        self.criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=1)

    def forward(self, pred_mask, gt_mask):
        return {"total_loss": self.criterion(pred_mask, gt_mask)}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="full", choices=list(config.MODEL_REGISTRY))
    p.add_argument("--dataset", required=True, choices=list(config.DATASETS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weights", default="1,1,1", help="mask,boundary,distance 权重（C3）")
    p.add_argument("--mask-only", action="store_true", help="只训练 mask 头（C2）")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--tag", default=None, help="输出子目录名；默认 model_dataset_seed")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def parse_weights(s):
    parts = [float(x) for x in s.split(",")]
    assert len(parts) == 3, "--weights 需要 mask,boundary,distance 三个值"
    return parts[0], parts[1], parts[2]


def main():
    args = get_args()
    set_seed(args.seed)

    ds = config.DATASETS[args.dataset]
    tag = args.tag or f"{args.model}_{args.dataset}_seed{args.seed}"
    if args.mask_only:
        tag += "_maskonly"
    else:
        tag += f"_w{args.weights.replace(',', '_')}"
    out_dir = os.path.join(config.CKPT_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[train] model={args.model} dataset={args.dataset} seed={args.seed} "
          f"weights={args.weights} mask_only={args.mask_only} epochs={args.epochs}")

    # ---- 数据 ----
    dataset = BasicDataset(ds["train"]["img"], ds["train"]["boundary"],
                           ds["train"]["mask"], ds["train"]["distance"])
    n_val = int(len(dataset) * config.VAL_PERCENT)
    n_train = len(dataset) - n_val
    gen = torch.Generator().manual_seed(args.seed)
    train, val = random_split(dataset, [n_train, n_val], generator=gen)
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batch, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)
    print(f"[train] total={len(dataset)} train={n_train} val={n_val}")

    # ---- 模型 ----
    net = build_model(args.model, device=str(device))
    net.to(device)

    # ---- 损失 ----
    if args.mask_only:
        criterion = MaskOnlyLoss().to(device)
    else:
        wm, wb, wd = parse_weights(args.weights)
        criterion = MultiTaskLoss(alpha=wm, beta=wb, gama=wd).to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)

    history = {"train_loss": [], "val_loss": [], "args": vars(args)}
    best_val = float("inf")

    # ---- 断点续训：上次被打断（有 resume.pth 且无 last.pth）则从断点继续，不重头训 ----
    resume_pth = os.path.join(out_dir, "resume.pth")
    last_pth = os.path.join(out_dir, "last.pth")
    start_epoch = 0
    if os.path.exists(resume_pth) and not os.path.exists(last_pth):
        ck = torch.load(resume_pth, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"]
        best_val = ck["best_val"]
        history = ck["history"]
        print(f"[train] resume from epoch {start_epoch} (best_val={best_val:.4f})")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        net.train()
        tot = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            imgs = batch["image"].to(device, dtype=torch.float32)
            true_boundary = batch["boundary"].to(device, dtype=torch.float32)
            true_mask = batch["mask"].to(device, dtype=torch.float32)
            true_distance = batch["distance"].to(device, dtype=torch.float32)

            optimizer.zero_grad()
            outs = net(imgs)                      # [boundary, mask, distance]
            pred_boundary, pred_mask, pred_distance = outs

            if args.mask_only:
                loss = criterion(pred_mask, true_mask)
            else:
                loss = criterion(pred_boundary, pred_mask, pred_distance,
                                 true_boundary, true_mask, true_distance)
            loss_train = loss["total_loss"]
            loss_train.backward()
            optimizer.step()

            tot += loss_train.item()
            pbar.set_postfix(loss=f"{loss_train.item():.4f}")

        train_loss = tot / max(len(train_loader), 1)

        # ---- 验证 ----
        net.eval()
        vtot = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device, dtype=torch.float32)
                true_boundary = batch["boundary"].to(device, dtype=torch.float32)
                true_mask = batch["mask"].to(device, dtype=torch.float32)
                true_distance = batch["distance"].to(device, dtype=torch.float32)
                outs = net(imgs)
                pred_boundary, pred_mask, pred_distance = outs
                if args.mask_only:
                    loss = criterion(pred_mask, true_mask)
                else:
                    loss = criterion(pred_boundary, pred_mask, pred_distance,
                                     true_boundary, true_mask, true_distance)
                vtot += loss["total_loss"].item()
        val_loss = vtot / max(len(val_loader), 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"[train] epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(net.state_dict(), os.path.join(out_dir, "best.pth"))

        if epoch % 50 == 0 or epoch == args.epochs:
            torch.save(net.state_dict(), os.path.join(out_dir, f"epoch{epoch}.pth"))

        # 断点：保存完整训练状态（模型+优化器+epoch+best_val+history），中断后可续训
        torch.save({
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "history": history,
        }, resume_pth)

    torch.save(net.state_dict(), os.path.join(out_dir, "last.pth"))
    if os.path.exists(resume_pth):
        os.remove(resume_pth)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] done. outputs -> {out_dir}")


if __name__ == "__main__":
    main()
