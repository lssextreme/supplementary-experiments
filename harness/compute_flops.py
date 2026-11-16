# -*- coding: utf-8 -*-
"""C4 · 参数量 / FLOPs 口径统计。

回答审稿意见 R3：trainable vs total 参数量、是否包含冻结 FastSAM-x 前向。
- trainable params：net.parameters() 里 requires_grad=True 的部分（适配器+增强模块+解码器+头）。
- frozen encoder params：FastSAM-x 主干（被冻结）。
- total params = trainable + frozen encoder。
FLOPs 用 thop 尽量统计（若没装 thop 则跳过并提示）。
"""
import os
import sys

import torch

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_PARENT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from harnet_model import build_model


def params_summary(net):
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    registered = sum(p.numel() for p in net.parameters())
    frozen_encoder = 0
    if hasattr(net, "model") and hasattr(net.model, "model"):
        frozen_encoder = sum(p.numel() for p in net.model.model.parameters())
    return {
        "trainable": trainable,
        "registered_total": registered,
        "frozen_encoder": frozen_encoder,
        "total": registered + frozen_encoder,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="full", choices=list(config.MODEL_REGISTRY))
    p.add_argument("--size", type=int, default=512, help="输入分辨率，用于 FLOPs")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    net = build_model(args.model, device=args.device)
    net.eval()

    ps = params_summary(net)
    print("=" * 60)
    print(f"参数量统计（{args.model}）")
    print("=" * 60)
    print(f"  trainable (适配器+增强模块+解码器+头): {ps['trainable']/1e6:.3f} M")
    print(f"  frozen FastSAM-x 主干:               {ps['frozen_encoder']/1e6:.3f} M")
    print(f"  total (trainable + frozen):          {ps['total']/1e6:.3f} M")
    print(f"  （registered_total = net.parameters() = {ps['registered_total']/1e6:.3f} M，"
          f"不含 FastSAM-x，因其是普通属性而非 nn.Module 子模块）")

    # ---- FLOPs（thop 尽力而为）----
    try:
        from thop import profile, clever_format
        dummy = torch.randn(1, 3, args.size, args.size).to(args.device)
        net.to(args.device)
        try:
            macs, params = profile(net, inputs=(dummy,), verbose=False)
            flops = 2 * macs
            print("-" * 60)
            print(f"FLOPs（thop 对 net.forward 统计，512x512）")
            print(f"  MACs:  {clever_format([macs], '%.3f')}")
            print(f"  FLOPs: {clever_format([flops], '%.3f')} ({flops/1e9:.2f} GFLOPs)")
            print("  说明：thop 只统计 net 注册的 nn.Module（解码器+适配器+头）。"
                  "冻结的 FastSAM-x 主干若不是注册子模块则未计入；"
                  "论文 93.25 GFLOPs 若为含主干前向的口径，需另加主干 FLOPs。")
        except Exception as e:
            print(f"[warn] thop.profile 失败：{e}")
    except ImportError:
        print("-" * 60)
        print("[warn] 未安装 thop，跳过 FLOPs 统计。")
        print("       安装：conda run -n rs-torch pip install thop")


if __name__ == "__main__":
    main()
