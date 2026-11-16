# -*- coding: utf-8 -*-
"""R3 分项测量: 编码器/适配器/增强模块/解码器/头的 参数量 与 FLOPs（512x512 输入）。
参数: 直接按模块分组统计（trainable / frozen）。
FLOPs: torch.utils.flop_counter.FlopCounterMode 按组统计（含冻结编码器前向）。
"""
import os
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(r"D:\学业相关\HXJ大论文\lss_BsiNet", "revise_exp")
_PARENT = os.path.dirname(_REPO)
for _p in (_REPO, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from harnet_model import build_model

device = "cuda:0" if torch.cuda.is_available() else "cpu"
net = build_model("full", device=device)
net.to(device)
net.eval()

GROUPS = [
    ("FastSAM encoder", ["model"]),
    ("Adapters (4/8/16/32)", ["Adapter4", "Adapter8", "Adapter16", "Adapter32"]),
    ("SpaFE (SE)", ["spafe1", "spafe2"]),
    ("SemFE (GHF)", ["semfe"]),
    ("ARU decoder", ["decoder"]),
    ("SGE + conv_final heads", ["sge", "conv_final1", "conv_final2", "conv_final3"]),
]

print("=== 参数量 (512x512, model=full) ===")
tot_train = 0
tot_frozen = 0
for gname, attrs in GROUPS:
    train = frozen = 0
    for a in attrs:
        m = getattr(net, a, None)
        if m is None:
            continue
        # FastSAM wrapper: 真实 torch 模型在 .model 下
        if a == "model":
            m = m.model
        for p in m.parameters():
            if p.requires_grad:
                train += p.numel()
            else:
                frozen += p.numel()
    tot_train += train
    tot_frozen += frozen
    print(f"{gname:28s} trainable={train/1e6:9.4f}M  frozen={frozen/1e6:9.4f}M  total={(train+frozen)/1e6:9.4f}M")
print(f"{'SUM':28s} trainable={tot_train/1e6:9.4f}M  frozen={tot_frozen/1e6:9.4f}M  total={(tot_train+tot_frozen)/1e6:9.4f}M")

print()
print("=== FLOPs 分项 (1x3x512x512, torch.profiler with_flops, 含冻结编码器前向) ===")
import torch.profiler

x = torch.randn(1, 3, 512, 512, device=device, dtype=torch.float32)
with torch.no_grad():
    # warmup
    net(x)
    torch.cuda.synchronize() if device.startswith("cuda") else None
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        with_flops=True,
    ) as prof:
        net(x)

PREFIX = [
    ("FastSAM encoder", ["model.model"]),
    ("Adapters (4/8/16/32)", ["Adapter4", "Adapter8", "Adapter16", "Adapter32"]),
    ("SpaFE (SE)", ["spafe1", "spafe2"]),
    ("SemFE (GHF)", ["semfe"]),
    ("ARU decoder", ["decoder"]),
    ("SGE + heads", ["sge", "conv_final1", "conv_final2", "conv_final3"]),
]

def group_of(name):
    for g, prefs in PREFIX:
        for p in prefs:
            if p in name:
                return g
    return "other"

grouped = {}
total = 0.0
for evt in prof.key_averages():
    g = group_of(evt.key)
    grouped[g] = grouped.get(g, 0.0) + (evt.flops or 0.0)
    total += evt.flops or 0.0

for g, _ in PREFIX:
    v = grouped.get(g, 0.0)
    print(f"{g:28s} {v/1e9:9.3f} GFLOPs")
if grouped.get("other", 0.0) > 0:
    print(f"{'other':28s} {grouped['other']/1e9:9.3f} GFLOPs")
print(f"{'SUM':28s} {total/1e9:9.3f} GFLOPs   (论文表4: 93.25)")
