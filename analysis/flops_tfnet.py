# -*- coding: utf-8 -*-
"""TFNet 的 FLOPs/参数量测量（与 HARNet 同口径: thop, MACs x2, 1x3x512x512, imgsz=512）。"""
import os
import sys

import torch
from thop import profile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(r"D:\学业相关\HXJ大论文\lss_BsiNet", "revise_exp")
_PARENT = os.path.dirname(_REPO)
for _p in (_REPO, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.TFNet import TFNet

device = "cuda:0" if torch.cuda.is_available() else "cpu"
net = TFNet(model_name=os.path.join(_PARENT, "FastSAM-x.pt"),
            device=str(device), conf=0.4, iou=0.9, imgsz=512,
            retina_masks=True, done_warmup=True)
net.to(device)
net.eval()

x = torch.randn(1, 3, 512, 512, device=device, dtype=torch.float32)
with torch.no_grad():
    out = net(x)
    if isinstance(out, (list, tuple)):
        print("outputs:", [tuple(o.shape) for o in out])
    else:
        print("output:", out.shape)

    flops_enc, _ = profile(net.model.model, inputs=(x,), verbose=False)
    print(f"FastSAM encoder: {flops_enc/1e9:.3f} GFLOPs")

    flops_net, params = profile(net, inputs=(x,), verbose=False)
    print(f"whole-net thop:  {flops_net/1e9:.3f} GFLOPs, params {params/1e6:.4f}M")

    train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"trainable: {train/1e6:.4f}M")
