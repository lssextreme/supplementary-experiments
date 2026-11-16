# -*- coding: utf-8 -*-
"""R3 FLOPs 分项 (thop, MACs x2 口径, 1x3x512x512):
用真实前向捕获各组的输入张量，逐组 thop.profile，最后与整网 thop 总数对账。
"""
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

import config
from harnet_model import build_model

device = "cuda:0" if torch.cuda.is_available() else "cpu"
net = build_model("full", device=device)
net.to(device)
for _m in [getattr(net, a, None) for a in
           ["model", "Adapter4", "Adapter8", "Adapter16", "Adapter32",
            "spafe1", "spafe2", "semfe", "decoder", "sge",
            "conv_final1", "conv_final2", "conv_final3"]]:
    if _m is not None:
        _m.to(device)
net.eval()

x = torch.randn(1, 3, 512, 512, device=device, dtype=torch.float32)

with torch.no_grad():
    # 1) 编码器: FastSAM 内部 torch 模型 (wrapper 的 predict 只取特征)
    flops_enc, _ = profile(net.model.model, inputs=(x,), verbose=False)

    featsA = net.run_encoder(x)
    print("feat pyramid shapes:", [tuple(f.shape) for f in featsA])

    # 2) Adapters
    inp4, inp8 = featsA[3].clone(), featsA[0].clone()
    inp16, inp32 = featsA[1].clone(), featsA[2].clone()
    f4 = net.Adapter4(inp4)
    f8 = net.Adapter8(inp8)
    f16 = net.Adapter16(inp16)
    f32 = net.Adapter32(inp32)
    flops_ad = 0.0
    for m, t in [(net.Adapter4, inp4), (net.Adapter8, inp8),
                 (net.Adapter16, inp16), (net.Adapter32, inp32)]:
        fl, _ = profile(m, inputs=(t,), verbose=False)
        flops_ad += fl

    # 3) SpaFE
    s4 = net.spafe1(f4)
    s8 = net.spafe2(f8)
    fl1, _ = profile(net.spafe1, inputs=(f4,), verbose=False)
    fl2, _ = profile(net.spafe2, inputs=(f8,), verbose=False)
    flops_spafe = fl1 + fl2

    # 4) SemFE (两输入)
    s32 = net.semfe(f16, f32)
    flops_semfe, _ = profile(net.semfe, inputs=(f16, f32), verbose=False)

    # 5) decoder
    enc_list = [s4, s8, f16, s32]
    x9 = net.decoder(enc_list)
    flops_dec, _ = profile(net.decoder, inputs=(enc_list,), verbose=False)

    # 6) SGE + 三头
    xo = net.sge(x9)
    flops_sge, _ = profile(net.sge, inputs=(x9,), verbose=False)
    flops_heads = 0.0
    for h in [net.conv_final1, net.conv_final2, net.conv_final3]:
        fl, _ = profile(h, inputs=(xo,), verbose=False)
        flops_heads += fl

    # 整网对账
    flops_total_net, params_total = profile(net, inputs=(x,), verbose=False)

groups = [
    ("FastSAM encoder", flops_enc),
    ("Adapters (4/8/16/32)", flops_ad),
    ("SpaFE (SE)", flops_spafe),
    ("SemFE (GHF)", flops_semfe),
    ("ARU decoder", flops_dec),
    ("SGE", flops_sge),
    ("conv_final heads (x3)", flops_heads),
]
print(f"=== FLOPs 分项 (thop, MACs x2, 1x3x512x512, 含冻结编码器前向) ===")
s = 0.0
for g, f in groups:
    print(f"{g:28s} {f/1e9:9.3f} GFLOPs")
    s += f
print(f"{'SUM(逐组)':28s} {s/1e9:9.3f} GFLOPs")
print(f"{'SUM(整网 thop)':28s} {flops_total_net/1e9:9.3f} GFLOPs   (论文表4: 93.25)")
print(f"{'thop params':28s} {params_total/1e6:.4f}M")
