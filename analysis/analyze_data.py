# -*- coding: utf-8 -*-
"""R5/R6 数据核验（无需训练的小实验）:
1) 各数据集 train/test mask 的农田覆盖率统计（验证 <40% 过滤声称）
2) 相邻 patch 是否 50% 重叠（stride 256 声称）验证
3) 读取 multiseed_summary.json 打印每 seed IoU（供 R1 图）
"""
import json
import os
import numpy as np
from PIL import Image

DATA = r"D:\data1"
OUT = r"D:\a_firstpaper\compare\work_figs"

def coverage_of_tif(path):
    im = Image.open(path)
    a = np.array(im)
    if a.ndim == 3:
        a = a[..., 0]
    a = a.astype(np.float32)
    if a.max() > 1.5:  # 0-255 编码
        a = (a > 127).astype(np.float32)
    else:
        a = (a > 0.5).astype(np.float32)
    return float(a.mean())

print("=== 1) 覆盖率统计 (mask 中农田像素占比) ===")
cover = {}
for name, tsub in [("Shangdong", "image"), ("Netherlands", "img"), ("Sichuan", "image")]:
    for split in ["train", "test"]:
        mdir = os.path.join(DATA, name, split, "mask")
        files = sorted(os.listdir(mdir))
        vals = []
        for f in files:
            vals.append(coverage_of_tif(os.path.join(mdir, f)))
        vals = np.array(vals)
        cover[f"{name}/{split}"] = vals
        below40 = int((vals < 0.40).sum())
        print(f"{name}/{split}: n={len(vals)}  min={vals.min()*100:.1f}%  max={vals.max()*100:.1f}%  "
              f"median={np.median(vals)*100:.1f}%  <40%: {below40}  "
              f"[40-50):{int(((vals>=0.40)&(vals<0.50)).sum())}  [50-70):{int(((vals>=0.50)&(vals<0.70)).sum())}  >=70%:{int((vals>=0.70).sum())}")

print()
print("=== 2) 相邻 patch 重叠验证 (Shandong train, clipped_100..103) ===")
pairs = [("clipped_100.tif", "clipped_101.tif"), ("clipped_101.tif", "clipped_102.tif"),
         ("clipped_102.tif", "clipped_103.tif")]
base = os.path.join(DATA, "Shangdong", "train", "image")
for f1, f2 in pairs:
    a = np.array(Image.open(os.path.join(base, f1)))
    b = np.array(Image.open(os.path.join(base, f2)))
    print(f"{f1} {a.shape} vs {f2} {b.shape}")
    res = {}
    if a.shape[1] >= 256 and b.shape[1] >= 256:
        res["水平: A右半==B左半 (A在左)"] = bool(np.array_equal(a[:, 256:, :], b[:, :256, :]))
        res["水平: A左半==B右半 (B在左)"] = bool(np.array_equal(a[:, :256, :], b[:, 256:, :]))
    if a.shape[0] >= 256 and b.shape[0] >= 256:
        res["垂直: A下半==B上半 (A在上)"] = bool(np.array_equal(a[256:, :, :], b[:256, :, :]))
        res["垂直: A上半==B下半 (B在上)"] = bool(np.array_equal(a[:256, :, :], b[256:, :, :]))
    for k, v in res.items():
        print(f"   {k}: {v}")

print()
print("=== 3) multiseed_summary.json 每 seed IoU ===")
mp = os.path.join(r"D:\学业相关\HXJ大论文\lss_BsiNet\revise_exp\outputs\results", "multiseed_summary.json")
d = json.load(open(mp, encoding="utf-8"))
print(json.dumps(d, indent=1, ensure_ascii=False)[:2500])
