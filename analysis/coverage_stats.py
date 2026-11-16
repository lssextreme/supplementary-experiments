# -*- coding: utf-8 -*-
"""各测试集 mask 覆盖率的最小/中位/最大，供 R5 回复文字引用。"""
import os

import numpy as np
from PIL import Image

DATA = r"D:\data1"
for region, folder in [("Shandong", "Shangdong"), ("Netherlands", "Netherlands"), ("Sichuan", "Sichuan")]:
    d = os.path.join(DATA, folder, "test", "mask")
    vals = []
    for f in sorted(os.listdir(d)):
        if f.startswith("."):
            continue
        a = np.array(Image.open(os.path.join(d, f)))
        if a.ndim == 3:
            a = a[..., 0]
        vals.append(float((a > 127).mean()) * 100.0)
    vals = np.array(vals)
    print(f"{region}: n={len(vals)} min={vals.min():.1f}% median={np.median(vals):.1f}% max={vals.max():.1f}% "
          f"<20%: {(vals < 20).sum()}  <40%: {(vals < 40).sum()}")
