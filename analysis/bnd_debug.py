# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image

DATA = r"D:\data1"

def cov(path):
    a = np.array(Image.open(path))
    if a.ndim == 3:
        a = a[..., 0]
    return float((a > 127).mean())

for split in ["train", "test"]:
    bdir = os.path.join(DATA, "Shangdong", split, "boundary")
    mdir = os.path.join(DATA, "Shangdong", split, "mask")
    bf = sorted(os.listdir(bdir))
    bv = [cov(os.path.join(bdir, f)) for f in bf]
    mv = [cov(os.path.join(mdir, f)) for f in sorted(os.listdir(mdir))]
    bv = np.array(bv); mv = np.array(mv)
    print(f"Shangdong/{split}: boundary n={len(bv)} min={bv.min()*100:.2f}% med={np.median(bv)*100:.2f}% max={bv.max()*100:.2f}%")
    print(f"               mask     n={len(mv)} min={mv.min()*100:.2f}% med={np.median(mv)*100:.2f}% max={mv.max()*100:.2f}%")

# 看看 boundary 与 mask 的关系: boundary 是否在 mask 边缘?
import random
bdir = os.path.join(DATA, "Shangdong", "test", "boundary")
mdir = os.path.join(DATA, "Shangdong", "test", "mask")
for f in sorted(os.listdir(bdir))[:3]:
    b = (np.array(Image.open(os.path.join(bdir, f))) > 127)
    m = (np.array(Image.open(os.path.join(mdir, f))) > 127)
    inter = (b & m).sum()
    print(f, "bnd覆盖%", round(b.mean()*100, 2), "mask覆盖%", round(m.mean()*100, 2),
          "bnd∩mask/bnd =", round(inter / max(b.sum(), 1) * 100, 1), "%")
