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

print("=== boundary vs mask coverage (test) ===")
for name in ["Shangdong", "Netherlands", "Sichuan"]:
    bdir = os.path.join(DATA, name, "test", "boundary")
    mdir = os.path.join(DATA, name, "test", "mask")
    if not os.path.isdir(bdir):
        print(name, "no boundary dir"); continue
    bf = sorted(os.listdir(bdir))
    mf = sorted(os.listdir(mdir))
    bv = np.array([cov(os.path.join(bdir, f)) for f in bf])
    mv = np.array([cov(os.path.join(mdir, f)) for f in mf])
    print(f"{name}: boundary n={len(bv)} min={bv.min()*100:.1f}% med={np.median(bv)*100:.1f}% <40%:{(bv<0.4).sum()}")
    print(f"        mask     n={len(mv)} min={mv.min()*100:.1f}% med={np.median(mv)*100:.1f}% <40%:{(mv<0.4).sum()}")
