# -*- coding: utf-8 -*-
"""全量 patch 重叠验证: 对每张 512x512 图的左/右/上/下 256px 条带做哈希,
跨文件匹配。若 stride=256 (50%重叠) 则应有大量条带级精确匹配。"""
import os
import hashlib
import numpy as np
from PIL import Image

DATA = r"D:\data1"

def band_hash(arr):
    return hashlib.md5(arr.tobytes()).hexdigest()[:16]

results = {}
for name in ["Shangdong", "Netherlands", "Sichuan"]:
    for split in ["train", "test"]:
        idir = None
        for tsub in ["image", "img"]:
            cand = os.path.join(DATA, name, split, tsub)
            if os.path.isdir(cand):
                idir = cand
                break
        if idir is None:
            continue
        files = sorted(os.listdir(idir))
        bands = {}  # hash -> list of (file, band)
        for f in files:
            try:
                a = np.array(Image.open(os.path.join(idir, f)))
            except Exception as e:
                print("read fail", f, e); continue
            if a.ndim == 2:
                a = a[..., None]
            h, w = a.shape[:2]
            halfw, halfh = w // 2, h // 2
            for band, arr in [("L", a[:, :halfw]), ("R", a[:, halfw:]),
                              ("T", a[:halfh, :]), ("B", a[halfh:, :])]:
                bands.setdefault(band_hash(arr), []).append((f, band))
        matches = 0
        pair_list = []
        for hsh, lst in bands.items():
            if len(lst) > 1:
                matches += 1
                pair_list.append(lst)
        results[f"{name}/{split}"] = (len(files), matches, pair_list)
        print(f"{name}/{split}: {len(files)} files, {matches} 组条带匹配")
        for lst in pair_list[:8]:
            print("    ", lst)

# 汇总
total_m = sum(r[1] for r in results.values())
print(f"\n总计条带匹配组数: {total_m} / 检查 {sum(r[0] for r in results.values())} 张图")
