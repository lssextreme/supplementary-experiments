# -*- coding: utf-8 -*-
"""C1 · 多种子汇总：mean±std + 配对 t 检验 + Wilcoxon 符号秩检验 + LaTeX 片段。

输入：run_multiseed.py 生成的 outputs/results/multiseed_summary.json
结构：{model: {dataset: {seed: {"Acc","F1","Recall","Prec","IoU"}}}}
可选 --baselines <json>：基线多种子结果，用于 HARNet(full) vs 基线的配对显著性检验。

显著性检验口径（论文返修第二轮，6 个种子 42/123/2024/7/100/2025）：
  - 配对 t 检验（双尾，df=n-1）；
  - Wilcoxon 符号秩检验（精确枚举，n=6 时最小可达双尾 p=0.03125）；
  两者均不依赖 scipy（scipy 仅作为 t 检验的加速路径，未安装时自动回退到内置实现）。
"""
import argparse
import json
import math
import os
import sys
from itertools import product
from math import exp, lgamma, log

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import config


def mean_std(vals):
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


# ---- 内置 t 分布（无需 scipy）：正则化不完全 beta -> t CDF ----
def _betacf(a, b, x):
    MAXIT = 200
    EPS = 3e-12
    FPMIN = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        del_ = d * c
        h *= del_
        if abs(del_ - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_sf(t, df):
    """双尾生存函数 P(|T|>t) 的 1/2（即单尾），t>=0。"""
    if t <= 0:
        return 0.5
    x = df / (df + t * t)
    return 0.5 * _betai(df / 2.0, 0.5, x)


def t_pvalue(t, df):
    """双尾 t 检验 p 值（优先 scipy，缺失则内置实现）。"""
    t = abs(t)
    try:
        from scipy.stats import t as tdist
        return float(2 * tdist.sf(t, df))
    except Exception:
        return float(2 * _t_sf(t, df))


def paired_ttest(a, b):
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 2:
        return None
    d = np.array(a) - np.array(b)
    m = d.mean()
    s = d.std(ddof=1)
    if s == 0:
        return float("nan"), float("nan"), n
    t = m / (s / np.sqrt(n))
    p = t_pvalue(t, n - 1)
    return float(t), float(p), n


def wilcoxon(a, b):
    """Wilcoxon 符号秩检验（精确枚举 2^n 符号分配），返回 (统计量 W, 双尾 p, n)。

    n=6 时最小可达双尾 p=2/2^6=0.03125，故 6/6 全胜出时 p 恒为 0.03125。
    """
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    n = min(len(a), len(b))
    d = [a[i] - b[i] for i in range(n) if abs(a[i] - b[i]) > 1e-9]
    n = len(d)
    if n == 0:
        return 0.0, float("nan"), n
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs(d[order[j]]) == abs(d[order[i]]):
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    W = sum(ranks[i] for i in range(n) if d[i] > 0)
    mean = n * (n + 1) / 4.0
    ext = abs(W - mean)
    cnt = 0
    for signs in product([-1, 1], repeat=n):
        s = sum(ranks[i] for i in range(n) if signs[i] > 0)
        if abs(s - mean) >= ext - 1e-9:
            cnt += 1
    return float(W), float(cnt / 2**n), n


def _align(summary, other, ds):
    common = sorted(set(summary["full"][ds]) & set(other[ds]))
    return common


def _report_pair(container, a, b, tag):
    """对按 seed 对齐的两组 IoU 序列做配对 t + Wilcoxon，并打印。"""
    r = paired_ttest(a, b)
    if r is None:
        print(f"  {tag}: 种子不足")
        return
    t, p, n = r
    w, wp, wn = wilcoxon(a, b)
    pstr = f"{p:.4f}" if p >= 0.001 else "<0.001"
    wpstr = f"{wp:.4f}" if wp >= 0.001 else "<0.001"
    print(f"  {tag}: t={t:6.3f} p={pstr} | Wilcoxon p={wpstr} (n={n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=os.path.join(config.RESULT_DIR, "multiseed_summary.json"))
    ap.add_argument("--baselines", default=None, help="基线多种子结果 json")
    ap.add_argument("--metric", default="IoU", choices=["Acc", "F1", "Recall", "Prec", "IoU"])
    args = ap.parse_args()

    with open(args.summary) as f:
        summary = json.load(f)

    print(f"===== mean±std（metric={args.metric}，单位 %） =====")
    table = {}
    for model in summary:
        for ds in summary[model]:
            seeds = summary[model][ds]
            vals = [seeds[s][args.metric] for s in seeds if args.metric in seeds[s]]
            m, s = mean_std(vals)
            table.setdefault(model, {})[ds] = (m, s)
            print(f"  {model:18s} {ds:12s}  {m:6.2f} ± {s:5.2f}  (n={len(vals)})")

    # ---- 配对 t 检验 + Wilcoxon（full vs 其它，按数据集）----
    if "full" in summary:
        print("\n===== 显著性检验（full vs 其它，IoU：配对 t + Wilcoxon） =====")
        for model in summary:
            if model == "full":
                continue
            for ds in summary["full"]:
                if ds not in summary.get(model, {}):
                    continue
                common = _align(summary, summary[model], ds)
                if len(common) < 2:
                    continue
                a = [summary["full"][ds][s]["IoU"] for s in common]
                b = [summary[model][ds][s]["IoU"] for s in common]
                _report_pair(None, a, b, f"full vs {model:16s} @ {ds:12s}")

    # ---- 基线对比（若提供）----
    if args.baselines and os.path.exists(args.baselines):
        with open(args.baselines) as f:
            base = json.load(f)
        print("\n===== full vs 基线（显著性检验，IoU：配对 t + Wilcoxon） =====")
        for bname in base:
            for ds in summary.get("full", {}):
                if ds not in base.get(bname, {}):
                    continue
                common = sorted(set(summary["full"][ds]) & set(base[bname][ds]))
                if len(common) < 2:
                    continue
                a = [summary["full"][ds][s]["IoU"] for s in common]
                b = [base[bname][ds][s]["IoU"] for s in common]
                _report_pair(None, a, b, f"full vs {bname:16s} @ {ds:12s}")

    # ---- LaTeX 片段 ----
    print("\n===== LaTeX 片段（mean±std） =====")
    datasets_order = [d for d in config.DATASETS if d in table.get("full", {})]
    for model in table:
        row = " & ".join(
            f"{table[model][ds][0]:.2f}$\\pm${table[model][ds][1]:.2f}"
            if ds in table[model] else "--"
            for ds in datasets_order
        )
        print(f"  {model} & {row} \\\\")


if __name__ == "__main__":
    main()
