# -*- coding: utf-8 -*-
"""C1 · 多种子汇总：mean±std + 配对 t 检验 + LaTeX 片段。

输入：run_multiseed.py 生成的 outputs/results/multiseed_summary.json
结构：{model: {dataset: {seed: {"Acc","F1","Recall","Prec","IoU"}}}}
可选 --baselines <json>：基线多种子结果，用于 HARNet(full) vs 基线的配对 t 检验。
"""
import argparse
import json
import os
import sys

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
    try:
        from scipy.stats import t as tdist
        p = 2 * tdist.sf(abs(t), n - 1)
    except Exception:
        p = float("nan")
    return float(t), float(p), n


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

    # ---- 配对 t 检验（full vs 其它，按数据集）----
    if "full" in summary:
        print("\n===== 配对 t 检验（full vs 其它，IoU） =====")
        for model in summary:
            if model == "full":
                continue
            for ds in summary["full"]:
                if ds not in summary.get(model, {}):
                    continue
                a = [summary["full"][ds][s]["IoU"] for s in summary["full"][ds]]
                b = [summary[model][ds][s]["IoU"] for s in summary[model][ds]]
                # 按 seed 对齐
                common = sorted(set(summary["full"][ds]) & set(summary[model][ds]))
                a = [summary["full"][ds][s]["IoU"] for s in common]
                b = [summary[model][ds][s]["IoU"] for s in common]
                r = paired_ttest(a, b)
                if r is None:
                    print(f"  full vs {model} @ {ds}: 种子不足")
                else:
                    t, p, n = r
                    print(f"  full vs {model:16s} @ {ds:12s}: t={t:6.3f} p={p:.4f} (n={n})")

    # ---- 基线对比（若提供）----
    if args.baselines and os.path.exists(args.baselines):
        with open(args.baselines) as f:
            base = json.load(f)
        print("\n===== full vs 基线（配对 t 检验，IoU） =====")
        for bname in base:
            for ds in summary.get("full", {}):
                if ds not in base.get(bname, {}):
                    continue
                common = sorted(set(summary["full"][ds]) & set(base[bname][ds]))
                if len(common) < 2:
                    continue
                a = [summary["full"][ds][s]["IoU"] for s in common]
                b = [base[bname][ds][s]["IoU"] for s in common]
                t, p, n = paired_ttest(a, b)
                print(f"  full vs {bname:16s} @ {ds:12s}: t={t:6.3f} p={p:.4f} (n={n})")

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
