# -*- coding: utf-8 -*-
"""C3 · 损失权重敏感性实验。

在单一数据集（默认 shandong）上，对 config.WEIGHT_RATIOS 里的每种
(mask,boundary,distance) 权重训练并评估，输出 IoU 对比表。
输出：outputs/results/sensitivity_summary.json
"""
import argparse
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import config


def run(cmd, cwd):
    print("\n>>> " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        print(f"[FAIL] {' '.join(cmd)}")
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="shandong", choices=list(config.DATASETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[config.SEEDS[0]],
                    help="敏感性只反映趋势，默认单种子以省算力；可传多个种子求均值")
    ap.add_argument("--model", default="full", choices=list(config.MODEL_REGISTRY))
    args = ap.parse_args()

    train_py = os.path.join(_HERE, "train.py")
    eval_py = os.path.join(_HERE, "evaluate.py")

    summary = {}
    for wname, (wm, wb, wd) in config.WEIGHT_RATIOS.items():
        wstr = f"{wm},{wb},{wd}"
        for seed in args.seeds:
            ckpt_dir = os.path.join(config.CKPT_DIR,
                                    f"{args.model}_{args.dataset}_seed{seed}_w{wstr.replace(',', '_')}")
            ckpt = os.path.join(ckpt_dir, "best.pth")
            last_pth = os.path.join(ckpt_dir, "last.pth")
            if not os.path.exists(last_pth):
                run([sys.executable, train_py, "--model", args.model, "--dataset", args.dataset,
                     "--seed", str(seed), "--weights", wstr], _PARENT)
            else:
                print(f"[skip train] {last_pth} 已存在（已完成）")

            etag = f"{args.model}_{args.dataset}_seed{seed}_w{wstr.replace(',', '_')}"
            res_path = os.path.join(config.RESULT_DIR, etag + ".json")
            if not os.path.exists(res_path):
                run([sys.executable, eval_py, "--model", args.model, "--dataset", args.dataset,
                     "--checkpoint", ckpt, "--seed", str(seed), "--tag", etag], _PARENT)
            with open(res_path) as f:
                r = json.load(f)
            summary.setdefault(wname, {})[seed] = r["global"]

    out = os.path.join(config.RESULT_DIR, "sensitivity_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== 损失权重敏感性（IoU %） =====")
    print(f"{'weights':12s}  {'IoU':>8s}  {'F1':>8s}  {'Acc':>8s}")
    for wname in summary:
        ious = [summary[wname][s]["IoU"] for s in summary[wname]]
        f1s = [summary[wname][s]["F1"] for s in summary[wname]]
        accs = [summary[wname][s]["Acc"] for s in summary[wname]]
        m_i = sum(ious) / len(ious)
        m_f = sum(f1s) / len(f1s)
        m_a = sum(accs) / len(accs)
        print(f"{wname:12s}  {m_i:8.2f}  {m_f:8.2f}  {m_a:8.2f}")
    print(f"\n[run_sensitivity] 汇总 -> {out}")


if __name__ == "__main__":
    main()
