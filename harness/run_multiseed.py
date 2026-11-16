# -*- coding: utf-8 -*-
"""C1 · 多种子训练+评估（HARNet full 及其消融变体），汇总 mean±std。

用法（在 rs-torch 环境）：
    python revise_exp/run_multiseed.py --models full no_semfe no_spafe no_eca no_aru --datasets shandong netherlands sichuan --seeds 42 123 2024
输出：
    outputs/results/multiseed_summary.json
        {model: {dataset: {seed: {"Acc","F1","Recall","Prec","IoU"}}}}
随后用 stats.py 计算 mean±std 与配对 t 检验。
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
    ap.add_argument("--models", nargs="+", default=["full"],
                    choices=list(config.MODEL_REGISTRY))
    ap.add_argument("--datasets", nargs="+", default=list(config.DATASETS),
                    choices=list(config.DATASETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    ap.add_argument("--weights", default="1,1,1", help="mask,boundary,distance（C1 用论文基准 1,1,1）")
    args = ap.parse_args()

    train_py = os.path.join(_HERE, "train.py")
    eval_py = os.path.join(_HERE, "evaluate.py")

    summary = {}
    for model in args.models:
        for ds in args.datasets:
            for seed in args.seeds:
                ckpt_dir = os.path.join(config.CKPT_DIR,
                                        f"{model}_{ds}_seed{seed}_w{args.weights.replace(',', '_')}")
                ckpt = os.path.join(ckpt_dir, "best.pth")
                last_pth = os.path.join(ckpt_dir, "last.pth")
                if not os.path.exists(last_pth):
                    run([sys.executable, train_py, "--model", model, "--dataset", ds,
                         "--seed", str(seed), "--weights", args.weights], _PARENT)
                else:
                    print(f"[skip train] {last_pth} 已存在（已完成）")

                etag = f"{model}_{ds}_seed{seed}"
                res_path = os.path.join(config.RESULT_DIR, etag + ".json")
                if not os.path.exists(res_path):
                    run([sys.executable, eval_py, "--model", model, "--dataset", ds,
                         "--checkpoint", ckpt, "--seed", str(seed), "--tag", etag], _PARENT)
                with open(res_path) as f:
                    r = json.load(f)
                summary.setdefault(model, {}).setdefault(ds, {})[seed] = r["global"]

    out = os.path.join(config.RESULT_DIR, "multiseed_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[run_multiseed] 汇总 -> {out}")
    print("  下一步：python revise_exp/stats.py 计算 mean±std 与配对 t 检验")


if __name__ == "__main__":
    main()
