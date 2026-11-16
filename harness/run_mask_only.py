# -*- coding: utf-8 -*-
"""C2 · mask-only 单任务消融（去掉 boundary/distance 辅助头，只训 mask）。

在全部数据集上多种子训练并评估，与三任务 full 对比，说明多任务辅助头的收益。
输出：outputs/results/maskonly_summary.json
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
    ap.add_argument("--datasets", nargs="+", default=list(config.DATASETS),
                    choices=list(config.DATASETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    ap.add_argument("--model", default="full", choices=list(config.MODEL_REGISTRY))
    args = ap.parse_args()

    train_py = os.path.join(_HERE, "train.py")
    eval_py = os.path.join(_HERE, "evaluate.py")

    summary = {}
    for ds in args.datasets:
        for seed in args.seeds:
            ckpt_dir = os.path.join(config.CKPT_DIR,
                                    f"{args.model}_{ds}_seed{seed}_maskonly")
            ckpt = os.path.join(ckpt_dir, "best.pth")
            last_pth = os.path.join(ckpt_dir, "last.pth")
            if not os.path.exists(last_pth):
                run([sys.executable, train_py, "--model", args.model, "--dataset", ds,
                     "--seed", str(seed), "--mask-only"], _PARENT)
            else:
                print(f"[skip train] {last_pth} 已存在（已完成）")

            etag = f"{args.model}_{ds}_seed{seed}_maskonly"
            res_path = os.path.join(config.RESULT_DIR, etag + ".json")
            if not os.path.exists(res_path):
                run([sys.executable, eval_py, "--model", args.model, "--dataset", ds,
                     "--checkpoint", ckpt, "--seed", str(seed), "--tag", etag], _PARENT)
            with open(res_path) as f:
                r = json.load(f)
            summary.setdefault(ds, {})[seed] = r["global"]

    out = os.path.join(config.RESULT_DIR, "maskonly_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[run_mask_only] 汇总 -> {out}")


if __name__ == "__main__":
    main()
