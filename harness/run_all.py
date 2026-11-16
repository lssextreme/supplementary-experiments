# -*- coding: utf-8 -*-
"""HARNet 返修实验一键脚本：按 C1→C7 顺序跑完所有需要补的实验。

用法（在 rs-torch 环境，工作目录 = lss_BsiNet）：
    python revise_exp/run_all.py              # 跑全部（已完成的步骤自动跳过）
    python revise_exp/run_all.py --skip c3 c5 # 跳过某些步骤
    python revise_exp/run_all.py --only c1 c4 # 只跑某些步骤
    python revise_exp/run_all.py --keep-going # 单步失败后继续后续步骤（默认失败即停）

对应审稿意见（见 revise_exp/README.md 与 a_firstpaper 的 HARNet_修改方案.md）：
    C1  R1  多种子(≥3) + 配对 t 检验          → run_multiseed.py + stats.py
    C2  R2  mask-only 单任务消融              → run_mask_only.py
    C3  R2  损失权重敏感性                     → run_sensitivity.py
    C4  R3  参数量/FLOPs 口径                 → compute_flops.py
    C5  R5  未过滤(<40% 农田)测试集评估        → 需先配置未过滤数据集（见下）
    C6  R8  失败样例可视化                     → failure_cases.py
    C7  R4  TFNet 荷兰异常诊断                → 需 TFNet 代码（本包不直接跑）

跑完后：
    - outputs/results/ 下有 multiseed_summary.json / maskonly_summary.json /
      sensitivity_summary.json 及逐种子指标 json；
    - outputs/failure_cases/sichuan/ 下有失败样例四联图；
    - 用 stats.py 的 mean±std 更新论文 Table 1/2，把 C1-C7 结论填进答复信
      （a_firstpaper/Response_to_Reviewers.tex 里的 \\todo 占位符）。
"""
import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                       # lss_BsiNet
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import config

# 所有可跑步骤，按推荐顺序排列
ALL_STEPS = ["c1", "c2", "c3", "c4", "c5", "c6", "c7"]

# C1 里的消融变体（对应论文 Table 2 的 Case A–G）
ABLATION_MODELS = ["full", "no_semfe", "no_spafe", "no_eca", "no_aru",
                   "no_spafe_no_eca", "only_aru"]


def banner(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def run(cmd, cwd=_ROOT):
    print("\n>>> " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"命令失败（exit={r.returncode}）: {' '.join(cmd)}")


def py(script, *args):
    return [sys.executable, os.path.join("revise_exp", script), *args]


# ---------------- 各步骤 ----------------

def step_c1():
    banner("C1 · 多种子训练+评估（full 及 6 个消融变体，3 数据集 × 3 种子）→ 汇总 + 配对 t 检验")
    # R1 多种子只需 HARNet 本体（full）；6 个消融沿用论文现有单次值，不重跑（最快）
    run(py("run_multiseed.py",
           "--models", "full",
           "--datasets", *list(config.DATASETS),
           "--seeds", *[str(s) for s in config.SEEDS],
           "--weights", "1,1,1"))
    # 汇总统计 + LaTeX 片段（full vs 各消融的配对 t 检验）
    run(py("stats.py"))


def step_c2():
    banner("C2 · mask-only 单任务消融（3 数据集 × 3 种子）")
    # mask-only 只补 Table 2 Case H（Shandong），单种子即可
    run(py("run_mask_only.py",
           "--datasets", "shandong",
           "--seeds", str(config.SEEDS[0])))


def step_c3():
    banner("C3 · 损失权重敏感性（shandong，4 组权重）")
    run(py("run_sensitivity.py", "--dataset", "shandong", "--seeds", str(config.SEEDS[0])))


def step_c4():
    banner("C4 · 参数量(trainable/total) + FLOPs 口径")
    run(py("compute_flops.py", "--model", "full", "--size", "512"))


def step_c5():
    banner("C5 · 未过滤(<40% 农田)测试集评估 —— 需要先配置未过滤数据集")
    # 未过滤测试集是独立目录，不在 config.DATASETS（已核对为过滤后数据集）里。
    # 到位后：在 config.DATASETS 里加一个 key（如 "shandong_unfiltered"），再跑：
    #   python revise_exp/evaluate.py --dataset shandong_unfiltered \
    #        --model full --checkpoint outputs/checkpoints/full_shandong_seed42_w1_1_1/best.pth
    unfiltered = [k for k in config.DATASETS if "unfiltered" in k.lower()]
    if unfiltered:
        for ds in unfiltered:
            ckpt = os.path.join(config.CKPT_DIR, "full_shandong_seed42_w1_1_1", "best.pth")
            run(py("evaluate.py", "--model", "full", "--dataset", ds,
                   "--checkpoint", ckpt, "--tag", f"full_{ds}_unfiltered"))
    else:
        print("[SKIP C5] config.DATASETS 里没有 *_unfiltered 数据集 key。")
        print("  数据到位后在 config.DATASETS 增加未过滤测试集 key，再重跑本步骤（见上面注释）。")


def step_c6():
    banner("C6 · 失败样例可视化（sichuan，最低 IoU 的 20 张四联图）")
    # 复用 C1 训出的 full 模型在 sichuan 的 best 权重
    ckpt = os.path.join(config.CKPT_DIR, "full_sichuan_seed42_w1_1_1", "best.pth")
    if not os.path.exists(ckpt):
        print(f"[WARN C6] 未找到 {ckpt}，请先跑 C1；本步骤跳过。")
        return
    run(py("failure_cases.py", "--dataset", "sichuan", "--model", "full",
           "--checkpoint", ckpt, "--seed", str(config.SEEDS[0]), "--topk", "20"))


def step_c7():
    banner("C7 · TFNet 荷兰异常诊断 —— 需要 TFNet 代码（本包不直接跑）")
    print("[MANUAL C7] TFNet 是独立基线代码库，不在 lss_BsiNet 内。")
    print("  请在 TFNet 代码里重跑 Netherlands（确认超参/划分一致），核对 73.14% vs Shandong 89.02% 的差异来源，")
    print("  把诊断结论写进答复信 R4 的 \\todo 占位符。")
    print("  若 C1 已重跑过 TFNet 多种子，则用那份结果直接覆盖 C7，无需单独再跑。")


STEPS = {
    "c1": step_c1,
    "c2": step_c2,
    "c3": step_c3,
    "c4": step_c4,
    "c5": step_c5,
    "c6": step_c6,
    "c7": step_c7,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="+", choices=ALL_STEPS, default=[],
                    help="要跳过的步骤（如 c3 c5）")
    ap.add_argument("--only", nargs="+", choices=ALL_STEPS, default=None,
                    help="只跑这些步骤（如 c1 c4）")
    ap.add_argument("--keep-going", action="store_true",
                    help="单步失败后继续后续步骤（默认失败即停止）")
    args = ap.parse_args()

    steps = [s for s in ALL_STEPS
             if s not in args.skip and (args.only is None or s in args.only)]

    print("HARNet 返修实验一键脚本")
    print(f"  工作目录 : {_ROOT}")
    print(f"  数据集   : {list(config.DATASETS)}")
    print(f"  种子     : {config.SEEDS}")
    print(f"  将按顺序执行 : {', '.join(steps) or '(无)'}")

    failed = []
    for s in steps:
        try:
            STEPS[s]()
        except Exception as e:
            print(f"\n[FAIL] 步骤 {s} 失败: {e}")
            failed.append(s)
            if not args.keep_going:
                print("  其余步骤未执行（加 --keep-going 可继续）。")
                break

    banner("执行完成")
    if failed:
        print(f"失败步骤: {failed}")
    else:
        print("所有选定步骤已执行。")
    print("下一步：")
    print("  1) 用 revise_exp/stats.py 的 mean±std 更新论文 Table 1/2；")
    print("  2) 把 C1–C7 结论填进 a_firstpaper/Response_to_Reviewers.tex 的 \\todo 占位符；")
    print("  3) C5/C7 若因数据/代码未到位而跳过，补跑后再补答复信。")


if __name__ == "__main__":
    main()
