# -*- coding: utf-8 -*-
"""HARNet 返修实验统一配置。

运行前只需要改本文件顶部的路径（DATA_ROOT / 各数据集子目录名）。
其余脚本（train.py / evaluate.py / run_*.py / compute_flops.py / stats.py）都从这里读参数。
"""
import os

# ============ 路径 ============
# 工程根目录 = lss_BsiNet（本文件位于其下的 revise_exp/）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# FastSAM-x 权重（沿用 train.py 里的默认权重文件，位于 lss_BsiNet 根目录）
FASTSAM_WEIGHT = os.path.join(ROOT, "FastSAM-x.pt")

# 数据集根目录 —— 已核对为 D:\data1 下的三个子目录
DATA_ROOT = r"D:\data1"

# 目录结构（已核对 D:\data1 实际目录；注意三点与初版假设不同）：
#   1) 训练图像子目录叫 `image`（不是 `img`）；距离子目录叫 `distance`（不是 `dist_contour`）。
#   2) Netherlands 的测试图像子目录是 `img`（与其余两家的 `image` 不一致，单独处理）。
#   3) 路径统一加尾部分隔符，因 dataset.py 里是 `glob(dir + id + '.*')`，不带分隔符会匹配不到。
def _join(*parts):
    return os.path.join(DATA_ROOT, *parts) + os.sep


def _ds(name, test_img_sub="image"):
    return {
        "train": {
            "img":      _join(name, "train", "image"),
            "boundary": _join(name, "train", "boundary"),
            "mask":     _join(name, "train", "mask"),
            "distance": _join(name, "train", "distance"),
        },
        "test": {
            "img":  _join(name, "test", test_img_sub),
            "mask": _join(name, "test", "mask"),
        },
    }


# 三个数据集对应的子目录名（与 D:\data1 下的实际文件夹名一致）
DATASETS = {
    "shandong":    _ds("Shangdong"),
    "netherlands": _ds("Netherlands", test_img_sub="img"),
    "sichuan":     _ds("Sichuan"),
}

# ============ 输出目录 ============
_BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_BASE, "outputs")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
PRED_DIR = os.path.join(OUT_DIR, "predictions")
RESULT_DIR = os.path.join(OUT_DIR, "results")
for _d in (OUT_DIR, CKPT_DIR, PRED_DIR, RESULT_DIR):
    os.makedirs(_d, exist_ok=True)

# ============ 训练超参（以论文正文为准）============
EPOCHS = 150
BATCH_SIZE = 2
LR = 0.001
WEIGHT_DECAY = 1e-8
VAL_PERCENT = 0.10          # 训练集的 10% 作为验证（论文：9:1 划分，训练集 10% 验证）
NUM_WORKERS = 0          # 回退为 0：本机仅 15.7G 内存，4 worker 各自拷贝 torch 触发 DataLoader MemoryError（WinError 1455 页文件不足）；0 稳定但 GPU 约 11%
# FastSAM 推理参数（论文 4.2.1 / 复杂度表均以 512×512 输入为准；补丁原始尺寸即 512）
CONF = 0.4
IOU = 0.9
IMGSZ = 512
# 掩码二值化阈值
MASK_THRESHOLD = 0.5

# ============ 多种子（C1） ============
SEEDS = [42, 123, 2024]

# ============ 损失权重敏感性（C3） ============
# 顺序：(mask, boundary, distance)。"1:1:1" 是论文报告的基准。
WEIGHT_RATIOS = {
    "1:1:1":     (1, 1, 1),        # 用 int 使目录名 w1_1_1 与 C1 的 seed 基准一致，直接复用 C1 已训好的权重，免重复训练
    "1:0.5:0.5": (1.0, 0.5, 0.5),
    "1:1:0.5":   (1.0, 1.0, 0.5),
    "1:2:1":     (1.0, 2.0, 1.0),
}

# ============ 模型注册（消融变体 = lss_BsiNet/models 下的 HARNet 专属文件）============
# key -> (module, class)。Case 对应论文 Table 2（tab:ablation_study）。
MODEL_REGISTRY = {
    "full":            ("models.harnet",                 "HARNet"),  # A: SemFE + SpaFE + ECA + ARU
    "no_semfe":        ("models.harnet_no_semfe",        "HARNet"),  # B: 去掉 SemFE(GHF)
    "no_spafe":        ("models.harnet_no_spafe",        "HARNet"),  # C: 去掉 SpaFE(SE)
    "no_eca":          ("models.harnet_no_eca",          "HARNet"),  # D: 去掉 ECA
    "no_aru":          ("models.harnet_no_aru",          "HARNet"),  # E: 去掉 ARU
    "no_spafe_no_eca": ("models.harnet_no_eca_no_spafe", "HARNet"),  # F
    "only_aru":        ("models.harnet_only_aru",        "HARNet"),  # G
}

# 论文主对比表的基线（C1 需重跑，见 README 第 C1 节）
BASELINES = ["TFNet", "REAUNet", "SEANet", "BsiNet", "DeepLabV3+"]
