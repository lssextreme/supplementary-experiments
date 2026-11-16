# -*- coding: utf-8 -*-
"""公共工具：随机种子、二分类指标。"""
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accumulate_confusion(pred, target):
    """累加二分类混淆矩阵计数 (TP, FP, FN, TN)。

    pred / target: 0/1 的 numpy uint8 数组（或可二值化的概率图）。
    """
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    tp = int(np.logical_and(pred == 1, target == 1).sum())
    fp = int(np.logical_and(pred == 1, target == 0).sum())
    fn = int(np.logical_and(pred == 0, target == 1).sum())
    tn = int(np.logical_and(pred == 0, target == 0).sum())
    return tp, fp, fn, tn


def metrics_from_confusion(tp, fp, fn, tn):
    eps = 1e-10
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "Acc": accuracy * 100.0,
        "F1": f1 * 100.0,
        "Recall": recall * 100.0,
        "Prec": precision * 100.0,
        "IoU": iou * 100.0,
    }


def metrics_from_preds(pred, target):
    """单张图二分类指标（0-100）。"""
    tp, fp, fn, tn = accumulate_confusion(pred, target)
    return metrics_from_confusion(tp, fp, fn, tn)
