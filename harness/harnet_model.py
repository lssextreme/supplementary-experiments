# -*- coding: utf-8 -*-
"""HARNet 模型加载器。

在已改名为论文专属名的 models/harnet*.py 基础上（不再改网络结构）：
  1) 用 forward3 覆盖 forward，恢复论文的「三任务输出」 [boundary, mask, distance]；
  2) 提供 build_model(name) 统一入口，供 train/evaluate/compute_flops 使用。

原始 harnet.py 的 forward 里 `return [x_out1, x_out2]#, x_out3` 是把 distance 头
注释掉用于匹配预测代码的；训练时是三个任务。这里把三个头都返回。
"""
import importlib
import os
import sys

import torch
from torch.nn import functional as F

# 把 lss_BsiNet 根目录加进 sys.path，以便 import models.* / dataset / loss_yb
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from config import MODEL_REGISTRY, FASTSAM_WEIGHT, CONF, IOU, IMGSZ  # noqa: E402


def forward3(self, x):
    """三任务 forward，等价于原始 forward，只是把 distance 头一并返回。"""
    featsA = self.run_encoder(x)

    featA_s4 = self.Adapter4(featsA[3].clone())
    featA_s8 = self.Adapter8(featsA[0].clone())

    drops = getattr(self, "drops", None)
    if drops is not None:
        featA_s4 = drops(featA_s4)
        featA_s8 = drops(featA_s8)

    # SpaFE (SE) —— 无 SpaFE 的消融变体里 self.spafe1/spafe2 不存在
    if hasattr(self, "spafe1"):
        featA_s4 = self.spafe1(featA_s4)
    if hasattr(self, "spafe2"):
        featA_s8 = self.spafe2(featA_s8)

    featA_s16 = self.Adapter16(featsA[1].clone())
    featA_s32 = self.Adapter32(featsA[2].clone())
    if drops is not None:
        featA_s16 = drops(featA_s16)
        featA_s32 = drops(featA_s32)

    # SemFE (GHF) —— 无 SemFE 的消融变体里 self.semfe 不存在
    if hasattr(self, "semfe"):
        featA_s32 = self.semfe(featA_s16, featA_s32)

    encoder = [featA_s4, featA_s8, featA_s16, featA_s32]
    x_9 = self.decoder(encoder)
    x_out = self.sge(x_9)

    x_out1 = self.conv_final1(x_out)   # boundary (logits)
    x_out2 = self.conv_final2(x_out)   # mask    (logits)
    x_out3 = self.conv_final3(x_out)   # distance (回归)
    if self.num_classes > 1:
        x_out1 = F.log_softmax(x_out1, dim=1)
        x_out2 = F.log_softmax(x_out2, dim=1)
    x_out3 = torch.sigmoid(x_out3)

    # 顺序与 train.py / loss_yb.MultiTaskLoss 的解包顺序一致：
    #   pred_boundary, pred_mask, pred_distance = net(imgs)
    return [x_out1, x_out2, x_out3]


def build_model(name, device="cuda", model_name=None):
    """按 config.MODEL_REGISTRY 加载某个（消融）变体，并套上三任务 forward。"""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"未知模型 '{name}'，可选：{list(MODEL_REGISTRY)}")

    mod_path, cls_name = MODEL_REGISTRY[name]
    module = importlib.import_module(mod_path)
    BaseNet = getattr(module, cls_name)

    class Net3Head(BaseNet):
        forward = forward3

    if model_name is None:
        model_name = FASTSAM_WEIGHT
    return Net3Head(model_name=model_name, device=device, conf=CONF, iou=IOU, imgsz=IMGSZ)
