# HARNet 返修实验包（revise_exp）

本目录是 **HARNet（Applied Soft Computing 小修）** 的补实验脚本，把审稿意见逐条落实到可复现代码。
模型文件已统一改名为论文专属名（`models/harnet*.py`，原 `newdecoder7*.py / new7_*.py`），本包在其上做**非侵入式**加载与统一训练/评估。

---

## 1. 环境与运行前提

- 用本地 `rs-torch` conda 环境跑。所有命令都在 `lss_BsiNet/` 目录下执行：

```bash
conda activate rs-torch
cd /d/学业相关/HXJ大论文/lss_BsiNet
```

- **先改 `revise_exp/config.py` 顶部**：
  - `DATA_ROOT` —— 改成重新下载后的数据集实际根目录。
  - `DATASETS` 里三个子目录名（`shandong_256 / netherlands_256 / sichuan_256`）与
    `train/img|boundary|mask|dist_contour`、`test/img|mask` 结构按实际目录核对。
  - `FASTSAM_WEIGHT` —— 从官方 [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) 仓库下载 `FastSAM-x.pt` 并指向它（本仓库不随附任何权重）。

- **⚠️ 已发现 rs-torch 缺两个包，跑之前必须先装**（否则 `import ultralytics` 直接失败）：
  ```bash
  /c/Users/Administrator/.conda/envs/rs-torch/python.exe -m pip install requests psutil
  ```
  原因：`lss_BsiNet/ultralytics/`（项目内捆绑的 FastSAM 版 ultralytics 8.0.120）在
  `__init__.py` 顶层 `from ultralytics.hub import start` 需要 `requests`，
  `yolo/utils/checks.py` 等需要 `psutil`。这两个包在 base 环境 `D:\conda` 里有、但 rs-torch 没有。

- 其它依赖：`torch, numpy, PIL, tqdm, matplotlib`（失败样例用）；`thop`（FLOPs 用，可选）；`scipy`（t 检验 p 值，可选）。
  缺 `thop/scipy` 时脚本会跳过并提示。

---

## 2. 文件清单

| 文件 | 作用 | 对应审稿意见 |
|---|---|---|
| `config.py` | 全部路径/超参/种子/权重/模型注册的唯一入口 | — |
| `common.py` | 随机种子、二分类指标 | — |
| `harnet_model.py` | 加载父目录模型并恢复**三任务输出** `[boundary,mask,distance]` | 与训练一致 |
| `train.py` | 统一训练（多种子/权重/消融/mask-only） | C1/C2/C3 |
| `evaluate.py` | 测试集推理 + 5 指标（Acc/F1/Recall/Prec/IoU） | C1/C2/C3/C5 |
| `run_multiseed.py` | 多种子训练+评估，汇总 | **C1** |
| `run_mask_only.py` | mask-only 单任务消融 | **C2** |
| `run_sensitivity.py` | 损失权重敏感性 | **C3** |
| `compute_flops.py` | trainable/total 参数量 + FLOPs | **C4** |
| `stats.py` | mean±std + 配对 t 检验 + LaTeX 片段 | **C1** |
| `failure_cases.py` | 最低 IoU 样本四联图 | **C6** |

> 训练超参已按**论文正文**固定：Adam lr=0.001、150 epochs、batch=2、无梯度累积、
> 损失 1:1:1、512×512、验证集 10%、CONF=0.4/IOU=0.9/IMGSZ=1024。

---

## 3. 逐条跑法

### C1 · 多种子（≥3）+ 配对 t 检验
```bash
python revise_exp/run_multiseed.py \
  --models full no_semfe no_spafe no_eca no_aru no_spafe_no_eca only_aru \
  --datasets shandong netherlands sichuan \
  --seeds 42 123 2024
python revise_exp/stats.py
```
- 产物：`outputs/results/multiseed_summary.json`；控制台打印 mean±std、full vs 各消融的配对 t 检验、LaTeX 片段。
- 基线（TFNet/REAUNet/SEANet/BsiNet/DeepLabV3+）在各自代码里用同样 ≥3 种子重跑后，
  把结果整理成 `{baseline:{dataset:{seed:{"IoU":...}}}}` JSON，再：
  `python revise_exp/stats.py --baselines <path>` 得到 full vs 基线的配对 t 检验。

### C2 · mask-only 消融
```bash
python revise_exp/run_mask_only.py --datasets shandong netherlands sichuan --seeds 42 123 2024
```
- 产物：`outputs/results/maskonly_summary.json`，与三任务 full 的 IoU 对比说明辅助头收益。

### C3 · 损失权重敏感性
```bash
python revise_exp/run_sensitivity.py --dataset shandong --seeds 42
```
- 对 `config.WEIGHT_RATIOS`（1:1:1 / 1:0.5:0.5 / 1:1:0.5 / 1:2:1）各训一次，输出 IoU 对比，
  论证论文基准 1:1:1 稳健。可按算力加 `--seeds 42 123 2024`。

### C4 · 参数量/FLOPs 口径
```bash
python revise_exp/compute_flops.py --model full --size 512
```
- 明确区分 **trainable（1.78M，对应论文 Discussion）**、**frozen FastSAM-x 主干**、**total**。
- FLOPs 部分需要 `thop`；未装则先 `pip install thop`。

### C5 · 未过滤（<40% 农田）评估
- 论文正文实验是“过滤 <40% 农田占比”后的补丁；C5 需要**不带该过滤**的测试集再评一次，
  说明过滤只影响样本构成、不改变结论。
- 把未过滤测试集路径配成 `config.DATASETS` 里新增一个 dataset key（如 `shandong_unfiltered`），
  然后直接 `python revise_exp/evaluate.py --dataset shandong_unfiltered --model full --checkpoint <best.pth>`。

### C6 · 失败样例可视化
```bash
python revise_exp/failure_cases.py --dataset sichuan --checkpoint <sichuan best.pth> --topk 20
```
- 产物：`outputs/failure_cases/sichuan/*.png`（输入/GT/预测/叠加四联图，按 IoU 升序）。

### C7 · TFNet 在 Netherlands 的离群点诊断
- 属于基线复现与数据检查，需在 TFNet 代码里定位 Netherlands 低 IoU 原因（标注质量/类别不平衡/边界糊），
  本包不直接跑 TFNet；诊断结论写进返修回复信。

---

## 4. 论文 vs 代码 差异（已在脚本里按论文口径统一，但请确认）

| 项 | 论文正文 | 原始代码 | 本包处理 |
|---|---|---|---|
| epochs | 150 | 100（`train.py` 默认） | 用 150（`config.EPOCHS`） |
| lr | 0.001 | 0.0001 | 用 0.001（`config.LR`） |
| 损失权重 | 1:1:1 | 0.5/0.3/0.2（中间值） | 用 1:1:1 为基准 |
| 输出头 | 三任务 mask/boundary/distance | `return [x_out1,x_out2]#,x_out3`（distance 为匹配预测代码而注释） | `forward3` 恢复三头 |
| 梯度累积 | 无 | 无 | 无 |

**特征抽取机制已查清（供写返修回复信参考）：**

1. **FastSAM 多尺度特征**：`run_encoder` 的 `featsA[3]/[0]/[1]/[2]` 下标并非来自标准 `FastSAM.predict`。
   `lss_BsiNet` 项目里捆绑了一套改过的 ultralytics 8.0.120（`lss_BsiNet/ultralytics/`），其
   `yolo/engine/predictor.py` 第 244-246 行把 `stream_inference` 改成：
   ```python
   preds = self.model(im, ...)
   ms_feats = preds[-1]      # 取多尺度特征
   return ms_feats           # 提前返回，不再做后处理
   ```
   即 `self.model(...)` 返回 `[stride8(320ch), stride16(640ch), stride32(640ch), stride4(160ch)]` 四个特征图，
   HARNet 按 `feats[0]→stride8, feats[1]→stride16, feats[2]→stride32, feats[3]→stride4` 接 Adapter。
   本包在 `harnet_model.py` 里把 `lss_BsiNet` 加入 `sys.path`，因此 `import ultralytics` 解析到的正是这套捆绑 fork
   （不是 pip 安装的），复现路径与原 `train.py` 完全一致。回复信可据此写清“冻结 FastSAM-x 编码器如何输出多尺度特征”。

2. **通道数 eq(1)**：论文公式写 80/160/320/640，代码 Adapter 是 160/320/640/640（stride4/8/16/32）。
   若 eq(1) 描述的是主干特征通道，需与代码核对后统一（见返修回复信）。

---

## 5. 输出目录

```
revise_exp/outputs/
  checkpoints/<tag>/{best.pth, epochN.pth, last.pth, history.json}
  predictions/<tag>/*.png
  results/<tag>.json
  failure_cases/<dataset>/*.png
```

跑完后：用 `stats.py` 的 mean±std 更新论文 Table 1/2；把 C1-C7 的结论逐条写进返修回复信（Response to Reviewers）。
