# LGMFormer 复现项目 — PyCharm 使用说明

## 一、打开项目

1. PyCharm → `File` → `Open` → 选择目录 `D:\project\source\LGMFormer`
2. 配置解释器（关键，否则会 import 失败）：
   `File` → `Settings` → `Project: LGMFormer` → `Python Interpreter` →
   选 **Add Interpreter → Conda Environment → 已存在的环境** →
   路径填 `D:\software\Anaconda\env\pytorch\python.exe`
   （或直接选该 exe 作为现有解释器）

## 二、目录说明

```
LGMFormer/
├── train.py              # 训练主脚本（所有实验入口）
├── predict.py            # 单图推理
├── compare_runs.py       # 汇总 runs/ 结果生成消融对照表
├── README.md             # 项目说明（含各阶段命令）
├── 复现报告_LGMFormer.md  # 两天复现的完整报告
├── requirements.txt      # 依赖说明
├── src/
│   ├── models.py         # UNetResNet18 + HLFG/LGFI/WMSA/SegBranch/LGMFormer
│   ├── data.py           # ARIS 数据集（含 --aug 增强）
│   ├── losses.py         # MSE / DM-Count / CE+Dice
│   ├── metrics.py        # MAE/RMSE、OA/mIoU/F1
│   └── utils.py          # CSVLogger、set_seed、density_to_dots
├── weights/              # ResNet18 预训练权重缓存
└── runs/                 # 每次实验一个目录：log.csv / best.pth / test_metrics.json
```

数据在 `D:\project\source\ARIS\`（图像/密度图 npy + seg_labels + generate_seg_labels.py）。

## 三、跑实验（在 PyCharm 的 Terminal 里，用 conda env pytorch）

把下面命令里的 `python` 换成
`D:\software\Anaconda\env\pytorch\python.exe`（或直接在 PyCharm 解释器里跑）。

```bash
# 基线（已跑完，31 分钟）
python train.py --epochs 300 --val-every 10 --batch 12 --accum 3 --out runs/baseline

# 论文核心（已跑完，最佳 MAE 5.458）
python train.py --seg --sedh --loss dmcount --epochs 300 --batch 12 --accum 3 --out runs/mtl_sedh_dm

# 完整 LGMFormer transformer 解码器
python train.py --arch lgmformer --seg --sedh --loss dmcount \
    --epochs 300 --batch 12 --accum 3 --out runs/lgm_full

# 快速冒烟（1 轮，验证代码不崩）
python train.py --seg --sedh --loss dmcount --epochs 1 --limit 24 --batch 4 --accum 3 --out runs/smoke

# 看所有实验对照表
python compare_runs.py
```

## 四、关键开关

| 开关 | 作用 |
|---|---|
| `--arch {unet,lgmformer}` | 选解码器（U-Net 适配版 / 论文 transformer 版） |
| `--hlfg --hlfg-mode {cat3,add,replace}` | HLFG 融合方式 |
| `--lgfi` | LGFI 模块 |
| `--seg --sedh` | 分割分支 + SEDH |
| `--loss {mse,dmcount}` | 密度回归损失 |
| `--density-scale N` | 密度缩放（100=加速收敛 / 1=论文原生损失量级） |
| `--seg-weight F` | 分割损失权重 |
| `--aug` | 数据增强（**已证伪，别用**——破坏声呐点目标语义） |

## 五、坑（已在代码注释里说明，别重复踩）

- `numpy` 别升 2.x（与 torch 1.13 不兼容）
- 十字卷积偶数核需不对称 padding，且水平/垂直 pad 的宽高别写反
- SEDH 的 P_f/P_n 取 `prob[:,1:2]`/`[:,2:3]`（鱼=1/噪声=2），别取成背景
- 稀疏密度回归用 softplus + 按密度均值反解偏置（别用 ReLU 零初始化，梯度会被截断）
- 数据增强（擦除/颜色抖动）对本任务有害，已实验证伪

详见 `复现报告_LGMFormer.md` 的「关键发现与教训」。
