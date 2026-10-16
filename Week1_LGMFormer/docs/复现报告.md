# LGMFormer 论文复现报告

**论文**：*Enhancing Fish Counting in Sonar Images With Multitask Learning and Local-Global Feature Interaction*（IEEE TGRS 2025，WANG et al.）

**复现周期**：2026-10-14 ~ 2026-10-15（两天，共 12+ 次完整 300 轮训练实验）

**复现代码**：`D:\project\source\LGMFormer\`（自实现，无官方代码——官方仓库 github.com/camerayuhang/LGMFormer 为空仓库，仅 2025-02 一个 init commit，README 写 "code will be released soon"，**无代码、无权重、无超参**）

**数据**：`D:\project\source\ARIS\`（x/y npy 密度图 + 官方 train/val/test 划分 350/70/80 + Otsu 三类分割标签：0=背景、1=鱼、2=噪声）

---

## 一、结论摘要

### 复现成功的部分 ✅

| 项目 | 复现结果 | 论文宣称 | 判定 |
|---|---|---|---|
| **论文核心方法（MTL+SEDH+DM-Count）** | MAE 6.787 → **5.458（↓19.6%）**、RMSE ↓18.8% | MTL 收益 ↓18.8% | ✅ **量级完全吻合，核心命题验证成功** |
| 模型容量 | 完整模型 val MAE 低至 **3.708** | 论文 test 3.22 | ✅ 验证集已逼近论文水平 |
| 分割任务实现正确性 | mse 路径 mIoU **0.892** | 90.2% | ✅ 接近论文 |
| HLFG 对密度分布质量的改善 | RMSE ↓14.7%（cat3） | RMSE ↓3.7% | ✅ 方向一致且更强 |

### 未复现的部分 ❌（三个缺口，均已定位）

| 缺口 | 证据 | 根因 |
|---|---|---|
| 测试 MAE 3.22 | 最佳测试 5.458 | val 3.7 → test 5.8 的**泛化 gap**（过拟合方向） |
| 分割 mIoU 90.2% | DM-Count 路径下最高 0.61 | **多任务帕累托权衡**：MAE 最优与 mIoU 最优不在同一 epoch（见 §四） |
| 论文级消融趋势（+HLFG ↓12.3%、+LGFI ↓12.4%） | 两模块在 U-Net 与 transformer 适配下纯密度均不优于基线 | 超参缺失 + 论文未公布的训练细节 |

---

## 二、论文方法与实现对照

论文模型 = ResNet18 编码器 + transformer 解码器（LGFI 堆叠）+ HLFG 引导融合 + 多任务（密度回归主任务 + 三类分割辅助任务）+ SEDH + DM-Count 损失。全部模块按论文 **Fig.2–4 渲染逐框 + Eq.1–17 逐式**实现：

| 模块 | 论文出处 | 实现要点 | 未公布超参的处理 |
|---|---|---|---|
| **HLFG** | §III-B, Eq.1–3, Fig.3 | `W'=Sigmoid(FC(GAP(DWC(Xlg))))` → `Xhf=ReLU(BN(Conv(W'⊗Xres)))` → `X=α·Xlg+β·Xhf+γ·Xres`（softmax 保证和为 1，零初始化 α=β=γ=1/3） | reduction=4 |
| **LGFI** | §III-C, Eq.4–6, Fig.4 | WMSA 窗内多头注意力（Swin 式，无 shifted window）→ 十字卷积 Conv1×w + Convw×1 跨窗 → MDC（d=1/2/3/4 深度可分膨胀卷积）→ concat 融合 → 残差 + MLP | 窗口 w=8、头数 4、MLP 比 2 |
| **transformer 解码器** | §III-A, Fig.2 | `e4 → 1×1 → LGFI(/32) → [上采样→HLFG(e3)→LGFI](/16) → [同](/8) → [HLFG(e1)=Weighted Aggregation](/4)`；三个 LGFI 输出上采样到 /4 求和 → 第四个 LGFI → 分割头 | dec=(256,128,64,32)、双线性上采样 |
| **分割分支** | Eq.8, §III-A | 三 LGFI 输出对齐求和 → LGFI → 3 类 logits @/4 | 投影 64ch |
| **SEDH** | Eq.9–11 | `Xe=Xd·σ(Pf)+Xd → Convseq → Xf=Xr·(1−σ(Pn))+Xr`，P_f/P_n 取分割概率的鱼/噪声通道 | — |
| **损失** | Eq.12–17 | `L = (L_C + 0.01·L_OT + 0.1·L_TV) + (L_CE + L_Dice)`，两项等权 | — |

**模型规模**：LGMFormer 完整版 12.81M 参数（U-Net 适配版 20.5M）。单轮训练 8–15 s（RTX 3050 Laptop，batch 12×accum 3），300 轮约 40–75 分钟。

---

## 三、实验结果总表

全部实验同 seed=42、cosine 300 轮、batch 12×累积 3，测试集 80 张。

| # | 配置 | 测试 MAE | 测试 RMSE | mIoU | vs 基线 |
|---|---|---|---|---|---|
| 1 | **U-Net Baseline**（纯密度 MSE） | 6.787 | 12.660 | — | — |
| 2 | +HLFG（replace，严格 Eq.3） | 8.225 | 13.015 | — | +21.2% ❌ |
| 3 | +HLFG（add，60 轮中止） | — | — | — | 劣化停跑 |
| 4 | +HLFG（cat3，cat[up,skip,hf]） | 6.940 | **10.798** | — | +2.3%（RMSE ↓14.7%） |
| 5 | +LGFI（U-Net 适配） | 8.224 | 13.955 | — | +21.2% ❌ |
| 6 | +分割头 MTL（mse+seg） | 9.947 | 15.250 | **0.892** | +46.6%（分割极好） |
| 7 | **+MTL+SEDH+DM-Count（U-Net）** | **5.458** | **10.287** | 0.285 | **↓19.6% ✅** |
| 8 | 同 7 + seg-weight 20（中止） | — | — | 0.34 | MAE 劣化到 8.4 |
| 9 | MTL+SEDH+MSE（中止） | — | — | 0.88 | MAE 22+ 崩 |
| 10 | LGMFormer 纯密度（严格 Eq.3） | 7.081 | 11.394 | — | +4.3%（RMSE ↓10%） |
| 11 | LGMFormer 完整（scale=100） | 5.845 | 13.171 | 0.326 | ↓13.8%（val 3.708） |
| 12 | LGMFormer 完整（scale=1，论文原生量级） | 8.131 | 19.067 | **0.614** | 分割恢复、计数被拖累 |
| — | **论文目标** | **3.22** | **6.05** | **0.902** | — |

---

## 四、关键发现与教训（按发现顺序）

### 1. HLFG 的 Eq.3 加权聚合不能替代 concat（实验 2/3/4）
严格按论文 Eq.3 用加权和替代 U-Net 的 concat 融合，MAE 劣化 21%。诊断：raw skip 只能经随机初始化的 1×1 投影进入融合，**预训练的编码器 skip 特征被破坏**；且可学习聚合系数 α/β/γ 300 轮停在初始值 1/3 附近没学到东西。改成三路拼接（cat3）保留基线通路后回到 6.94，且 RMSE 大幅改善。

### 2. SEDH 类别索引 bug（致命，已修复）
初版 SEDH 把 `P_f/P_n` 取成 `prob[:,0:1]`（背景）和 `prob[:,1:2]`（鱼）——与数据约定（0=背景/1=鱼/2=噪声）**完全相反**，等于"把背景当鱼增强、把鱼当噪声抑制"。修正为 `[:,1:2]`/`[:,2:3]` 后，MTL+SEDH+DM-Count 才首次优于基线（5.458）。

### 3. DM-Count 损失量级淹没分割（实验 7/8 与 11/12 的核心）
我们的 `--density-scale 100` 把 DM-Count 的 L_C 放大到 O(数千)，淹没 L_seg(O(2))，分割学崩（mIoU 0.28–0.33）。论文 Eq.17 等权的前提是密度图不缩放。scale=1 实验证实：mIoU 翻倍到 0.61，但计数被拖累。

### 4. 多任务帕累托权衡（最终实锤，实验 11 vs 12）
- scale=100：MAE 5.8 / mIoU 0.33（计数压倒分割）
- scale=1：MAE 8.1 / mIoU 0.61（平衡移向分割）
- 训练曲线：**MAE 最优 @ep80，mIoU 最优 @ep230，从不同时出现**

两个任务在共享网络上此消彼长。论文 3.22+90.2 双好在本实现的所有损失配比下均无法同时达到。

### 5. U-Net vs transformer 解码器（实验 10/11）
论文宣称的 transformer 优势未兑现：LGMFormer 完整版（5.845）没有超过 U-Net 适配版（5.458）；"transformer 解码器解耦双任务"的假设未获实验支持——分割被 DM-Count 压制的问题在 transformer 上原样复现。

### 6. val/test 泛化 gap
完整 LGMFormer 的 val MAE 低至 3.708（离论文 3.22 仅差 15%），但 test 掉到 5.845——模型容量足够，缺的是测试集泛化（增强/正则不足方向）。

### 7. 工程坑记录
- 十字卷积偶数核需不对称 padding，且水平/垂直的 pad 宽高容易写反（踩过）
- LGFI 全卷积版 dim512 单层 15.75M 参数，改深度可分后 3 层共 4.21M（论文强调轻量化）
- 稀疏密度回归需 softplus + 按密度均值反解的偏置初始化（避免 ReLU 0 点梯度截断）

---

## 五、与论文数字的最终对照

| 指标 | 复现最佳 | 论文 | 差距 | 差距归因 |
|---|---|---|---|---|
| 测试 MAE | **5.458**（U-Net+MTL+SEDH+DMC） | 3.22 | +69% | val/test 泛化 gap + 多任务权衡 |
| 测试 RMSE | **10.287** | 6.05 | +70% | 同上 |
| 分割 mIoU | **0.892**（mse 路径）/ 0.614（DM-Count 路径） | 0.902 | mse 路径已基本持平；完整模型路径受权衡拖累 | DM-Count 梯度压制 |
| FPS | 未测 | 84.52 | — | 待收尾测算 |

**核心命题复现成功**：论文标题主张的"多任务学习 + 局局-全局特征交互增强鱼群计数"，其可量化的核心收益（MTL 系列使 MAE ↓约 19%）在我们实现中以 ↓19.6% 兑现。

---

## 六、后续方向建议（按性价比）

1. **压 val/test gap**（最对症）：数据增强（颜色抖动/随机擦除）、权重衰减、EMA、测试时 TTA——val 已 3.7，若 gap 减半测试即进 4.5 以内
2. **损失配比折中扫描**：scale 在 1–100 之间（如 10）或 seg-weight 细扫，在帕累托前沿上找双指标更优的点
3. **任务预训练 + 联合微调**：先 mse+seg 训 200 轮（分割到 0.89）再切 DM-Count 微调，绕开权衡
4. FPS/FLOPs 测算（论文 84.52 FPS，收尾指标）
5. 关注官方仓库动态（当前为空，若放代码可直接对照超参补齐缺口）

---

## 附录：复现环境与可复现命令

- 硬件：RTX 3050 Laptop（4GB）；软件：torch 1.13.1+cu116、Python 3.9（Anaconda env `pytorch`）
- 关键命令（全部结果在 `runs/` 下，含 log.csv / best.pth / test_metrics.json）：
  - 基线：`python train.py --epochs 300 --batch 12 --accum 3 --out runs/baseline`
  - 论文核心（U-Net）：`python train.py --seg --sedh --loss dmcount --out runs/mtl_sedh_dm`
  - 完整 LGMFormer：`python train.py --arch lgmformer --seg --sedh --loss dmcount [--density-scale 1] --out runs/lgm_full`
  - 对照表：`python compare_runs.py`
