# 进度记录

> 每周更新一次，记录本周做了什么、结果如何、下周计划。
> 对应论文阅读笔记见 [`papers/`](papers/)，完整实验数据见 [`runs/`](runs/)。

## 第 1 周（2026-10-14 ~ 2026-10-16）：复现 LGMFormer 声呐鱼群计数

### 本周工作

| 环节 | 内容 | 产出 |
|---|---|---|
| 读论文 | 精读 LGMFormer（IEEE Sensors J. 2025），梳理 Eq.1–17 与 Fig.2–5 | [复现报告](Week1_LGMFormer/docs/复现报告.md) |
| 数据准备 | 下载 ARIS 数据集（500 张），按论文 Fig.5 用 Otsu 自动生成三类分割标签 | `seg_labels/`，一致率 99.4% |
| 搭基线 | ResNet18 + U-Net 密度回归 | MAE 6.666 |
| 复现模块 | HLFG / LGFI / SEDH / MTL / DM-Count 五个模块 + 论文原架构 Transformer 解码器 | `src/models.py` |
| 实验 | **11 次 300 轮完整实验**（覆盖论文全部消融项） | 见 README 结果总览 |
| 验收 | 端到端四阶段验收（数据→权重→评测→可视化） | [验收报告](Week1_LGMFormer/docs/全流程验收报告.md) |
| 汇报 | 13 页学术风格 PPT | `Week1_LGMFormer/docs/presentation/` |

### 本周结果

- **完整方案（MTL + SEDH + DM-Count）测试 MAE 5.458，较基线 6.666 降低 18.1%**
  —— 与论文宣称的 MTL 收益 18.8% 量级吻合，**核心命题复现成立**
- Transformer 完整方案验证集 MAE 3.708，已接近论文的 3.22（但测试集 5.845，泛化 gap）
- 已定位三个差距缺口：多任务 Pareto 权衡 / val→test 泛化 gap / 官方超参缺失

### 论文阅读

- CSRNet（CVPR 2018）、DM-Count（NeurIPS 2020）、LGMFormer（2025）
  → 整理成[技术谱系笔记](papers/2026-10-15_技术谱系_人群计数到鱼群计数.md)

### 下周计划

1. **压泛化 gap**：EMA 权重平均 + 测试时增强（TTA），不改数据语义
2. **补效率指标**：FPS / FLOPs / 参数量，与论文 Table VII 可比
3. **绕开多任务权衡**：任务预训练 + 联合微调的两阶段方案
4. **继续读论文**：人群计数方向 2–3 篇（重点看注意力模块在密集计数上的有效性）

---

## 模板（后续每周复制）

```markdown
## 第 N 周（YYYY-MM-DD ~ YYYY-MM-DD）：主题

### 本周工作
### 本周结果
### 论文阅读
### 下周计划
### 遇到的问题
```
