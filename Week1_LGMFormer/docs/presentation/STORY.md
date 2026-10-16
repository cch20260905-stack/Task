# STORY.md · LGMFormer 声呐鱼群计数论文复现汇报

## ① 用户意图对齐

**目标受众**：研究生（一年级）向导师汇报阶段性科研工作。导师有专业背景，能看懂技术细节，会追问方法与结果的可信度。

**核心目标**：讲完后导师相信三件事——(1) 我真的完整读懂了论文并按公式/图逐条复现了方法；(2) 我拿到了可复现的量化结果（MAE ↓18.1%，与论文宣称量级吻合）；(3) 我对未达标的部分有准确的归因判断，不是"跑不出来"而是"定位到了具体障碍"。

**PPT 长度**：共 13 页（Hero 4 页 = 30%）。

**视觉调性**：学术严谨 / 证据驱动 / 蓝白克制 / 诚实量化。

**内容边界**：
- 必讲：论文方法与我实现的一一对应、11 个实验的对照表、最佳结果 MAE 5.458（↓18.1% vs 基线）、三个未达标缺口及归因、下一步计划
- 不讲：逐行代码细节、超参网格搜索过程、调试期间的失败记录
- 禁碰：夸大结果、掩饰缺口、虚构论文未公开的超参

---

## ② 页面布局骨架

**页面总数与分章（13 页）**：

| 章节 | 页码 | 页面 |
|---|---|---|
| 开篇 | P1–P2 | P1 封面(hero) / P2 目录 |
| 一、背景与目标 | P3–P5 | P3 章节扉页(section 01) / P4 论文方法概览 / P5 我的任务边界 |
| 二、实现与方法 | P6–P8 | P6 章节扉页(section 02) / P7 模块实现对照 / P8 数据侧工作(分割标签) |
| 三、实验与结果 | P9–P11 | P9 章节扉页(section 03) / P10 11实验对照表(hero) / P11 训练曲线对比 |
| 四、差距归因与计划 | P12–P13 | P12 三个缺口归因(hero) / P13 下一步计划 + 结束(hero) |

**目录↔章节扉页契约**：目录声明 **4 个章节**（①背景与目标 ②实现与方法 ③实验与结果 ④差距归因与计划），对应 4 个 section 扉页：
- 01 背景与目标 → P3
- 02 实现与方法 → P6
- 03 实验与结果 → P9
- 04 差距归因与计划 → P12（chapter 封面与内容合并页，编号 04 标注）
✓ 4 个 section、编号 01–04 连续、标题与目录逐字一致。

**Hero 页定位**（4/13 = 30%）：P1 封面 / P10 对照表 / P12 缺口归因 / P13 计划+结束。任意两个 Hero 之间至少间隔 1 个 supporting 页。✓

**rhythm 曲线**：
P1 peak → P2 valley → P3 transition → P4 valley → P5 valley → P6 transition → P7 valley → P8 valley → P9 transition → P10 peak → P11 valley → P12 peak → P13 peak
（无连续 ≥3 页 valley；P5→P6 由 transition 打破，✓）

**非对称版式预算**：13 页中非对称 ≥ 6 页（46%）。分配：P1 全幅图+骑线文字 / P4 左大图+右侧文字 / P5 左标题+右内容 / P7 非对称双栏 / P8 左大图+右侧文字 / P11 左大图+右侧文字 / P12 巨型数字+洞察。

**对称版式预算**（最多 2 页）：P2 目录（N卡片横排不适用，用编号列表）/ P10 图表+洞察（表格页）。

---

## ③ 页面大纲

### P1 封面
- title: LGMFormer 声呐鱼群计数论文复现
- type: cover | role: hero | rhythm: peak
- layout: 全幅图+骑线文字
- visual: L1: 背景用训练曲线图局部虚化（占右 45%）+ 深蓝侧栏 | visual_role: atmosphere
- density: 字数约 60 / 图片 1 张 / 留白约 30%
- anti_pattern: 禁止党政红金；禁止印章烟花；禁止人物大头照；禁止把标题塞进半透明卡片
- description: 论文 Enhancing Fish Counting in Sonar Images With Multitask Learning and Local-Global Feature Interaction 复现报告——两天 11 次实验，核心方法 MAE 降低 18.1%。

### P2 目录
- title: 汇报提纲
- type: catalog | role: supporting | rhythm: valley
- layout: 左标题+右内容（编号列表）
- visual: L3: 编号方块 01–04 | visual_role: evidence
- density: 字数约 120 / 图片 0 张 / 留白约 25%
- anti_pattern: 禁止圆球编号；禁止英文比中文抢眼；禁止书法繁体数字
- description: 四个部分：背景与目标 → 实现与方法 → 实验与结果 → 差距归因与计划。

### P3 章节扉页 01
- title: 背景与目标
- type: section | role: transition | rhythm: transition
- layout: 巨型数字+洞察
- visual: 大数字「01」最小 160px | visual_role: anchor
- density: 字数约 80 / 图片 0 张 / 留白约 40%
- anti_pattern: 禁止四卡片预览；禁止铺满正文段落；禁止与封面雷同
- description: 论文提出 LGMFormer，用多任务学习（密度回归 + 三类分割）与局部-全局特征交互提升声呐鱼群计数精度，官方宣称 MAE 3.22 / mIoU 90.2%。

### P4 论文方法概览
- title: 论文方法概览
- type: content | role: supporting | rhythm: valley
- layout: 左大图+右侧文字
- visual: L1: 论文 Fig.2 结构示意（用 SVG 复刻，占左 55%）| visual_role: evidence
- density: 字数约 230 / 图片 1 张 / 留白约 20%
- anti_pattern: 禁止把架构图缩成 200×70 角标；禁止 50:50 等分双栏
- description: 编码器 ResNet18 + transformer 解码器（LGFI 堆叠）+ HLFG 引导融合 + MTL 分割 + SEDH 精修——四个创新点串成一条"用分割辅助计数"的主线。

### P5 我的任务边界
- title: 复现条件与现实约束
- type: content | role: supporting | rhythm: valley
- layout: 左标题+右内容
- visual: FAIcon列表 | visual_role: evidence
- density: 字数约 200 / 图片 0 张 / 留白约 20%
- anti_pattern: 禁止等宽四卡；禁止信息板式罗列无判断
- description: 官方代码仓库为空（GitHub 仅 "code will be released soon"）→ 必须从论文公式与图逐条自实现；论文关键超参（decoder 通道、层数）未公布 → 需自定并实验验证。

### P6 章节扉页 02
- title: 实现与方法
- type: section | role: transition | rhythm: transition
- layout: 巨型数字+洞察
- visual: 大数字「02」最小 160px | visual_role: anchor
- density: 字数约 80 / 图片 0 张 / 留白约 40%
- anti_pattern: 禁止四卡片预览；禁止与 P3 版式雷同
- description: 数据侧（Otsu 三类标签）+ 基线 + 四个核心模块，全部按论文 Eq.1–17 与 Fig.2–5 实现。

### P7 模块实现对照
- title: 四个核心模块的实现对照
- type: content | role: supporting | rhythm: valley
- layout: 非对称双栏（60:40）
- visual: L2: HLFG/LGFI/SEDH/MTL 结构小图（占窄栏）| visual_role: evidence
- density: 字数约 240 / 图片 2 张 / 留白约 18%
- anti_pattern: 禁止等宽四卡横排；禁止把公式缩到不可读
- description: HLFG 用高层特征生成通道注意力引导低层（Eq.1–3）；LGFI 用窗内注意力+十字卷积+多尺度空洞卷积（Eq.4–6）；SEDH 用分割概率增强鱼区、抑制噪声（Eq.9–11）；MTL 用 CE+Dice 辅助监督。

### P8 数据侧工作（Otsu 分割标签）
- title: Otsu 自动生成三类分割标签
- type: content | role: supporting | rhythm: valley
- layout: 左大图+右侧文字
- visual: L1: seg_check.png（占左 58%）| visual_role: evidence
- density: 字数约 210 / 图片 1 张 / 留白约 22%
- anti_pattern: 禁止把全流程拼图缩小；禁止只贴图不给判断
- description: 按论文 Fig.5 五步流程（灰度→Otsu→密度图掩码→位置筛选→后处理）生成 500 张 fish/noise/background 标签；密度图 Otsu 等效阈值与论文固定 0.001 的一致率 99.4%——验证标签可靠。

### P9 章节扉页 03
- title: 实验与结果
- type: section | role: transition | rhythm: transition
- layout: 巨型数字+洞察
- visual: 大数字「03」最小 160px | visual_role: anchor
- density: 字数约 80 / 图片 0 张 / 留白约 40%
- anti_pattern: 禁止四卡片预览；禁止与 P6 版式雷同（改用左对齐）
- description: 11 次 300 轮完整实验，覆盖论文全部消融项与两种解码器架构。

### P10 11 实验对照表（Hero）
- title: 全部实验结果对照
- type: content | role: hero | rhythm: peak
- layout: 图表+洞察（表格为主）
- visual: Chart(表格) 占 ≥65% + 底部结论条 | visual_role: anchor
- density: 字数约 190 / 表格 1 张 / 留白约 15%
- anti_pattern: 禁止把最佳行淹没在表中（必须高亮）；禁止只列表无结论
- description: 完整方案（MTL+SEDH+DM-Count）测试 MAE 5.458、RMSE 10.287，较基线 6.666 降低 18.1%——与论文宣称的 MTL 收益 18.8% 量级吻合，核心命题复现成功。

### P11 训练曲线对比
- title: 收敛过程对比
- type: content | role: supporting | rhythm: valley
- layout: 左大图+右侧文字
- visual: L1: training_curves.png（占左 62%）| visual_role: evidence
- density: 字数约 180 / 图片 1 张 / 留白约 20%
- anti_pattern: 禁止把双联图压小；禁止只贴图不给结论
- description: 完整方案（红线）在 MAE 与 RMSE 两图全程最低，说明优势来自整条收敛轨迹而非单点波动；LGFI 与 HLFG-replace 曲线明显高于基线，佐证其在本实现的适配下不 work。

### P12 三个缺口归因（Hero）
- title: 距离论文完整数字的三个缺口
- type: content | role: hero | rhythm: peak
- layout: 巨型数字+洞察
- visual: 大数字「5.458 vs 3.22」+ 三缺口 | visual_role: anchor
- density: 字数约 260 / 图片 0 张 / 留白约 20%
- anti_pattern: 禁止只报缺口不给原因；禁止把缺口写成"暂时未完成"式含糊表述
- description: 缺口一多任务权衡——实验证实 MAE 最优与 mIoU 最优不在同一训练阶段，两任务此消彼长；缺口二泛化 gap——val 3.708 已接近论文 3.22，但 test 掉到 5.845，属过拟合方向，数据增强已实验证伪；缺口三官方超参缺失——仓库为空，decoder 通道/训练细节无对照依据。

### P13 下一步计划 + 结束
- title: 下一步计划
- type: ending | role: hero | rhythm: peak
- layout: 非对称双栏（65:35）
- visual: L2: 计划步骤胶囊串 | visual_role: anchor
- density: 字数约 170 / 图片 0 张 / 留白约 25%
- anti_pattern: 禁止把计划写成空泛口号；禁止烟花金装饰
- description: 近期：压 val/test gap（EMA 与测试时增强，不改数据语义）、补充 FPS/FLOPs 效率指标、任务预训练+联合微调绕开权衡。中期：跟踪官方仓库、尝试跨数据集泛化验证。
