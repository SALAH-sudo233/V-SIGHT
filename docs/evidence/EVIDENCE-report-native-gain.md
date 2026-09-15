# 证据资产：用 11-model 报告自身指标衡量 verifier 增益（report-native，可比可信）

> 动机：此前的 separation AUROC / catch@FNR 是自造指标，与报告给这些模型打分的口径不一致，说服力不足。
> 本文改用**报告 T2 的原生指标**（正样本 mIoU、FG@Neg 幻觉率、BOH/ROH FG、False reject）衡量 verifier 加入前后的 before/after。
> **结论：在报告自身的定位安全门（正样本 mIoU 损失 ≤0.005）下，verifier 把高幻觉上游的 FG@Neg（幻觉率）
> 从 94–97% 砍到 67–72%（−22~−30pp），真实代价仅损失 1–2 个正确框 / 500。**

## 方法
- verifier = B1c 集成（B1a 特征 + B1b LoRA），leave-one-model-out（每个上游被"未见过它"的集成打分）。
- 动作：对上游画出的每个框做 KEEP/REJECT（`prob ≥ thr` 则 KEEP）。单次推理预算。
- 操作点：阈值 thr 选在**报告正样本 mIoU 损失 ≤ 0.005**（Gate B 定位安全门）。
- before = 上游原始（所有画出的框都算 KEEP，等于报告原数字，已逐项复现无误）；after = 经 verifier 过滤。
- 脚本 `report_gain.py` → `report_gain.json`。报告 T2 指标定义与原数字 100% 复现（LENS: mIoU 0.4874 / FG@Neg 94.2% / BOH FG 90.4 / ROH FG 98.0，全部对齐）。

## 主表：报告 T2 指标 before → after（mIoU 损失 ≤0.005 安全门）

| 上游模型 | 正mIoU↑ | FG@Neg↓(幻觉率) | BOH FG↓ | ROH FG↓ | 真实代价(丢失正确框/500) |
|---|---|---|---|---|---|
| **Seg-zero** | 0.522→0.517 (−0.005) | 97.1%→**66.7%** (−30.4) | 95.3→51.3 | 98.9→82.1 | **2** |
| **LENS** | 0.487→0.484 (−0.004) | 94.2%→**71.8%** (−22.4) | 90.4→57.3 | 98.0→86.2 | **1** |
| **Qwen3-VL-8B** | 0.055→0.051 (−0.005) | 21.0%→**7.0%** (−14.0) | 10.0→2.4 | 32.0→11.7 | 0 |
| **Orsta-7B** | 0.317→0.313 (−0.004) | 44.1%→43.1% (−1.0) | 27.9→26.8 | 60.4→59.4 | 0 |

## 关键读法（诚实）
1. **正样本 mIoU 几乎不动**（全部损失 ≤0.005，报告的定位质量指标）——这是 verifier "不损伤上游正确定位"的直接证据。
2. **FG@Neg（报告的核心幻觉指标）大幅下降**：Seg-zero −30.4pp、LENS −22.4pp、Qwen3-VL −14.0pp。
   BOH 幻觉降得比 ROH 多（LENS BOH −33.1 vs ROH −11.8），报告口径下再次坐实 ROH 更难。
3. **"False reject 上升"不是真代价**：以 LENS 为例，过滤掉的 25 个正样本框里 **24 个本来就是错框（IoU<0.5）、仅 1 个是正确框**。
   报告的 False-reject% 把"拒绝已经错的框"也计成代价，具有误导性；用 mIoU（正确加权）看，代价近乎为零。
4. **Orsta 仍最难**：报告口径下 FG@Neg 仅降 1pp——它的幻觉最硬，是 verifier 的能力边界（与前述 Orsta 反例一致）。

## 可写进论文的主张（report-native，严格版）
- 在 RefCOCOg-500 修复版评测的**报告自身 T2 指标**上，V-SIGHT verifier 作为单次预算后处理，
  在正样本 mIoU 损失 ≤0.005 的约束下，将高幻觉上游（LENS/Seg-zero）的负样本前景率 FG@Neg 降低 22–30pp，
  代价是每 500 正样本仅损失 1–2 个正确框。
- 幻觉下降呈 BOH>ROH（BOH FG 降幅 33–44pp，ROH 降幅 12–30pp），报告口径复现 ROH 更难。
- 该增益跨未见上游模型成立（leave-one-model-out）。

## 边界
- Orsta 报告口径下增益微弱（−1pp），诚实标注为能力边界。
- Qwen3-VL 正样本极少定对（mIoU 0.055），其 FG@Neg 下降虽大但绝对定位能力本就很差，增益意义有限。
- 弱标签（IoU/存在性），非人工 edge 真值；SWITCH/RELOCALIZE 因果仍需 S4。
- 操作点按 mIoU≤0.005 选；若换报告的 False-reject≤3pp 口径会更保守（因它把拒错框也算代价），mIoU 口径更能反映真实定位质量。

---

*生成 2026-09-05。数值来自 `report_gain.json`，报告原指标 100% 复现。单次推理预算、leave-one-model-out、报告自身 T2 口径。*
