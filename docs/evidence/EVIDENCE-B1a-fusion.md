# 证据资产：B1a 融合校准 — 打破 FNR≤3pp 安全门（Gate B 判据 2 可达）

> 问题：S5 显示零训练 verifier 的置信度二值饱和，严格安全门 FNR≤3pp 下捕获仅 ~10%。
> 用极少信号融合校准，能否在 FNR≤3pp 下把捕获率拉起来，且**跨未见上游模型**成立（真·即插即用）？
> **结论：能。leave-one-model-out 融合在 4 个留出上游模型上 AUROC 全面超过 raw verdict，
> 并在 FNR≤3pp 下把 BOH 捕获从 ~10% 拉到 30–68%（ROH 11–60%）。Gate B 判据 2 变为可达。**

方法：逻辑回归融合 [verdict one-hot + confidence + vlm_score + 框几何(logaspect/logarea/尺度)] → P(框真正确)。
**弱标签、零人工**：y=1 当且仅当 `label_exists ∧ iou≥0.5`（存在+定位都对）。
**留一模型交叉验证**：在 N−1 个上游模型上训练融合权重，在**留出的第 N 个**上测试——直接检验"校准能否迁移到未见上游模型"这个即插即用主张。
脚本：`b1a_fusion_cv.py` → `b1a_fusion_cv.json`。单次推理预算不变（融合特征都在预算内、无额外前向）。

---

## 1. 融合 vs raw verdict（correct-vs-halluc separation AUROC，留出模型上）

| 留出上游模型 | 融合 AUROC BOH | raw BOH | 融合 AUROC ROH | raw ROH |
|---|---:|---:|---:|---:|
| LENS | **0.889** | 0.877 | **0.749** | 0.740 |
| Orsta-7B | **0.792** | 0.704 | **0.701** | 0.639 |
| Qwen3-VL-8B | **0.860** | 0.625 | **0.799** | 0.566 |
| Seg-zero | **0.891** | 0.856 | **0.742** | 0.728 |

**四个留出模型融合 AUROC 全部 ≥ raw**，Qwen3-VL 提升最大（BOH +0.235、ROH +0.233）——
说明几何+校准把 raw verdict 饱和丢掉的信息补了回来，且这套校准**能迁移到训练时未见过的上游模型**。
ROH<BOH 难度结构在融合后依然保持。

## 2. 关键：打破 FNR≤3pp 安全门

raw verdict 在 FNR≤3pp 下捕获仅 ~7–10%（S5 §4）。融合后（留出模型，out-of-sample）：

| 留出上游模型 | catch@FNR≤3pp (BOH/ROH) | catch@FNR≤5pp | catch@FNR≤10pp |
|---|---|---|---|
| LENS | 0.335 / 0.114 | 0.457 / 0.219 | 0.707 / 0.481 |
| Orsta-7B | 0.298 / 0.219 | 0.337 / 0.245 | 0.398 / 0.318 |
| Qwen3-VL-8B | **0.680 / 0.600** | 0.680 / 0.600 | 0.740 / 0.628 |
| Seg-zero | 0.406 / 0.169 | 0.550 / 0.298 | 0.706 / 0.467 |

**在严格 FNR≤3pp 门下，BOH 捕获从 raw 的 ~10% 提升到 30–68%**（真实 out-of-sample FNR 全部 ≤3pp）。
放宽到 FNR≤10pp，BOH 捕获 40–74%、ROH 32–63%。**Gate B 判据 2（安全门 FNR≤3pp）从"达不到"变为"可达"。**

## 3. 可写进论文的主张（严格版）

1. **零训练 verifier + 轻量融合校准 = 即插即用可用**：leave-one-model-out 融合在 4 个未见上游模型上
   AUROC 全面超过 raw verdict，证明校准可迁移（非过拟合到特定上游）。
2. **打破安全门**：FNR≤3pp 下 BOH 捕获 30–68%、ROH 11–60%（raw 仅 ~10%），单次推理预算不变。
3. **成本几乎为零**：融合特征（verdict/几何）都在预算内，逻辑回归 CPU 秒级训练，弱标签零人工。
4. **ROH<BOH 结构一致保持**：融合后 ROH 捕获仍系统性低于 BOH，坐实贡献一，也标出 B1b LoRA 的靶心（拉高 ROH）。

## 4. 诚实边界

- **弱标签**：y 用 `label_exists ∧ iou≥0.5`，即"存在正确+定位正确"，是代理标签而非人工裁定的 edge truth（后者要 S4 reference box）。
  这对 accept/reject 层是干净的；SWITCH/RELOCALIZE 的因果正确性仍需 S4。
- **几何特征弱**：只用了框自身的 aspect/area（无图像尺寸、无候选池 best-of-5 margin）。加入 GroundingDINO 候选特征应能再提升——留作 B1b/消融。
- **Qwen3-VL 测试样本少**（正确正样本 12），其 0.68/0.60 的高捕获统计不稳；稳健结论以 LENS/Seg-zero（n_test 2371/2442）为准。
- **仍非训练版天花板**：融合是线性校准。B1b LoRA（学连续 graded 置信 + ROH 难例）目标是进一步抬高 ROH 捕获，这是训练相对融合的可证伪增量假设。

---

*生成 2026-09-05。数值来自 `b1a_fusion_cv.json`（leave-one-model-out，4 上游模型，n_test 830–2442）。
弱标签零人工、CPU 训练、单次推理预算。B1b LoRA 训练管线并行铺设中。*
