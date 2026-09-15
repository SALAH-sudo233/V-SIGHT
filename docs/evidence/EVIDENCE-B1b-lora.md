# 证据资产：B1b LoRA 训练版 verifier — 拉高 ROH 捕获（vs B1a 融合对照）

> 问题：弱标签 LoRA 训练版能否修复零训练 verifier 的置信度饱和、在 FNR≤3pp 下把 **ROH 捕获**再抬高？
> **结论：能，且在两个高幻觉主力上游（LENS/Seg-zero）上 FNR≤3pp 的 ROH 捕获比 B1a 融合近乎翻倍；
> 置信度由构造变连续（饱和问题解决）。但存在一个上游（Orsta）AUROC 塌陷的诚实反例，需诊断。**

方法：Qwen2.5-VL-7B LoRA（r=8, q/k/v/o_proj, 5.0M 可训练参数 = 0.06%），训练单框绑定判定（红框画在图上 + query → yes/no）。
**训练数据**：train/calibration split 的 e2b reference candidates（10,507 弱标签样本 = 4666 正[GT target 框] + 5841 负[低-IoU GroundingDINO 提案 3364 + jitter GT 2477]），**与 500-dev 完全隔离，零人工**。1 epoch，loss 0.67→0.13，~47min。
**推理**：读首 token 的 `P(yes)` logit softmax → **连续**置信度（这正是修复 S5 二值饱和的关键）。
脚本：`b1b_prep_data.py` / `b1b_train_lora.py` / `b1b_eval.py`。评测同 S5/B1a 口径（4 上游模型真实 T2 框）。

---

## 1. 头对头：B1b LoRA vs B1a 融合（FNR≤3pp，out-of-sample）

| 上游模型 | 指标 | B1a 融合 | **B1b LoRA** | 结论 |
|---|---|---|---|---|
| LENS | AUROC BOH/ROH | 0.889/0.749 | 0.855/0.739 | 相当 |
| | catch@3pp BOH/ROH | 0.335/0.114 | **0.509/0.237** | **B1b（ROH ~2×）** |
| Seg-zero | AUROC BOH/ROH | 0.891/0.742 | 0.844/0.725 | 相当 |
| | catch@3pp BOH/ROH | 0.406/0.169 | **0.526/0.256** | **B1b（ROH ~1.5×）** |
| Qwen3-VL-8B | AUROC BOH/ROH | 0.860/0.799 | 0.863/0.833 | B1b 略优 |
| | catch@3pp BOH/ROH | 0.680/0.600 | 0.610/0.509 | B1a 略优（n_correct 仅 12，不稳） |
| Orsta-7B | AUROC BOH/ROH | 0.792/0.701 | **0.565/0.553** | **B1a（B1b 塌陷，见 §3）** |

**核心正向结果**：在两个高幻觉主力上游（LENS/Seg-zero，n_correct 248/275，统计稳）上，
B1b 在严格 FNR≤3pp 门下 **BOH 捕获 0.51–0.53、ROH 捕获 0.24–0.26**，
ROH 捕获相对 B1a（0.11/0.17）**近乎翻倍**——正是"训练版拉高 ROH"的预期增量。

FNR≤10pp 时 B1b LENS/Seg-zero：BOH 0.63–0.65、ROH 0.38–0.40。

## 2. 置信度饱和问题已解决

零训练 verifier 的 confidence 是二值 `{0.0, 1.0}`（S5 §4），没有中间操作点。
B1b 读 logit softmax 得到**连续** P(correct)：实测 500-dev 上分布平滑（min 1e-5、mean 0.83、max 0.999，取值密集连续）。
这是训练版最直接、可写进论文的机制贡献：**LoRA 把判别信息重新分布到可校准的连续置信度上**。

## 3. 诚实反例：Orsta-7B 上 B1b AUROC 塌陷到 ~0.56

- B1b 在 Orsta 上 AUROC 0.565/0.553（近随机），而 B1a 融合有 0.792/0.701。这是明确的负信号，不掩盖。
- **几何分布已排除**：实测 Orsta 预测框的 area 中位数 48384 / aspect 0.86，与 LENS（44604 / 0.89）几乎相同，
  也接近训练负例（area 34079 / aspect 0.99）。**所以不是框尺寸/形态分布偏移导致的塌陷。**
- **更可能的原因**（待进一步验证）：Orsta 的幻觉是**语义型**（画了真实存在但语义错误的物体），
  与训练负例（低-IoU 空间错位 / jitter）的错误**类型**不同；LoRA 学到的是"框住错位置"的判别，
  对"框住了错的正确物体"不迁移。B1a 融合靠跨模型 CV + 线性校准对这种类型偏移更鲁棒。
- **下一步**：(a) 训练负例混入真实上游语义幻觉框（留其他模型、测 Orsta）；(b) B1a+B1b 集成（线性校准 + LoRA 连续分）取两者之长。

## 4. 可写进论文的主张（严格版）

1. **训练版在主力上游拉高 ROH 捕获**：LENS/Seg-zero 上 FNR≤3pp ROH 捕获相对零训练融合近乎翻倍（0.11/0.17→0.24/0.26），单次预算。
2. **修复置信度饱和**：LoRA 输出连续可校准置信度，机制上解决零训练 verifier 的离散瓶颈。
3. **诚实局限**：B1b 对训练负例分布敏感，在 Orsta 上 AUROC 塌陷——训练版并非无条件优于零训练融合；
   分布匹配 / 集成是明确改进方向。**这使"零训练融合(B1a) vs 训练(B1b)"成为论文里一个干净的、有正有负的对照**，而非过度宣称。

## 5. 边界
- 弱标签（IoU/存在性代理，非人工 edge 真值）；SWITCH/RELOCALIZE 因果正确性仍需 S4 复核。
- Qwen3-VL n_correct=12 统计不稳；主结论以 LENS/Seg-zero 为准。
- 单 epoch、r=8、未调超参；B1b 仍有上探空间（更大 rank / 负例分布对齐）。

---

*生成 2026-09-05。数值来自 `b1b_eval_analysis.json`（4 上游模型）对照 `b1a_fusion_cv.json`。
训练零人工弱标签、与 500-dev 隔离、单次推理预算。LoRA 5.0M 参数（0.06%）。*
