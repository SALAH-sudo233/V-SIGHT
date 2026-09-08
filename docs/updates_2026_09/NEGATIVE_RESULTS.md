# V-SIGHT 负结果归档（2026-09）

集中记录本项目已证否的路线与诚实负结果。这些不是失败，而是**排除了错误方向、支撑核心命题**的证据，
是论文 limitation / ablation 章节的资产。过程性代码与逐样本产物已归档（仓库 git 历史 + 服务器
`experiments/_archive/`），此处只留结论。

---

## 1. decoder-trajectory TRACE — 近随机（弃用为诊断性 ablation）

- **假设**：VLM 解码 bbox 时的 decoder trajectory（token 轨迹/注意力流）携带绑定正确性信号。
- **结果**：trajectory-only separation AUROC ≈ **0.5009**（近随机）；trajectory + swaps 落后于静态 typed geometry，
  grouped-bootstrap CI 下界 < 0。静态最终层 typed geometry 是当时最强诊断（AUROC 0.6135），但也不足以支撑方法。
- **结论**：TRACE 保留为诊断性负结果与匹配 ablation；不在 proxy 上拟合任何 TRACE 阈值或学习式 TRACE 头。

## 2. train-free 注意力信号 — AUROC<0.5（对 MTLA 类方法的 rebuttal）

- **假设**：从 VLM 内部注意力（RAFT transport 类）免费提取信号即可判别绑定。
- **结果**：4 个 RAFT 注意力信号（top_transport / neg_edge_uncertainty / decoder_top_transport /
  transport_relative_margin）对 chosen>rejected 的 separation AUROC **全部 < 0.5**（0.23–0.31，与绑定正确性
  系统性反相关）；500-dev 上 0.51–0.61 ≈ 随机。即便 oracle 符号翻转，|AUROC−0.5| 也仅 0.19–0.27，仍低于 VLM verifier。
- **结论**：作为即插即用 train-free 信号，注意力抽取**直接失败**。这是对"利用 token/attention 内证据"
  （MTLA 类）方法在 grounding-bbox 任务上的正面 rebuttal——输出是坐标而非语言 token，机制错位。

## 3. 学习式 MLP verifier（早期 E2）— 未过迁移门

- **假设**：在候选特征（几何 + train-free 信号）上训一个小 MLP/CLIP/Qwen-LoRA selector 即可判别绑定。
- **结果**：CLIP 与 Qwen-LoRA candidate verifier 均未通过 oracle-gap 与 repaired-500 迁移门。
- **根因（2026-09 侦察确认）**：不是 MLP 不行，而是**喂给它的便宜信号本身对 ROH 无判别力**——检测器一阶信号
  对 BOH AUROC 0.63–0.71、对 ROH 仅 0.51–0.58（≈随机）。ROH 内在需要 VLM 级语义推理。
- **结论**：便宜特征 + 小 head 的路线对 ROH 有天花板（~0.55）；这直接导出了"ROH 必须 VLM、BOH 可便宜"的分层设计。
- **注**：2026-09 的弱标签 LoRA verifier（`b1b_*`，微调 3B/7B VLM）是**不同设定**，已在真实上游模型输出上取得正向增益，
  不与本条冲突——区别正是"小 head on 便宜信号" vs "微调 VLM 语义推理器"。

## 4. B1b-v2 负例工程重训 — 整体退化（v1 配方更优）

- **假设**：Orsta 上 verifier 塌陷是因为训练负例是空间型（低-IoU/jitter），补入真实语义幻觉负例可救。
- **结果**：加入 8 个其他上游模型的真实语义幻觉框做负例重训（v1→v2），**整体退化**：
  LENS ROH catch@FNR≤3pp 0.237→0.112、Seg-zero 0.256→0.107、Qwen3-VL 0.509→0.128；Orsta 仍未救回
  （AUROC 0.565→0.597，仍近随机）。
- **结论**：混入异质含噪的真实幻觉负例**稀释**了干净的空间信号。**v1 空间负例是更优训练配方**；
  Orsta 的难度是内在的，只有 B1c 集成（借 B1a 特征）能部分救（AUROC 0.565→0.714）。防止了"负例工程"错误方向。

## 5. CCV / CABLE train-free 动作策略 — 未安全解决关系绑定

- CCV/CABLE 提供 typed claim parser、composite 单次检测器、固定 proposal 池、原子/边 ledger、校准与安全指标，
  作为**实现资产**保留；但其 **train-free 动作策略未安全解决关系绑定**，未产出可宣称的学习式结果。
- 真正兑现 ROH 增益的是"把上游框画在图上问 VLM"这一语义验证步骤，而非围绕它的 CCV 结构本身。

---

## 与核心提案的张力（诚实定位）

上述负结果共同推翻了初始提案的一个**方法论假设**——"用便宜的 train-free、单次推理预算的后处理即可解决 ROH"：

| 原始承诺 | 实验现实 | 判定 |
|---|---|---|
| 单次推理预算 | ROH 必须第二次 VLM 前向（便宜信号做不到） | 相悖 |
| train-free / 低成本 | ROH 需 VLM 语义推理 + LoRA；train-free 注意力/CCV/MLP 均负结果 | 相悖 |
| grounding ≠ hallucination mitigation | 三层证据 + 因果机制成立 | **加强** |
| ROH 系统性比 BOH 难 | 成立且有训练范式层面因果解释 | **加强** |

**核心科学命题被证实并加强**；被推翻的是"免费信号绕过 ROH"的工程假设。据此把方法重新定位为：
**一次可控的额外 VLM 验证（3B，分层触发：BOH 走便宜检测器、仅 ROH 升级），选择性修复 RL 上游被
"正样本-only 训练"牺牲的拒答与绑定能力；飞轮为 human-in-the-loop 半自动数据引擎，而非全自动持续改进。**

---

## 归档位置（可恢复）

- 仓库：过程脚本经 `git rm` 删除，Git 历史永久保留（`b1a_calibrate.py`、`b1b_prep_v2_realneg.py`）。
- 服务器：`experiments/_archive/<timestamp>/`（scripts / process_jsonl / lora，34MB），移动而非硬删。
- 各负结果的结论数值仍在 `experiments/roh_boh/results/*.json`（如 `b1b_v2_eval_analysis.json`、`s3_500.summary.json`）。
