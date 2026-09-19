# 证据资产：verifier 即插即用过滤真实上游模型的幻觉框（S5-plugin，Gate B 主结果雏形）

> 问题：V-SIGHT 的 VLM verifier 作为**即插即用**后处理，作用在真实上游 grounding 模型的输出框上，
> 能否在**单次推理预算**下砍掉幻觉框、提升接受集精度，并复现 ROH>BOH 难度结构？
> **结论：判别信号强且 ROH>BOH 在 4 个真实上游模型全部成立（correct-vs-halluc AUROC 0.57–0.88）；
> 动作层对高幻觉上游把接受集精度提升近一倍（+8.7–13.5pp）；但零训练 verifier 的置信度二值饱和，
> 严格安全门 FNR≤3pp 达不到（实际 12–16pp）——这是训练/融合(B1a/B1b)要解决的精确问题，不是负结果。**

数据：11-model 评测的真实 T2 预测框（`benchmark/.../run_500_semantic_strict/*/records.jsonl`，`pred_bbox_xyxy`/`iou`/`pred_found`）。
方法：对每个"上游画了框"的样本，把**上游预测框**画在图上，问 Qwen2.5-VL graded yes/no（S2b 同款，1 次调用/框）。
脚本：`s5_plugin_filter.py` → `s5_<model>.jsonl`；分析 `s5_analysis.py`、`b1a_calibrate.py`。

---

## 1. 上游模型的幻觉画像（为什么需要 verifier）

500-dev 上，2000 个负样本（目标不该存在）中上游仍画框（=幻觉）的数量：

| 上游模型 | 负样本画框数(幻觉) | 正样本定对(IoU≥.5) | 上游接受集精度 |
|---|---:|---:|---:|
| LENS | 1884 / 2000 | 248 / 500 | 0.105 |
| Seg-zero | ~1450 | 275 / 500 | 0.113 |
| Orsta-7B | 883 / 2000 | 151 / 500 | 0.111 |
| Qwen3-VL-8B | 420 / 2000 | 12 / 500 | 0.015 |

LENS 是极端"高召回、狂幻觉"上游（94% 负样本照样画框）；Qwen3-VL 相对克制。**上游接受集精度普遍很低（1.5%–11%）**——这正是 verifier 要修的病。

---

## 2. 核心结果（阈值无关）：verifier 分数干净区分正确框 vs 幻觉框，且 ROH>BOH

对"正确正样本框"打分 vs "幻觉负样本框"打分的 separation AUROC（threshold-free）：

| 上游模型 | AUROC 正确 vs BOH幻觉 | AUROC 正确 vs ROH幻觉 | **gap(BOH−ROH)** |
|---|---:|---:|---:|
| LENS | **0.877** | 0.740 | 0.137 |
| Seg-zero | **0.859** | 0.730 | 0.129 |
| Orsta-7B | 0.704 | 0.639 | 0.065 |
| Qwen3-VL-8B | 0.625 | 0.566 | 0.059 |

**四个真实上游模型 gap 全部为正**——即插即用过滤 + ROH>BOH 难度结构在真实上游输出层复现，
与 verifier 级 Gate A（gap 0.11–0.12）、model 级评测（gap 11–34pp）形成**三层一致证据**。

---

## 3. 动作层结果：接受集精度提升（verdict-based 决策）

决策规则用 **verdict 本身**（`no`→REJECT，`yes`→KEEP），而非 confidence 加权分数（原因见 §4）。
`no_only`（只拒 no）操作点：

| 上游模型 | 捕获BOH | 捕获ROH | 正例保留 | 加性FNR | 接受集精度 前→后 | 增益 |
|---|---:|---:|---:|---:|---:|---:|
| LENS | 0.729 | 0.530 | 0.883 | 0.117 | 0.105 → 0.194 | **+0.089** |
| Seg-zero | 0.698 | 0.523 | 0.862 | 0.138 | 0.113 → 0.199 | **+0.087** |
| Orsta-7B | 0.401 | 0.338 | 0.881 | 0.119 | 0.111 → 0.138 | +0.027 |
| Qwen3-VL-8B | 0.320 | 0.287 | 0.583 | 0.417 | 0.014 → 0.011 | −0.003 |

`no+unclear`（更激进）：LENS 精度 0.105→0.240（+0.135）、Seg-zero →0.248（+0.135），捕获更高但 FNR 升到 14–16pp。
**捕获率 ROH<BOH 在每个模型都成立**（LENS gap 0.20、Seg-zero 0.18），再次复现难度结构。
对两个高幻觉上游，单次预算过滤把接受集精度提升近一倍。

---

## 4. 诚实边界：零训练 verifier 置信度二值饱和 → FNR≤3pp 门达不到（B1a/B1b 动机）

**根因（已验证）**：Qwen2.5-VL 返回的 confidence 几乎二值——`{0.0: 1308, 1.0: 844, 0.8: 182}`（LENS）。
verdict `no` 有 1242 个（≈幻觉数），但模型给 `no` 常配 confidence 0.0，使 `score=−conf=0` 落进 KEEP 区；`unclear` 也归 0。
**所以 confidence 加权分数没有可用的中间操作点**：要么 FNR≈0/捕获≈10%，要么 FNR≈14–19%/捕获≈74%，中间是空的（B1a 效率前沿已实测证实）。

- 这是零训练信号的**固有属性**，不是 bug：verdict 判别力强（AUROC 0.74–0.88），但**置信度不校准、离散**。
- **对 Gate B FNR≤3pp 安全门**：零训练 verifier 达不到（实际操作在 FNR 12–16pp）。这正是训练/融合要解决的：
  - **B1a 融合**：verdict + 几何（候选面积/位置/best-of-5 margin）+ 上游置信 做逻辑回归，产出**连续**校准分数，
    才可能在 FNR≤3pp 下提升捕获（阈值无关 AUROC 0.88 说明上限存在）。
  - **B1b LoRA**：晚层 LoRA + graded 头，直接学出校准良好的连续置信。
- **Qwen3-VL 头部空间小**：少画幻觉框、正确正样本仅 12 个，增益不稳；主结论以高幻觉上游（LENS/Seg-zero）为准。
- **正样本"正确"= IoU≥0.5**；换 0.75 门改变绝对精度但不改可分性结论。

---

## 5. 可写进论文的主张（严格版）

1. **即插即用幻觉过滤在真实上游模型上成立**：V-SIGHT verifier 单次预算作用于 4 个上游 grounding 模型的输出，
   correct-vs-halluc 分数 AUROC 0.57–0.88，**全部模型 ROH<BOH 可分性**（gap 0.06–0.14）。
2. **对高幻觉上游收益显著**：verdict 过滤把接受集精度提升近一倍（LENS 0.105→0.194、Seg-zero 0.113→0.199），保留率 86–88%。
3. **ROH 系统性更难在部署层复现**：捕获率 ROH<BOH 每模型成立，与 verifier 级、model 级证据三层一致，坐实贡献一。
4. **零训练 verifier 的天花板 = 置信度校准**：判别信号足够（AUROC 0.88），但离散饱和置信度使 FNR≤3pp 安全门达不到。
   这是 B1a 融合 / B1b LoRA 的精确目标，也是训练版相对零训练版的**明确增量假设**（可证伪）。

---

*生成 2026-09-05，更新 verdict-based 动作层 + 饱和根因。数值来自 `s5_LENS/Segzero/Orsta/Qwen3VL.jsonl`
（LENS 2371、Orsta 1364、Qwen3-VL 830、Seg-zero 2442 verifier calls，全部完成）、`s5_analysis.json`、`b1a_calibrate.json`。
预算：每框 1 次 verifier 调用（单次推理预算）。*
