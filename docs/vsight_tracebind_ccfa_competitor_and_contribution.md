# V-SIGHT / TRACE-Bind：CCF-A 竞品、贡献创新点与可插拔后处理模块实现路线

**版本：** v0.1
**整理时间：** 2026-08-07
**用途：** 论文 proposal / 组会讨论 / 后续实验规划
**主题：** 面向 MLLM / LVLM grounding hallucination 的 ROH–BOH binding gap、竞品定位与 TRACE-Bind 可插拔后处理模块设计

---

## 0. 一句话结论

V-SIGHT / TRACE-Bind 不应被包装成泛化的“MLLM 幻觉抑制器”，而应定位为：

> **面向属性/关系绑定错误的诊断优先后处理框架：在一次上游 MLLM grounding 输出和一次本地 detector pass 的预算内，利用冻结 detector decoder 的跨层 proposal trajectory、query-local span provenance 和 candidate permutation counterfactual，构造可审计 binding ledger，并在严格 FNR / mIoU / ROH–BOH gap 安全门槛下执行 ACCEPT / REJECT / RELOCALIZE_FLAG。**

核心差异化关键词：

- **ROH–BOH binding gap**：relation / attribute hallucination 比 object / co-occurrence hallucination 更难处理；
- **typed target–atom–reference binding**：不是泛泛 query-region binding；
- **proposal-level evidence ledger**：不是自然语言解释；
- **trajectory + permutation counterfactual**：不是 final-layer confidence 或 attention map；
- **plug-and-play post-hoc verifier**：不重训上游模型，不依赖外部大模型或数据集词表。

---

## 1. Plan / 交付门槛 / 核验

### 1.1 Plan

1. 梳理 CCF-A 相关竞品与近邻工作类别；
2. 明确 V-SIGHT / TRACE-Bind 与各类竞品的差异、威胁和应对策略；
3. 整理我们的工作贡献与论文级创新点；
4. 给出可拔插后处理 TRACE-Bind 模块的实现方向；
5. 给出实验矩阵、消融、验收指标和投稿叙事边界。

### 1.2 交付门槛

本文档需要满足：

- 覆盖 HKVLM、OpenRef MCC、MTLA、Vision Confidence Estimator、SaGe、MIRROR、Evidence Ledger 等强近邻；
- 覆盖通用 MLLM hallucination benchmark / mitigation、visual grounding、scene graph、post-hoc verifier 等竞品类别；
- 对每类竞品给出威胁等级和应对策略；
- 明确我们的贡献不能泛化过度；
- 给出 TRACE-Bind 的模块接口、阶段实现路线和验收指标。

### 1.3 核验结论

本文档定位为 **研究路线与 proposal 草案**，不是最终论文结论。文中凡涉及 TRACE-Bind 的性能提升均应视为待验证假设；在 reviewed binding data 和 nested safety gate 通过前，不应宣称已经解决 ROH。

---

## 2. 问题定义：从 object hallucination 到 typed binding hallucination

传统多模态幻觉评测常关注：

```text
图像中是否存在模型说出的 object？
```

但 V-SIGHT 的观察更细：很多 grounding hallucination 并不是目标类别不存在，而是：

```text
target instance -- attribute/action/relation atom -- reference instance
```

这条 evidence edge 没有被正确绑定。

### 2.1 BOH 与 ROH

本文建议沿用如下区分：

| 缩写 | 含义 | 典型错误 |
|---|---|---|
| **BOH** | object / co-occurrence hallucination | 图中没有该对象，或对象共现先验误导模型 |
| **ROH** | relation / attribute hallucination | 对象存在，但属性、动作、关系或 reference instance 绑定错误 |

### 2.2 核心 claim

V-SIGHT 的核心经验发现应写成：

> Across multiple upstream MLLM / grounding models, post-hoc evidence gates reduce BOH more easily than ROH, revealing a persistent relation/attribute binding gap.

中文：

> 在多个上游模型上，现有后处理 evidence gate 更容易降低 object-level false grounding，却难以安全闭合 relation / attribute binding 错误，形成稳定的 ROH–BOH gap。

这比“我们又提出一个 hallucination verifier”更有论文价值。

---

## 3. CCF-A 竞品与近邻工作矩阵

### 3.1 威胁等级定义

| 等级 | 含义 |
|---|---|
| **高** | 与 V-SIGHT 的问题、接口、证据形态或实验设置高度重叠，必须作为强 baseline 或重点 related work |
| **中高** | 解决相邻问题，可能被 reviewer 用来质疑 novelty |
| **中** | 是自然 baseline 或相关方向，但任务边界不同 |
| **低** | 主要是灵感来源或辅助组件 |

---

### 3.2 竞品总表

| 竞品类别 / 代表 | 核心做法 | 与 V-SIGHT 相似点 | V-SIGHT / TRACE-Bind 差异 | 威胁等级 | 应对策略 |
|---|---|---|---|---|---|
| **HKVLM: Faithful Query–Region Binding for Frozen-Detector Visual Grounding** | 冻结 detector，学习 query-region binding hook，并用 verifier abstain | 主题非常接近：frozen detector、faithful binding、abstention | HKVLM 偏 query-region binding；V-SIGHT 主打 target–atom–reference typed binding、ROH–BOH gap、decoder trajectory ledger | **高** | 必须作为强竞品。强调 typed edge binding、trajectory + permutation witness、gap-balanced selective action |
| **OpenRef MCC: Open-World REC + Training-free Multi-task Consistency Checker** | 多任务一致性检查，提高 referring expression comprehension 稳定性 | training-free、consistency、open-world REC 接近 | OpenRef 更偏多任务/答案层一致性；TRACE 是 same-call proposal trajectory 一致性和 binding edge witness | **高** | 对比额外调用/任务需求；强调同一次 detector pass 内的 trajectory ledger |
| **MTLA: Multi-Token Localized Attention grounding confidence** | 用 localized attention 估计 grounding confidence | 都从模型内部信号估计 grounding 可信度 | 文档已有 attention/RAFT 负结果；TRACE 不把 attention 当 contradiction signal，而用 proposal refinement trajectory | **高** | 做 attention-only、trajectory-only、attention+TRACE 消融；主张 attention 仅为 white-box ablation |
| **Evidence Ledger / provenance** | 记录支持、反证、不确定性与证据来源 | 与 TRACE-Bind ledger 概念接近 | 多数 ledger 是文本/来源级解释；TRACE 是 proposal-level typed edge ledger，绑定 action policy | **高** | 强调结构化对象：span provenance、proposal trajectory、swap witness、action gate |
| **Post-hoc verifier / abstention / confidence estimator** | 对模型输出做 accept / reject / rerank / abstain | 与 V-SIGHT 主线最接近 | 许多方法依赖 final confidence、多次调用、外部 judge 或大量 proposals；V-SIGHT 严格 single-call / one-pass | **高** | 统一预算对比；报告 FNR、mIoU、action rate、ROH/BOH gap，防止 blanket rejection |
| **Object hallucination mitigation: VCD / DoLa / OPERA / Woodpecker** | contrastive decoding、解码干预、后验检查减少 hallucinated object | 都是 hallucination mitigation | 多数处理 object/token/response-level 幻觉，不专门处理 relation/attribute binding | **高** | 用 ROH–BOH gap 证明 object-level mitigation 不等价于 typed binding verification |
| **Claim-level / token-level hallucination detection: HalLoc / ZINA / token grounding detector** | 定位回答中 hallucinated token / phrase，并可能 editing | 与 claim / span-level grounding 很近 | 多从生成文本出发；V-SIGHT 从 grounding output b0 和 proposal binding edge 出发 | **高** | 强调任务边界：grounding verifier，不是 caption editing；对比 token-level vs proposal-edge-level evidence |
| **Visual grounding / REC: GroundingDINO / GLIP / MDETR / OWL-ViT / RefCOCO(g)** | 文本短语到图像区域定位 | 都处理 text-region mapping | 它们是上游 grounding / detector；V-SIGHT 是后处理 audit / triage | **中高** | 不声称替代 grounding 模型；把它们作为 proposal backbone / baseline |
| **Scene graph / relation reasoning: SaGe / MIRROR / SGG / relation alignment** | 显式节点-边图或 language-image relation alignment | 与 ROH 的 relation/attribute binding 相邻 | 多数训练新的 scene graph / relation model；TRACE 只在 post-hoc proposal graph 上构造 ledger | **中高** | 借鉴 relation family taxonomy，不依赖 predicate vocabulary，不训练上游 graph model |
| **Teacher judge / external verifier: CLIP / Qwen / LLM A-B judge** | 用外部模型做二次裁决或 label probe | 都能做 verifier | 依赖额外模型或调用；文档已有 transfer 不稳结论 | **中高** | 放到 diagnostic / upper-bound，不作为部署主线；用成本、延迟、可审计性反击 |
| **Vision Confidence Estimator / geometry calibration** | 用置信度、框几何、校准模型判断何时可信 | 自然 cheap baseline | 更擅长 BOH，不足以解决 typed ROH binding | **中** | 作为 cheap baseline；用 ROH delta FAR <= BOH delta FAR 和 positive mIoU gate 卡住 |
| **General MLLM hallucination benchmarks: POPE / CHAIR / HallusionBench / MMHal / MME / AMBER** | 标准化评测模型幻觉 | 都评估图像是否支持模型输出 | 多为 benchmark，不是 post-hoc action policy；多为 answer/object-level | **中** | 用作外部评测和 motivation；突出 binding/action-level 细粒度 |
| **SAM / SAM2 / Florence / YOLO-World 等 proposal / segmentation 增强** | 提高 proposal、mask、detection coverage | 可提升 evidence coverage | proposal coverage 不等于 binding correctness | **中** | 单独报告 coverage；证明 coverage 提升无法自动闭合 ROH gap |

---

## 4. 强近邻逐一点评

### 4.1 HKVLM

**当前状态：** arXiv 预印本，arXiv:2606.28862，当前页面未标注正式会议。
**潜力判断：** 有 CCF-A 潜力，但不是稳中；更像机制诊断型论文。

#### HKVLM 强点

- 切口非常准：query-region binding under frozen detector；
- 有 SeeErr / SayErr 这类 diagnostic decomposition；
- 不与 end-to-end raw localization SOTA 硬拼，而定位成 mechanism-level study；
- 与 faithful grounding、abstention、hallucination mitigation 热点高度相关。

#### HKVLM 弱点

- 可能被认为是 lightweight hook + abstention；
- 主要是 query-region binding，不一定覆盖 attribute / relation typed binding；
- abstention 方法容易被质疑为“拒得更多所以更 faithful”；
- 若没有严格 report recall / coverage / false abstention / calibration / positive utility，CCF-A 风险较高。

#### V-SIGHT 应对

V-SIGHT 不能再泛泛说 query-region binding，应明确升级为：

```text
target -- attribute/action/relation atom -- reference
```

也就是 **typed target–atom–reference binding verification**。

建议对比表：

| HKVLM | V-SIGHT / TRACE-Bind |
|---|---|
| query-region binding | target–atom–reference typed binding |
| frozen detector hook | frozen decoder trajectory ledger |
| abstention | ACCEPT / REJECT / RELOCALIZE_FLAG |
| SeeErr / SayErr | ROH–BOH binding gap |
| static support / hook | cross-layer rank stability + permutation witness |
| faithful grounding | gap-balanced selective control |

---

### 4.2 OpenRef MCC

**威胁：** 高。
它与 V-SIGHT 在 open-world REC、training-free consistency、referring expression checking 上接近。

#### 差异点

| OpenRef MCC | TRACE-Bind |
|---|---|
| 多任务一致性 | 同一次 detector decoder trajectory 一致性 |
| task / answer-level consistency | proposal-edge-level binding consistency |
| 可能需要额外 task/query/check | 固定 one upstream call + one detector pass |
| rerank / consistency checker | binding ledger + selective action policy |

#### 应对策略

- 必须比较 call budget / detector forwards / latency；
- 必须证明 TRACE 的 consistency 不是多任务自洽，而是 binding edge 稳定性；
- 可以加入 MCC-style consistency 作为 baseline 或 extension。

---

### 4.3 MTLA / localized attention confidence

**威胁：** 高。
它也试图从模型内部 signal 估计 grounding confidence。

#### V-SIGHT 应对

- 明确 attention 不能作为 contradiction 证据；
- attention 只作为 white-box ablation；
- TRACE 主证据是：
  - proposal refinement trajectory；
  - rank stability；
  - stabilization layer；
  - swap persistence；
  - alternative dominance；
  - trajectory entropy。

#### 必做消融

| 实验 | 目的 |
|---|---|
| attention-only | 验证 MTLA-style baseline |
| trajectory-only | 验证 TRACE 主贡献 |
| attention + trajectory | 验证是否互补 |
| final-layer confidence | 验证 trajectory 是否超过 static score |

---

### 4.4 Evidence Ledger

**威胁：** 高。
Ledger 概念非常近，但多数 evidence ledger 更偏解释或来源记录。

#### TRACE-Bind 区别

| 通用 Evidence Ledger | TRACE-Bind Ledger |
|---|---|
| 文本 / 来源 / rationale 级证据 | proposal-level typed edge evidence |
| 解释答案为何成立 | 决定 ACCEPT / REJECT / RELOCALIZE_FLAG |
| provenance logging | span provenance + decoder trajectory + swap witness |
| 可解释性为主 | 可解释性 + action safety + gap-balanced control |

---

### 4.5 Scene graph / relation reasoning：SaGe / MIRROR

**威胁：** 中高。
它们与 relation reasoning、language-image relation alignment 接近。

#### V-SIGHT 区别

- 不训练新 scene graph model；
- 不依赖 predicate vocabulary；
- 不把 relation name 作为 inference feature；
- 只在固定候选 proposal graph 上做 post-hoc typed edge verification；
- relation family 只用于评测 strata，不作为推理词表。

---

## 5. 我们工作的贡献与创新点

### 5.1 Contribution 1：ROH–BOH Binding Gap

提出并系统验证：

> 在 grounding hallucination 中，object existence 与 attribute/relation binding 是不同层次的问题；现有 post-hoc verifier 更容易降低 BOH，而对 ROH 更弱。

建议论文表述：

> We identify a persistent ROH–BOH binding gap across diverse upstream grounding models, showing that object-level evidence gates do not safely resolve relation/attribute binding errors.

核心证据应包括：

- 多上游模型；
- base/supervised 与 RL/reasoning 分组；
- T2/T4 等不同任务；
- positive mIoU / added FNR / action rate 联合报告；
- BOH 与 ROH 的 delta 分开报告。

---

### 5.2 Contribution 2：Typed Target–Atom–Reference Binding

将 grounding hallucination 从 query-region 扩展为 typed edge：

```text
target instance -- attribute/action/relation atom -- reference instance
```

这使 V-SIGHT 与 HKVLM / generic REC 区分开。

可覆盖的 typed error：

| 类型 | 示例 |
|---|---|
| object existence | 图中不存在目标对象 |
| attribute binding | “红色杯子”中杯子存在，但红色属性属于另一个实例 |
| action binding | “正在切菜的人”中人存在，但动作不属于该人 |
| spatial relation | “桌子左边的椅子”中椅子存在，但 spatial edge 错 |
| reference binding | target 与 reference 实例绑定错 |
| ambiguous / unobservable | 图像不足以确认关系或属性 |

---

### 5.3 Contribution 3：TRACE-Bind Proposal Trajectory Ledger

提出可插拔 TRACE-Bind：

> 使用冻结 detector decoder 的跨层 proposal refinement trajectory，记录每个 candidate proposal 在不同 decoder layer 的 hidden state、bbox、span score 和 edge assignment 变化。

核心统计：

```text
rank_stability
layer_agreement
stabilization_layer
swap_persistence
alternative_dominance
trajectory_entropy
bbox_convergence
edge_uncertainty
```

核心假设：

- 正确 binding 更早稳定、rank switching 更少、entropy 更低；
- 错误 ROH binding 更容易表现出 late stabilization、alternative dominance 或 swap-sensitive restoration edge。

注意：这是待证假设，不应提前写成事实。

---

### 5.4 Contribution 4：Vocabulary-free Span Provenance

TRACE-Bind 不依赖：

- COCO-80；
- dataset category vocabulary；
- predicate vocabulary；
- model identity；
- GT / IoU；
- hallucination type label。

Query compiler 只输出：

```text
target span | modifier/action span | reference span | full query
```

每个 span 保留：

```json
{
  "raw_span": "...",
  "token_indices": [0, 1, 2],
  "role": "target|modifier|predicate|reference",
  "parser_confidence": 0.0
}
```

无法安全切分时退化为 `target-only`，不伪造 relation negative。

---

### 5.5 Contribution 5：Permutation Counterfactual Witness

在同一次 detector pass 的 candidate set 内，对 target/reference assignment 做 permutation，计算：

```text
M_swap    = score(original assignment) - score(permuted assignment)
M_restore = score(best complete alternative edge) - score(upstream edge)
```

目的不是生成无限反事实，而是在固定候选集合内找到：

- 当前 upstream edge 是否稳定；
- 是否存在更合理 alternative edge；
- 错误 binding 是否可被 permutation 恢复；
- RELOCALIZE_FLAG 是否有 evidence witness。

---

### 5.6 Contribution 6：Gap-balanced Selective Action Policy

输出动作：

```text
ACCEPT              保留 b0
REJECT              目标缺乏充分可见支持，返回 null
RELOCALIZE_FLAG     目标存在但 b0 可能绑定到错误实例；后续 router 决定是否生成新框
```

严格门槛：

| 指标 | 目标 |
|---|---:|
| 每个 model/task added FNR | <= 3 pp |
| 每个 model/task positive mIoU loss | <= 0.005 |
| router nonzero-to-zero regression | <= 1% |
| ROH delta FAR | <= BOH delta FAR |
| 新增 reject | 必须报告 action rate 和正例影响 |
| RELOCALIZE precision | 预注册阈值后报告，不能只报 coverage |

这个贡献用于防止方法通过 blanket rejection 虚假降低 FAR。

---

## 6. 可拔插 TRACE-Bind 后处理模块实现方向

### 6.1 总体接口

输入：

```python
TRACEBindInput = {
    "image": I,
    "query": q,
    "upstream_box": b0,
    "detector_candidates": [p1, p2, ..., pk],
    "detector_decoder_states": [...],
    "budget": {
        "upstream_mllm_calls": 1,
        "detector_image_encoder_forwards": 1,
        "max_candidates": 5
    }
}
```

输出：

```python
TRACEBindOutput = {
    "action": "ACCEPT|REJECT|RELOCALIZE_FLAG",
    "binding_flag": "STABLE|BINDING_UNCERTAIN|CONFLICT",
    "support_score": float,
    "edge_score": float,
    "trajectory_ledger": {...},
    "counterfactual_witness": {...},
    "latency_ms": float,
    "memory_mb": float
}
```

---

### 6.2 模块分解

```text
Upstream MLLM grounding output b0
        │
        ▼
Composite detector one-pass candidates
        │
        ▼
Query compiler
        │
        ▼
Trajectory extractor
        │
        ▼
Assignment ledger builder
        │
        ▼
Counterfactual permutation engine
        │
        ▼
Train-free diagnostics / optional edge head
        │
        ▼
Risk controller
        │
        ▼
ACCEPT / REJECT / RELOCALIZE_FLAG
```

---

### 6.3 Query Compiler

职责：

- 解析原始 query；
- 切分 target / modifier / predicate / reference；
- 保留 token offsets；
- 不使用类别词表或 predicate 词表；
- 低置信度时退化为 target-only。

建议输出：

```json
{
  "query": "the man holding a red cup next to the table",
  "spans": [
    {"role": "target", "raw_span": "the man", "token_indices": [0, 1]},
    {"role": "modifier/action", "raw_span": "holding a red cup", "token_indices": [2, 3, 4, 5]},
    {"role": "reference", "raw_span": "the table", "token_indices": [8, 9]}
  ],
  "parser_confidence": 0.86,
  "fallback": false
}
```

---

### 6.4 Trajectory Extractor

职责：

- 从冻结 detector decoder 中读取每层 proposal refinement；
- 不调用第二次 image encoder；
- 不读取 GT / IoU / hallucination type；
- 对 Top-K proposals 记录 layer-wise 状态。

建议文件：

```text
src/vsight/trajectory_ledger.py
```

核心记录：

```python
T_i_l = {
    "layer": l,
    "proposal_id": i,
    "hidden_state": h_i_l,
    "bbox": b_i_l,
    "target_span_score": s_target,
    "modifier_span_score": s_modifier,
    "predicate_span_score": s_predicate,
    "reference_span_score": s_reference,
    "object_query_id": object_query_id
}
```

---

### 6.5 Assignment Ledger Builder

职责：

- 对 target proposal、atom span、reference proposal 形成 edge；
- 每层计算 assignment score；
- 记录 top edge 的 layer-wise 变化。

建议文件：

```text
src/vsight/trace_bind.py
```

Assignment score：

```text
A_l(i, atom, j) = node_i(atom) + edge_geometry(i, j) + node_j(reference)
```

其中 `edge_geometry` 只使用归一化 box / center / size / overlap，不是 relation 词表分类器。

---

### 6.6 Counterfactual Permutation Engine

职责：

- 在同一候选集合内构造 proposal permutation；
- 比较 upstream edge 与 alternative edge；
- 形成 restoration witness。

建议 counterfactual 类型：

| 类型 | 说明 |
|---|---|
| random swap | 随机交换 target/reference candidate |
| same-class hard negative swap | 同类但非目标实例交换 |
| spatially plausible swap | 空间上合理但语义错误的候选 |
| attribute-conflict swap | 属性冲突候选 |
| reference role swap | reference candidate 替换 |
| target-reference reversal | target 与 reference 角色反转 |
| full alternative edge search | 在 Top-K 内搜索最佳 alternative edge |

---

### 6.7 Train-free Diagnostics

第一阶段只输出 ledger，不改变 action policy。

诊断指标：

```text
rank_stability
layer_agreement
stabilization_layer
swap_persistence
alternative_dominance
trajectory_entropy
bbox_convergence
edge_uncertainty
```

目标：

- 比 final-layer static score 更能区分 supported / contradicted / unobservable；
- ROH 样本相对 BOH 表现出更高 instability；
- permutation witness 对错误 binding 有可解释恢复能力。

若这一步失败，应停止 TRACE learned head，不继续调阈值硬做。

---

### 6.8 Optional Learned Edge Head

仅当 train-free ledger 与双人审核数据证明有效后才训练。

建议头结构：

| Head | 作用 |
|---|---|
| existence head | 判断目标是否有可见支持 |
| attribute binding head | 判断属性是否属于 target |
| relation edge head | 判断 target-reference relation 是否成立 |
| ambiguity / unobservable head | 判断是否证据不足 |
| restoration head | 判断是否存在更合理 alternative edge |

参数规模：约 0.1M–0.5M。
输入：ledger statistics + node/edge trajectory features。
禁止输入：model identity、GT、IoU、hallucination type、dataset category vocabulary、predicate name。

---

### 6.9 Risk Controller

职责：

- 将 edge score 转成 selective action；
- 校准 threshold；
- 保证 FNR / mIoU / ROH–BOH gap 约束。

建议策略：

```text
if support stable and low uncertainty:
    ACCEPT
elif no visible support or high contradiction confidence:
    REJECT
elif alternative edge has strong restoration witness:
    RELOCALIZE_FLAG
else:
    ACCEPT or BINDING_UNCERTAIN depending on safety budget
```

注意：不允许通过大面积拒绝正例来换取 FAR 下降。

---

## 7. 分阶段实现路线

### Phase 0：负结果与 motivation 固化

目标：把现有失败路线整理成 strong motivation。

产出：

- flat detector score failure；
- geometry / ROI / CLIP / Qwen teacher failure；
- attention / RAFT diagnostic failure；
- CABLE train-free 的 FNR/mIoU 代价；
- ROH–BOH gap across models。

交付物：

```text
failure_matrix.md
baseline_failure_table.csv
```

---

### Phase 1：100 条 TRACE-Bind train-free pilot

目标：验证 ledger 是否有诊断价值。

任务：

- 实现 `trajectory_ledger.py`；
- 实现 `trace_bind.py`；
- 跑 100 条 development pilot；
- 输出 ledger JSONL；
- 人工审计典型 case。

验收：

- latency / memory 可报告；
- ledger 字段完整；
- static vs trajectory 初步 AUROC / precision / coverage 有趋势；
- 若无趋势，停止 learned head。

---

### Phase 2：双人 binding review + static vs trajectory

目标：验证 TRACE 是否超过 final static score。

数据：

- relation 200；
- attribute 150；
- object 150；
- supported / contradicted / unobservable；
- relation atom 必须有 reference bbox；
- 双人 reference bbox IoU >= 0.5。

核心对照：

| 对照 | 问题 |
|---|---|
| static final score vs trajectory | trajectory 是否必要 |
| with / without span provenance | query role 是否必要 |
| with / without permutation | counterfactual 是否必要 |
| with / without edge geometry | 是否只是几何 |
| object-only vs attribute / relation | ROH 是否特殊 |

---

### Phase 3：0.1M–0.5M edge head

目标：在最小参数下实现有限但稳定纠偏。

验收：

- 每个 held-out model/task added FNR <= 3 pp；
- positive mIoU loss <= 0.005；
- ROH delta FAR <= BOH delta FAR；
- action rate 清晰可解释；
- RELOCALIZE precision 预注册后报告。

---

### Phase 4：跨模型 / 跨数据 / 跨 detector 泛化

目标：防止 feature engineering 只对单数据集有效。

必须报告：

- RefCOCO / RefCOCO+ / RefCOCOg；
- base/supervised vs RL/reasoning；
- T2 / T4；
- held-out upstream model；
- image-group split；
- detector backbone sensitivity；
- prompt / query format sensitivity；
- bootstrap 95% CI；
- ECE / Brier；
- p50/p95 latency 和 peak memory。

---

## 8. 必须新增的实验与消融

### 8.1 TRACE 组件消融

| 消融 | 目的 |
|---|---|
| 去掉 trajectory statistics | 验证跨层轨迹是否必要 |
| 只用 final static score | 验证 TRACE 是否超过静态分数 |
| 去掉 permutation swap | 验证 counterfactual 是否必要 |
| 去掉 span provenance | 验证 query-local role 是否必要 |
| 去掉 edge geometry | 验证是否只是几何启发式 |
| 去掉 bbox convergence | 验证 box refinement 是否有贡献 |
| head size 0.1M / 0.3M / 0.5M / 1M | 验证小 head 是否足够 |

---

### 8.2 Counterfactual 强度消融

| 强度 | 说明 |
|---|---|
| random swap | 最弱反事实 |
| same-class hard negative | 同类实例混淆 |
| spatially plausible swap | 空间合理但语义错误 |
| attribute-conflict swap | 属性冲突 |
| reference role swap | reference 错绑 |
| target-reference reversal | 角色反转 |
| full alternative edge search | 最强 restoration witness |

预期：越接近真实 binding confusion 的 counterfactual，越能暴露 ROH 错误。

---

### 8.3 Provenance fidelity 实验

| 实验 | 目的 |
|---|---|
| 删除 target span | 验证 target provenance |
| 删除 modifier/action span | 验证 atom provenance |
| 删除 reference span | 验证 reference provenance |
| 打乱 span-role 对应 | 验证 role assignment |
| random span 替换 | 排除伪相关 |
| token order shuffle | 验证 token sequence sensitivity |

---

### 8.4 强 baseline 公平比较

必须纳入：

- HKVLM-style frozen-detector query-region binding；
- OpenRef MCC-style consistency checker；
- MTLA-style localized attention confidence；
- Vision Confidence Estimator / static confidence；
- geometry MLP / ExtraTrees；
- ROI appearance / ROI cross-attention；
- CLIP / Qwen teacher；
- VCD / OPERA / DoLa 类 hallucination mitigation；
- CABLE train-free；
- final-layer static score。

统一：

- same split；
- same proposal cap；
- same detector；
- same calibration budget；
- same FNR / mIoU / gap gate；
- report action rate。

---

## 9. 投稿叙事建议

### 9.1 不建议写法

避免以下过度主张：

- “We solve MLLM grounding hallucination.”
- “TRACE-Bind closes the ROH gap.”
- “Attention / trajectory proves causal binding.”
- “Our verifier is universally better than grounding models.”
- “Lower FAR means hallucination is mitigated.”

### 9.2 推荐写法

建议主线：

> We reveal a persistent ROH–BOH binding gap in post-hoc visual grounding verification. Existing confidence, geometry, ROI, teacher, and attention-based signals reduce object-level false grounding but fail to safely resolve relation/attribute binding under positive FNR and mIoU constraints. We propose TRACE-Bind, a diagnostic-first, plug-and-play post-hoc verifier that records frozen detector decoder trajectories and candidate permutation witnesses to audit typed target–atom–reference binding. When supported by reviewed binding labels, a lightweight edge head can partially correct binding failures under gap-balanced selective action constraints.

中文：

> 我们揭示了后处理 visual grounding verification 中稳定存在的 ROH–BOH binding gap。现有 confidence、geometry、ROI、teacher 和 attention 信号可以缓解 object-level false grounding，但无法在正例 FNR 和 mIoU 安全约束下稳定解决 relation / attribute binding。为此，我们提出 TRACE-Bind：一个诊断优先、可插拔的后处理 verifier，记录冻结 detector decoder 的 proposal trajectory 和 candidate permutation witness，用于审计 typed target–atom–reference binding。在双人审核标签支持下，小型 edge head 可在 gap-balanced selective action 约束下实现有限但稳定的纠偏。

---

## 10. Related Work 建议结构

### 10.1 MLLM hallucination evaluation and mitigation

覆盖：POPE、CHAIR、HallusionBench、MMHal、MME、AMBER、VCD、DoLa、OPERA、Woodpecker。

收束句：

> These works primarily evaluate or mitigate object-level or response-level hallucination, whereas V-SIGHT focuses on post-hoc verification of typed target–atom–reference binding.

---

### 10.2 Visual grounding and referring expression comprehension

覆盖：GroundingDINO、GLIP、MDETR、OWL-ViT、RefCOCO/+/g、OpenRef、HKVLM。

收束句：

> Unlike grounding models that produce boxes from scratch, V-SIGHT audits an existing upstream grounding decision under strict call and safety budgets.

---

### 10.3 Post-hoc verification, abstention, and confidence estimation

覆盖：HKVLM、Vision Confidence Estimator、OpenRef MCC、teacher judge、self-check。

收束句：

> Existing verifiers often rely on final-layer confidence, multi-call consistency, or external judges; TRACE-Bind instead records proposal refinement trajectories and counterfactual assignment witnesses from the same detector pass.

---

### 10.4 Structured evidence, scene graph, and counterfactual reasoning

覆盖：Evidence Ledger、SaGe、MIRROR、scene graph、attention / trajectory probing、counterfactual assignment。

收束句：

> V-SIGHT differs by tying structured evidence to selective actions and gap-balanced safety constraints, rather than using evidence only as explanation.

---

## 11. 下一步行动清单

### 11.1 本周可做

- [ ] 完成 HKVLM / OpenRef MCC / MTLA 三个强近邻的细读表；
- [ ] 把现有负结果整理成 `failure_matrix.md`；
- [ ] 实现最小 `trajectory_ledger.py`；
- [ ] 跑 100 条 train-free TRACE pilot；
- [ ] 输出 10 个可视化 case：5 个 supported，5 个 contradicted / ambiguous。

### 11.2 两周内可做

- [ ] 建立 500 image-group Stage-B review queue；
- [ ] 制定 supported / contradicted / unobservable 标注 guideline；
- [ ] 完成 static vs trajectory 的初步 AUROC / AUPRC / risk-coverage；
- [ ] 做 span provenance 和 permutation ablation；
- [ ] 决定是否进入 learned edge head。

### 11.3 一个月目标

- [ ] 训练 0.1M–0.5M edge head；
- [ ] 完成 nested held-out model + image fold evaluation；
- [ ] 形成主表：overall / T2 / T4 / BOH / ROH / base / RL；
- [ ] 输出 latency / memory / action rate；
- [ ] 准备 proposal 图和 related work 草稿。

---

## 12. 最终判断

V-SIGHT / TRACE-Bind 有 CCF-A 潜力，但前提是叙事必须收束：

> **不要做泛化 hallucination mitigation；要做 ROH-aware typed binding verification。**

HKVLM 已经占住了 query-region binding 这一近邻位置，因此我们的差异化必须更尖：

```text
HKVLM: query-region binding under frozen detector
V-SIGHT: typed target-atom-reference binding under ROH–BOH gap constraints
TRACE-Bind: decoder trajectory + permutation counterfactual ledger for plug-and-play post-hoc verification
```

最重要的实验不是追求单一 FAR 下降，而是证明：

1. ROH–BOH gap 是跨模型稳定存在的；
2. final-layer confidence / geometry / ROI / teacher / attention 不能安全闭合该 gap；
3. trajectory + permutation ledger 比 static score 更能识别 typed binding error；
4. learned edge head 在严格 FNR / mIoU / gap gate 下提供有限但稳定的改进；
5. 该模块可拔插、低成本、可审计，并不依赖上游模型重训或外部大模型。

如果这五点成立，V-SIGHT / TRACE-Bind 可以形成一条比普通 benchmark 或 verifier 更有区分度的 CCF-A 路线。
