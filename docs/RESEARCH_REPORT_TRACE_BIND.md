# V-SIGHT：开放词汇 Grounding 中的 Wrong-Instance Binding 因果诊断与选择性修复

**面向导师汇报的研究叙事**

**日期：** 2026-08-07
**状态：** 多模型问题诊断和 500 组 TRACE development proxy 已完成；TRACE 未通过 static baseline promotion gate。自 2026-08-10 起，binding truth 改用项目负责人单人权威审核（`single_project_owner`），随后进入 oracle attribution 和 selective relocalization 准备阶段；此前双人方案已废止但历史实验结果不改写。

## 核心故事

RL/reasoning grounding 模型可以提升正样本上的 IoU，但单看正样本 IoU 并不能说明模型可靠：它仍可能接受本应弃权的负样本，或在同类实例之间发生视觉漂移。这里没有证据证明 RL 因果性地引入了幻觉；可以确认的是，IoU 优化没有自动带来负样本拒答和实例绑定可靠性。

本项目不试图再训练一个更大的视觉模型去替代上游模型，而是把上游 RL 模型视为黑盒，加入一个**可插拔 verifier**：

```text
image + query + upstream bbox
        -> target/reference/atom evidence and oracle attribution
        -> ACCEPT | ABSTAIN | RELOCALIZE
```

它的目标是补上 IoU-only 优化没有覆盖的两件事：

1. 区分目标未覆盖、目标选错、reference 选错和 relation composition 失败；
2. 对不存在、不可观测或无法安全恢复的表达式弃权；
3. 对目标存在且正确候选已在 Top-K 的 visual drift，进行风险受控的重新定位。

这里的关键不是把 reject 数量做大，而是在降低负样本接受的同时，保持正样本定位精度。因此主张始终是“**系统级可靠性与有效 IoU 的提升**”，而不是声称 verifier 让上游模型本身获得了新的视觉能力。

## 1. 为什么 IoU 提升本身不够

Grounding 输出的是一串 bbox 坐标。若只在正样本上统计 IoU，模型可以在“会框出正确目标”的子集上表现很好，但仍在以下场景失败：

- 图中没有完整目标，却输出一个看似合理的框；
- 图中有目标类别，却框到不满足属性、动作或关系的同类实例；
- 上游框本来正确，但后续候选替换把它换成零 IoU 框。

这使得任务天然是一个受约束的 trade-off，而不是单指标排序：

| 只优化什么 | 可能得到的表面结果 | 被忽略的代价 |
|---|---|---|
| 正样本 IoU | 框的位置更准 | 负样本仍可能被接受，IoU 不能反映 hallucination |
| 幻觉率/FAR | 大量负样本被拒绝 | 正样本也可能被错拒，FNR 上升、mIoU 下降 |
| 候选替换 | 一部分 IoU=0 被修复 | 错误换框会产生 nonzero-to-zero regression |

因此，真正有效的方法必须同时报告负样本 false acceptance、added FNR、positive mIoU 和换框回归，不能用单个 IoU 或单个 hallucination rate 代替全部结论。

### 多模型证据：高定位能力与幻觉缓释并不等价

修复版 RefCOCOg-500 包含 11 个比较模型、一个 ROH-VCD 历史对照、500 个 image groups 和 90,000 条完整 records。下面几行并非模型排名，而是说明不同指标可以指向完全不同的行为：

| 代表模型 | T1 HR | T1 FNR | T2 positive mIoU | T2 FG@Neg | 读法 |
|---|---:|---:|---:|---:|---|
| Seg-zero | 44.8% | 3.4% | 0.5222 | 97.1% | 定位 IoU 高，但几乎不会过滤负样本 |
| Seg-R1 | 20.8% | 22.0% | 0.4976 | 61.8% | 降低接受率伴随较高错拒 |
| TreeVGR | 45.8% | 1.0% | 0.2811 | 75.6% | 很少拒答，但接受大量负样本 |
| UniVG-R1 | 23.2% | 56.0% | 0.1122 | 99.8% | 低 HR 主要靠大量正例拒绝，仍未缓解负样本接受 |

这正是本项目的出发点：**正样本 IoU 的提升不能自动解释为 grounding 更可信；hallucination 缓释必须是独立目标。**

## 2. RL/reasoning（CoT-style）模型带来的现实缺口

我们的主对标对象是 RL/reasoning 检测/分割模型。它们往往针对推理或正样本定位进行了强化，但开发评测显示，它们仍存在明显的负样本接受与视觉漂移问题。E3 将 11 个模型分为 `base/supervised`（3 个）和 `RL/reasoning`（8 个，作为 CoT-style 工程代理）进行统一评估：

| 模型族 | 任务 | 原始 FAR | verifier FAR delta | mIoU delta | added FNR | ROH-BOH gap delta |
|---|---|---:|---:|---:|---:|---:|
| base/supervised | T2 | 49.90% | -2.85 pp | -0.00119 | +0.40 pp | +3.17 pp |
| RL/reasoning | T2 | 65.66% | -3.78 pp | -0.00108 | +0.43 pp | +4.06 pp |
| base/supervised | T4 | 28.27% | -1.07 pp | -0.00169 | +0.47 pp | +0.40 pp |
| RL/reasoning | T4 | 51.18% | -1.94 pp | -0.00065 | +0.30 pp | +1.31 pp |

可以确认的是：在这套开发集上，reasoning/CoT-style 组的负样本接受率更高；同一个轻量 verifier 对它们有更大的总体 FAR 降幅，但主要处理的是简单的 BOH，ROH-BOH gap 仍在扩大。

不能确认的是“CoT 一定导致幻觉”。模型的架构、训练数据、输出协议和 RL 目标都同时变化，因此这里只有**分层现象**，没有 CoT 的因果结论。更稳妥的结论是：CoT 或 reasoning 文本并没有自动保证 bbox 对应的视觉 binding 正确；可信 caption/reasoning 也不能替代 bbox 验证。

这也是即插即用 verifier 的意义：不需要知道上游是否使用 CoT，也不把模型身份当作 feature，而是直接验证它输出的视觉 claim。

## 3. 已发现的四个实验问题

### 3.1 IoU=0 多数是对象/实例幻觉引起的定位偏移

对 valid-box IoU=0 的 114 个 development groups 的审计中，85 个被标为 same-category wrong-instance confusion，93 个具有高 instance-confusion risk，111 个是 relation expression。也就是说，很多“定位失败”并非模型完全看不到对象，而是看到了同类对象，却把它当成 query 所指的实例。

多 prompt 和候选池能一定程度上提高“正确框出现在候选中”的概率。历史的 state-preserving candidate policy 也确实能修复一部分零 IoU 案例。但它有两个根本限制：

1. 每增加 prompt、候选或视觉 pass 都会增加推理成本；对本项目的 RL/reasoning 对标模型，这个成本尤其不能忽略；
2. 候选池只是列出更多可能性，本身不判断原 query 是否为 hallucination，也不能可靠决定应该拒答还是换框。

因此候选池是 visual-drift 修复的**后半段工具**，不是 hallucination mitigation 的完整解法。

### 3.2 上游 logit 没有提供可直接使用的属性/空间证据

我们尝试过从 MLLM/detector 的中间 logit、attention 和 role masking 中抽取属性或空间关系的直接信号。现有 pilot 没有得到可安全使用的结果：decoder causal masking 的完整正 causal edge rate 约为 1%，而 predicate masking 后原 token logit 经常反向上升。

这不等于任何模型的 logit 都没有视觉信息；更准确地说，**在当前 grounding 输出链路中，没有发现一个可校准、可直接用于属性/空间 binding 的 logit readout**。因此对比解码、参数矩阵修正、向量加权等依赖语言 token 信号的干预，不能直接作为本任务的可靠解法。

解法自然转向后处理：不修改上游参数，也不假设内部语言 token 能解释 bbox，而是验证最终视觉 claim 是否有局部证据支持。

### 3.3 caption/CoT 与 bbox 不是可直接互换的监督信号

Grounding 最终输出是坐标，而非自然语言答案。许多 VLM 幻觉工作通过干预注意力或语言解码来影响最后一个 token；这类机制直接迁移到 bbox 任务时存在机制错位。

我们的 T4 结果表明，caption 命中和 bbox 正确不是同一事件。多个模型的 caption-grounding coupling gap 很大，例如 qwen2.5-vl-7b 为 29.8 pp、VisionReasoner 为 39.0 pp；同时 joint rate 分别仅为 2.5% 和 10.1%。这说明一个合理的 caption 或 CoT 过程，不能作为坐标输出正确的充分证据。

因此本工作不是否定 attention 或 CoT 对语言任务的作用，而是指出它们在 grounding 中必须经过 bbox-level 验证。当前证据支持“caption/CoT 对 bbox 的指导不稳定”，不支持把它们直接当作幻觉修正器。

### 3.4 任务链路应当从 hallucination 到 drift，形成闭环

我们需要的不是单个 re-ranker，而是一条完整但成本受控的链路：

```text
先判断：错误来自 proposal miss 还是 instance binding
   -> 当前实例与所有关键 atom 一致：ACCEPT
   -> wrong-instance 且正确实例在 Top-K：候选进入 RELOCALIZE 评估
   -> absent / ambiguous / 无安全替代：ABSTAIN
```

这个顺序很重要。没有前面的 visual claim verification，候选池只是在多个可能框之间猜测；有了可信的 existence/binding 判断，候选池才可以成为安全的视觉漂移修复器。

## 4. 方法定位：因果诊断为主，选择性修复为条件升级

### 主路径：冻结上游的 binding audit

部署接口保持不变：

```text
image + original query + one upstream bbox
    -> one local composite detector pass
    -> evidence state + oracle attribution
    -> ACCEPT / ABSTAIN / RELOCALIZE
```

- **BOH / object absence：** 用 object/full-expression existence evidence 提供保守弃权；
- **ROH / binding：** 用项目负责人审核的 target/reference/atom truth 判断对象、属性/动作和 reference 是否绑定到同一实例；
- **visual drift：** 只在“目标存在、原框被反证、替代候选完整支持”时，才从一次 pass 产生的固定候选池中选择替代框；
- **uncertainty：** evidence 不足时输出 `ABSTAIN`，不把不确定样本伪装成错误实例或安全修复。

这条路径不读取 model identity、GT、IoU、hallucination type 或数据集类别词表；核心预算是一条上游 MLLM answer 和一次 detector image-encoder forward。它满足“即插即用”的前提：可接在不同 RL/reasoning 模型之后，而不重训这些模型。

### TRACE-Bind 在这条叙事中的位置

TRACE-Bind 检验“跨层 proposal 稳定性是否包含额外 semantic binding 信息”。500 组 relation development proxy 给出了否定性结果：final static typed geometry 的 AUROC/AUPRC 为 0.6135/0.6097，trajectory-only 为 0.5009/0.5291，trajectory + swaps 为 0.5468/0.5559；后两者相对 static 的 grouped-bootstrap AUROC 差值置信区间整体为负。

因此不能再把 TRACE 写成核心方法。它保留为同 forward、低额外开销的诊断 feature family 和完整负结果，说明 proposal trajectory 可能稳定地收敛到错误实例。当前 proxy 也不是项目负责人审核的 wrong-instance truth，所以 reviewed evaluation 仍会把 TRACE 纳入 matched ablation，但不得据此训练 TRACE head 或拟合动作阈值。

## 5. 两条路线：为何选择 verifier 而非继续堆训练和外部模型

已有训练实验包括 Qwen base LoRA、外部 CLIP 视觉验证，以及百炼教师/蒸馏。它们的部分 strict calibration 结果可以达到 0.8 以上 mIoU，说明关系信息在足够强的监督或外部视觉能力下是可学习的。

但这些结果不能直接成为本项目的主方法：严格子集很小，后续 repaired-500 transfer 没有超过 fixed candidate pool；更重要的是，这条路线引入了新的外部视觉模型、教师成本或上游微调，改变了“低训练成本、可插拔后处理”的问题设定。

| 路线 | 优点 | 对本项目命题的代价 | 本项目角色 |
|---|---|---|---|
| 训练/监督 base model、LoRA、teacher distillation | 可以直接学习视觉关系，局部结果可能很高 | 训练和部署成本上升，难以区分是 verifier 改善还是新增模型能力 | upper bound、诊断和数据质量探针 |
| 外部 CLIP/VLM judge | 有更强的视觉语义 | 多模型依赖、额外 forward/API，难以成为统一 plug-in | 不进入核心部署路径 |
| 冻结上游 + verifier | 不改变 RL 模型，可跨模型接入，成本清楚 | 证据更弱，必须面对严格 FNR/mIoU 安全门槛 | 核心论文路径 |

因此，后续允许训练的对象应是 verifier 自身的小型 action/binding head，而不是继续加入更大的外部视觉证据或改造 base model。这样即使使用审核数据，也仍保持“上游冻结、即插即用、额外成本可控”的方法边界。

## 6. 目前结果告诉我们什么

| 实验 | 得到的结论 | 对主线的意义 |
|---|---|---|
| 多模型修复版评测 | ROH 显著难于 BOH；IoU、HR、FNR 不可互相替代 | 确立问题与联合指标 |
| 多 prompt / candidate policy | 可修复部分定位漂移 | 作为后段 relocalization，不是拒答机制 |
| E2/E2b CLIP、ROI、Qwen、teacher | 局部/小样本可高，但 image-disjoint transfer 失败 | 不继续作为核心方法 |
| E3 existence verifier | 所有 22 个 held-out model/task 组满足安全门槛，但 BOH 改善快于 ROH | 证明 plug-in existence gate 可行，不是完整答案 |
| CABLE train-free | FAR -2.60 pp、ROH -3.08 pp，但 added FNR +6.86 pp、mIoU -2.27 pp | 说明 ROH 信号存在，但当前拒答证据不够安全 |
| attention/logit intervention | role 结构可见，因果分数不可用 | 不直接迁移语言解码干预 |
| TRACE 500-group relation proxy | static AUROC 0.6135；trajectory-only 0.5009，trajectory + swaps 0.5468 | 停止 learned TRACE，保留 static baseline 与轨迹消融 |

最关键的结论是：我们不缺一个能把数字做高的模型，缺的是能区分 proposal miss 与 wrong-instance binding 的可信标注和因果诊断；修复模块只能建立在这一步之后。

## 7. 验证设计与停止条件

主实验以 unchanged upstream 为基线，按以下顺序验证：

1. 构造不按上游正确性筛选的 200--300 条 same-class natural audit cohort；
2. 项目负责人单人审核 target/reference box、atom state 和 ambiguity，记录 confidence、queue hash 与 `annotation_protocol=single_project_owner`，不报告 inter-reviewer agreement；
3. 后验连接固定 Top-K proposal，报告 target/reference coverage；
4. 按 `ParseErr / SeeErr / TargetErr / RefErr / RelErr` 做 oracle replacement attribution；
5. 在相同 reviewed rows 上比较 final confidence、static typed geometry、localized attention 和 TRACE；
6. 冻结诊断结果后，再评估 top-score rerank 与可选轻量 edge head；
7. 只有存在高精度安全区间时才评估 conditional relocalization。

每个版本都要同时 hold out 一个完整 upstream model 和一个 image fold，在另一 image fold 做半预算校准。主结果必须逐模型、逐任务、按 BOH/ROH 和 base/RL-reasoning 分层报告：

| 必须满足的条件 | 门槛 |
|---|---:|
| added FNR | 每个 model/task <= 3 pp |
| positive mIoU loss | 每个 model/task <= 0.005 |
| ROH FAR 降幅 | 不低于 BOH FAR 降幅 |
| conditional relocalization regression | nonzero-to-zero <= 1% |
| 成本 | 上游 MLLM call=1；核心 detector encoder forward=1 |

若不存在同时满足这些条件的阈值，不生成 deployment artifact。PRBench 和 sealed repaired-1996 仍只允许在所有规则、head 和阈值冻结后访问。

## 8. 创新性与论文风险评估

| 维度 | 保守评估 |
|---|---|
| 问题创新 | 高：把 RL grounding 的 IoU 提升与负样本接受、视觉漂移和 ROH-BOH gap 放在同一个可靠性框架评估 |
| 方法创新 | 待验证：冻结上游后的 selective relocalization 只有通过严格回归门槛才构成方法贡献；TRACE 已降为负结果 |
| 工程价值 | 高：可接入不同模型，不需重新训练上游，成本/延迟可单独核算 |
| 当前证据 | 中等：问题、多模型差异和负结果充分；项目负责人 binding truth 与因果拆解尚未完成 |
| 最大风险 | 高：若 paired counterfactual 和 oracle decomposition 也不能产生跨模型稳定结论，则问题贡献不足；若 router 不安全，则不能宣称修复 |

论文不能声称“RL 导致 hallucination”或“CoT 无效”。可以声称的是：在当前统一评测中，IoU-oriented grounding 模型存在未被正样本 IoU 捕获的负样本接受与视觉漂移；一个被冻结上游后的 verifier 是解决这一缺口的合理且可检验的系统方向。

## 9. 建议与导师讨论的决策

1. 主命题固定为：**区分 proposal miss 与 wrong-instance binding，并在可恢复子集上进行风险受控的 selective relocalization。**
2. LoRA、CLIP 和百炼教师固定为 upper bound/diagnostic，不再扩张为部署主线。
3. TRACE 已触发停止条件，不进入 learned head；static typed geometry 是当前正式 baseline。
4. 下一阶段只投入项目负责人 binding truth、paired counterfactual 和 oracle attribution；若训练 learned head，另建 train-only reviewed cohort。
5. 轻量 edge head 与 router 均为条件分支；任一安全门槛失败都不生成 deployment artifact。

## 10. 可复现证据与边界

- 修复版多模型结果：[十二模型修复版评测报告](../legacy/candidate_pool_v1/results/十二模型修复版评测报告_含ROH-VCD.md)
- 0802 relation verifier 记录：[E2B_PROGRESS.md](E2B_PROGRESS.md)
- E3、CABLE、attention diagnostic：[E3_SINGLE_PASS_PROGRESS.md](E3_SINGLE_PASS_PROGRESS.md) 与 [CABLE_METHOD.md](CABLE_METHOD.md)
- 标注与数据隔离协议：[SINGLE_PASS_ANNOTATION_PROTOCOL.md](SINGLE_PASS_ANNOTATION_PROTOCOL.md)

本报告中的全部数值来自 development 数据。CABLE、crop、attention 和 TRACE-Bind 都未被写成最终性能提升；它们只能在既定安全门槛与冻结协议通过后进入部署或 sealed evaluation。
