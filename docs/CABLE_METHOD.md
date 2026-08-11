# CCV-CABLE

CCV-CABLE（Counterfactual Atom-Binding Ledger with Gap-Balanced Selective
Control）是冻结上游 MLLM 和 detector 的 post-hoc verifier。严格主线保持每个
query 一次上游调用和一次 composite detector image forward；它不再把 detector
证据压成一个 flat score，而是记录 object/attribute node 与有向 relation edge。

## 单次 pass 证据契约

`TypedCounterfactualPromptCompiler` 把 claim、reference、完整表达式和安全反事实
编译进同一个 prompt。每个 segment 带 `claim | inverse | decoy | null`
provenance。当前硬 inverse 仅包括方向/角色 inverse、support/contact role swap，
以及 `open/closed`；颜色和大小不生成硬反例。

`AtomBindingLedger` 输出：

- object/attribute node：object 和 attribute support、共同定位、decoy margin、
  full support 与 candidate ambiguity；
- relation edge：claim、替代 target/reference、swap margin、inverse/symmetry/role
  consistency、full-edge agreement 与 uncertainty；
- 汇总 `M_contra`、`M_restore`、`U_edge` 及明确 witness 类型。

`between` 少于两个可区分 reference 时为 unknown。front/behind 等不能由 2D
geometry 解释的关系只有出现 localized inverse prompt witness 才能产生负证据。

## 动作安全

决策顺序为 absence/null contradiction、typed restoration、兼容旧 CCV 的安全
relocalization、明确 typed contradiction，最后是 preserve。缺证据、关系不可解释、
高 edge ambiguity 均返回上游结果并标记 `BINDING_UNCERTAIN`。attribute/relation
拒答必须同时具有明确 inverse/swap witness，学习头不能绕过此约束。

learned 版本使用独立 `absence / contradiction / restoration` 单调 head。部署阈值
只能按 task、atom type 和 calibration fold 查询；model identity 与 hallucination
type 只用于校准审计，绝不写入 inference lookup。

## 审核与冻结

Stage-B v2 对 applicable atom 使用 `supported / contradicted / unobservable` 三态。
可观测 relation 必须由项目负责人绘制一个 reference bbox；`RELOCALIZE` 必须绘制
一个 corrected target bbox。协议固定为 `single_project_owner`，不计算双人 bbox IoU
或 inter-reviewer agreement。`unobservable` 不进入 contradiction head；UNCERTAIN、
低置信度、敏感属性和 schema/evidence conflict 不进入 learned CABLE。

运行顺序：生成 500 条 train-only queue，完成项目负责人单人审核与 v2 schema gate，生成
composite evidence 和 reviewed manifest，训练三分支 head，再运行
`calibrate_cable_gap_controller.py`。nested safety/gap gate 失败时脚本不会写部署
artifact。PRBench 和 sealed repaired-1996 只能在规则、head 与阈值全部冻结后访问。

Conditional crop extension 位于 `vsight.cable_crop`，只对 `BINDING_UNCERTAIN`
触发，最多增加两个 detector forward、零次 MLLM 调用，必须与严格单次 pass 分开
报告。

## 首次 typed evidence 尝试（2026-08-05）

新 compiler 已在非密封 development queue 的全部 2,500 个唯一 query/image group
上真实重跑，输出位于 `outputs/cable_composite_v2_t010/`。该运行沿用此前固定的
GroundingDINO `(box,text)=(0.10,0.10)`，8 个 shard 均成功，全部记录为 v2 schema、
`evidence_complete=true` 且 `image_encoder_forwards=1`。平均每条 10.05 个 proposal，
object/reference/full-expression proposal 覆盖率分别为 93.96%、79.48% 和 49.08%。
1,433 条 query 编译出安全 counterfactual，578 条获得至少一个 inverse proposal，
即 conditional witness coverage 40.33%。聚合 detector 延迟为 144 ms p50 / 283 ms
p95，单 shard 峰值显存为 2.25--2.30 GB。结构审计保存在
`outputs/cable_composite_v2_t010/audit.summary.json`。

严格 single-pass train-free 评测位于 `outputs/cable_trainfree_v2_t010/`。完整 CABLE
在 55,000 条开发记录上使总体 FAR 下降 2.60 pp、ROH FAR 下降 3.08 pp、BOH FAR
下降 2.11 pp，ROH--BOH gap delta 为 -0.97 pp；但 added FNR 为 6.86%，正例 mIoU
下降 2.27 pp，21/22 个 model/task 组未通过安全 gate。因此该点只能作为 diagnostic，
不能作为性能改进或部署结果。

与 `cable_no_swap` 相比，完整 CABLE 只额外产生 57 个 REJECT，其中 41 个落在负例、
16 个落在正例；它额外降低 0.09 pp FAR，却增加 0.15 pp added FNR。inverse/symmetry
分支只改变 3 个 relation 负例的 `ACCEPT -> RELOCALIZE_FLAG`，严格主线未应用新框，
所以总体指标不变。这说明当前增量主要来自 candidate swap，且 witness precision 尚不
足以安全拒答。

随后运行 nested gap-balanced calibration。`t2_vqa_grounding` 的 calibration fold 1
不存在同时满足 FNR、mIoU、gap 和 FAR 下降约束的阈值；脚本按协议中止，未写
deployment artifact。项目负责人 Stage-B v2 审核仍未完成，所以 learned CABLE 继续保持
禁止训练状态。PRBench 与 sealed repaired-1996 未访问。

## Conditional crop pilot（2026-08-06）

标注等待期间运行了独立的 conditional-crop development pilot。样本按 11 个 model、
2 个 task、5 个正负 strata 分成 110 个桶，每桶固定抽取 6 条，共 660 条。非灰区
保持上游结果；只有 frozen single-pass 产生 `BINDING_UNCERTAIN` 时才运行 target
crop（1.5 倍），必要时再运行 target/reference union crop（1.25 倍）。运行没有新增
MLLM 调用，输出位于 `outputs/cable_crop_pilot_v2/`。

306/660 条（46.36%）触发 crop，63 条使用一次额外 forward，243 条使用两次；只有
19 条离开 uncertain。14 条变成 absence REJECT，其中 11 个负例、3 个正例，crop
contradiction precision 为 78.57%（Wilson 95% CI 52.41%--92.43%），负例覆盖率
5.34%。没有任何 `RELOCALIZE`。第二次 union crop 只把一条正例恢复为 ACCEPT，因
target-only 本来也保留上游，最终没有产生任何增量动作变化。

target-only 在该平衡 pilot 上的 FAR delta 为 -2.08 pp，relation/attribute FAR delta
分别为 -3.79/-0.76 pp；added FNR 为 +2.27 pp，正例 mIoU delta 为 -0.29 pp，
ROH--BOH gap delta 为 -0.38 pp。尽管总体 FNR/mIoU 数值在预算内，17/22 个小样本
model/task 组未通过完整 gate，且 positive nonzero-to-zero regression 为 2.25%，超过
1% 上限。因此该分支仍不安全。

成本方面，target-only 平均/p95 detector forwards 为 1.46/2，p50/p95 总 detector
延迟为 242/413 ms；target+union 为 1.83/3 和 255/556 ms，但质量指标完全相同。
运行同时暴露并修复了极端细长 crop 使 GroundingDINO encoder token 少于固定 top-k
的问题：现在只在原图范围内扩展窄边到 0.75--1.33 的安全宽高比，并保留全局坐标
映射。该修复有独立回归测试。

结论是 crop 增加了明显成本，却没有提供可用的 restoration edge；在 reviewed binding
数据到位前不扩大运行，也不把 crop 结果并入严格 single-pass 主结论。sealed 数据仍
未访问。

## Target role 与 token-span alignment pilot（2026-08-06）

relation parser 审计发现，2,334 条 relation query 中有 328 条（14.05%）把 target
与 reference 角色翻转：当 target 不在 COCO-80 而 reference 在词表中时，旧逻辑会
把 reference 当作 object。例如 `the helmet next to the red car` 曾被解析为 car；现在
只从 relation 前的 target span 解析 object，未知类别保留 query-only open-vocabulary
phrase，示例会正确得到 target=helmet、reference=red car。

同时实现了可选的 GroundingDINO token-span alignment。它仍然只执行一次
`model(**inputs)`，随后用原始 token logits 按精确 segment span 独立生成 proposal，
并排除 determiner token。在 100 个 development semantic query（跨 11 个模型、2 个
任务共 2,200 条记录）的配对 pilot 中，token-span 相比 legacy label 将 full coverage
从 46% 提高到 81%，inverse proposal coverage 从 13% 提高到 41%，reference
coverage 从 89% 提高到 94%；平均 proposal 数从 10.49 增至 11.97。这里 label 长度
统计仅记作 `long_label_proposals`，不能解释为 mixed-label 错误率。

但 frozen train-free action policy 无法安全消费这些新增信号。role fix + token-span 的
总体 FAR 下降 3.68 pp，同时 added FNR 为 7.14%，正例 mIoU 下降 2.67 pp，
ROH--BOH gap delta 反而为 +2.94 pp。29 个 typed reject 中仅 17 个为负例、12 个为
正例，precision 为 58.62%，22/22 个 model/task 组均未通过安全 gate。因此本轮只
保留 representation/provenance 改进作为 diagnostic：默认 alignment 恢复为
`legacy_label`，只有显式传入 `--alignment token_span` 才启用新路径；不进行全量 v3
evidence 重跑，也不把结果宣称为性能提升。下一步仍需通过 Stage-B 项目负责人审核数据训练
learned contradiction/restoration head。PRBench 与 sealed repaired-1996 未访问。

## CABLE-RAFT attention diagnostic（2026-08-06）

为避免把 COCO-80 alias 当作 open-vocabulary 能力，新增了可选的
`--alignment raft` 路径（RAFT：Role-Aware Attention Flow Transport）。该路径的
prompt 只包含原始完整 query，不展开类别、属性、动作或逆关系词表；一次
GroundingDINO forward 返回 encoder text→vision attention，随后在 token offsets 上
结合通用关系边界 grammar 进行无类别词表的 target/predicate/reference 结构分段，并
生成 attention transport ledger；该 grammar 只提供边界，不提供类别或反事实语义。
`legacy_label` 仍是默认值，旧 v3 evidence 不改变。

RAFT ledger 记录每个 span 的 attention entropy、候选区域质量、target/predicate/
reference transport、role-swap margin、edge ambiguity 以及 layer/head agreement。
它不把 attention 本身当作拒答证据：低分段置信度、不可分离的关系或高 edge ambiguity
均返回 `BINDING_UNCERTAIN`，当前实现只产出 v4 diagnostic evidence，尚未接入 action
改变。

单图可行性探针（GroundingDINO base，`the helmet next to the red car`）确认一次
forward 可返回 6 层 text→vision attention 和空间 feature-shape metadata；该路径约
0.8--1.4 s、峰值约 2 GB，明显高于旧无 attention 路径，后续 pilot 必须单独报告
latency/GPU memory。由于真实 attention 的 role 分段仍可能把修饰语吸收到 predicate，
在 100-query pilot 通过 edge witness precision 和全部安全 gate 前，不进行全量 v4
重跑，不允许新增 ROH rejection，也不访问 sealed 数据。

### VLM2 A800 development pilot（2026-08-06）

在 `yanzhonghao@10.160.4.185:2223` 的三张 A800 服务器上复用已有本地
GroundingDINO 权重，在空闲 GPU 上运行了更新 guard 后的 100 条 development queue。
该运行独立放在 `V-SIGHT-CCV-RAFT` 目录，未覆盖已有实验目录；每条 query 仍只有一次
detector image-encoder forward，未访问 PRBench 或 sealed 数据。结果保存在
`outputs/cable_raft_attention_vlm2_pilot/`。

结构审计结果：100/100 条为 v4 evidence 且 attention metadata 完整，运行无错误；
detector latency p50/p95 为 191.8/207.4 ms，峰值显存约 2.31 GB；94% query 生成
three-role span，edge rate 为 30%，平均 edge uncertainty 为 0.939。当前
`action_policy_applied=false`，RAFT 仅作为可审计 diagnostic，不能据此宣称 FAR/FNR
或 ROH--BOH gap 改善。这里的 191.8/207.4 ms 来自旧 runner 的异步 CUDA dispatch
计时；后续 causal pilot 增加显式 CUDA synchronize 后才得到可作为端到端 wall time
的数值，两者不能直接比较。

## Proposal-conditioned decoder attention（2026-08-06）

针对 encoder attention 的高 ambiguity，加入了 decoder-conditioned binding ledger。
GroundingDINO 的 decoder text cross-attention 提供 `[batch, head, object-query, text]`
权重，deformable vision cross-attention 提供 object-query 的 sampling weights；在
同一次 forward 的最后一个 decoder layer 用 pre-hook 重建 sampling locations，形成
`text role → object query → vision sample` 链。它不改变模型权重、不引入类别词表，也不
增加 image encoder forward。

在相同的 100 条 development pilot 上，decoder text layer agreement 为
0.928（encoder layer/head agreement 为 0.347/0.175），共 30 条 query 产生
proposal-distinct relation edges（78 条 edge candidates）。decoder edge 的平均
role-swap margin 为 0.0769，92.3% edge candidates 的 swap margin 为正；其
proposal-conditioned visual self-alignment 平均约 0.28。decoder evidence 的
edge rate 仍为 30%，所以这些数值只说明 proposal-conditioned 信号比原始 attention
更结构化，尚未证明 witness precision 或安全拒答能力。

当前 ledger 仍只写 diagnostic 字段：`decoder_edges`、`decoder_top_transport`、
`decoder_transport_relative_margin` 和 `decoder_edge_uncertainty`。下一步必须在
reviewed binding 数据上验证 causal token intervention 或 decoder-side role mask；在
此之前不允许 decoder attention 新增 `REJECT`/`RELOCALIZE`。

## Decoder-only causal role intervention（2026-08-06）

新增显式 `--raft-intervention` 诊断分支，默认关闭。原始 detector forward 后缓存
vision/text encoder memory、初始 object queries 和 reference points；target、predicate、
reference 三个 token mask 合并成一个 batched decoder replay。检测头仍使用原始 token
mask，因此 logit 变化来自 decoder 无法读取被屏蔽 role，而不是分类头直接隐藏该 token。
该分支平均 2.88 个 counterfactual branches、一次 decoder replay，额外 image encoder
forward 始终为 0。

同一 100 条 development pilot 的 signed logit-drop 结果为负：只有 1% query 形成
完整正 causal edge，所有 proposal-role effect 中只有 31.46% 为正，平均 relative drop
为 -0.0154。predicate token 被 mask 后经常使原 token logit 上升，说明注意力重分配
破坏了“logit drop 即支持证据”的假设；signed intervention 不得进入 reject policy。

同时记录了不带符号的 object-query hidden-state cosine drift。该 influence edge 覆盖率
为 30%，hidden drift p50/p95 为 0.00274/0.0513；78 条 influence edge candidates 中
88.46% 的 role-swap margin 为正，平均 margin 为 0.0332，但平均 uncertainty 仍为
0.9917。这只能用于 dependency/uncertainty routing，不能解释为 claim support 或
contradiction。

单次 batched decoder replay 的 p50/p95 为 21.7/22.4 ms；显式同步且包含 attention
materialization/intervention 的总 p50/p95 为 1.23/1.52 s，峰值显存约 2.31 GB。当前
停止继续调 causal threshold；下一步仅在 reviewed binding subset 上验证 influence
ranking 是否与真实 target/reference binding 一致，否则该分支作为 diagnostic 终止。

额外运行了相同前 20 条的同步 wall-time 对照：无 intervention RAFT 为
1.17/1.63 s，intervention 为 1.30/1.74 s；因此包含 batch materialization 等开销的
p50 增量约 133 ms（11.3%），不能只用 22 ms decoder kernel 时间代表端到端成本。

## 与 TRACE-Bind 路线的衔接（2026-08-07）

按照 `vsight_tracebind_ccfa_competitor_and_contribution.md` v0.1，CABLE 不再继续独立
扩张阈值、crop 或 causal intervention 分支。现有 static ledger、role provenance、
decoder attention、swap margin 和 intervention 负结果统一进入 TRACE-Bind Phase 0
failure matrix，并作为 100-query train-free pilot 的 matched static / attention baseline。

当前主动实现转向 decoder proposal trajectory、跨层 rank stability、bbox convergence
和 fixed-candidate assignment witness，执行协议见 `TRACE_BIND_PILOT.md`。CABLE 的
learned contradiction/restoration head 仍需 Stage-B 项目负责人审核，但它被放到 TRACE
static-vs-trajectory gate 之后；在此之前 CABLE/TRACE 均不得新增动作或生成 deployment
artifact。
