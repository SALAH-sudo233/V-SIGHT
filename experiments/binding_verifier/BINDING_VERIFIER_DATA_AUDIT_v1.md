# V-SIGHT Binding Verifier 训练数据审计报告 v1

审计对象：`refcocog_train2000.corrected_v2.json`（8,000 pairs / 2,000 images）
派生标注：`train2000.binding_audit_v1.jsonl`（SHA256 `c8492cf79a8bb367a5d0d72cf613c2c939b58efd4ac6b7875481315fd82d024a`）
标签来源声明：`challenge_type`/`ambiguity_status` 为 **programmatic 启发式**（`challenge_confidence=low`），**不是独立人工语义金标**。

依据 guidelines 第 15 节：先审计、统计、给示例、标不确定性，**尚未**大规模生成新数据。

---

## A. 当前数据已有的监督类型

- 严格 pair 结构：每 pair = `positive→KEEP` + `counterfactual negative→REJECT`。
- 只有 pair 级字段：`set/sid/ht/img/bbox/pos/neg/content_id/repair_version`。
- **缺失全部 binding provenance**：`challenge_type / ambiguity_status / label_source / reference_box / candidate_role / evidence_scope / hallucination_group` 在原文件中出现次数均为 **0**。
- 支持：格式学习、基础 KEEP/REJECT decision、属性/关系文本反事实。
- 不支持（凭现有字段）：同图同类实例绑定、relation direction、role swap、关系属于另一实例的拒识。

## B. 每类样本数量

| 维度 | 数量 |
|---|---|
| 总 pairs | 8,000 |
| BOH（object+co_occurrence） | 4,000 |
| ROH（attribute+relation） | 4,000 |
| object / co_occurrence / attribute / relation | 2,000 / 2,000 / 2,000 / 2,000 |
| unique images | 2,000（1996 集 7,984 pair + added4 16 pair） |
| KEEP prompts / REJECT prompts（拆分后） | 8,000 / 8,000（合计 16,000） |
| label_source=construction（未改） | 7,781 |
| label_source=independent_model（二次修复） | 219 |

challenge_type 分布（启发式）：
```
object_substitution              2000
cooccurrence_added_entity        1973
cooccurrence_internal_conflict     27
attribute_single_swap             920
attribute_multi_change            967
attribute_added                   113
relation_added_clause             959
relation_introduced               469
relation_other                    410
relation_direction_or_ref_changed 162
```

ambiguity_status（启发式）：`construction_only 5371 / unverified 2395 / clear 207 / ambiguous 27`。

## C. 每类是否真正支持 binding learning

- **object（BOH）**：负例为对象类别替换，多可由 target box 内容判定，支持基础 object decision。但按第 4 节，若 binding 统一设 `NA`，则只训练 decision 不训练 binding。
- **co_occurrence（BOH）**：1,973/2,000 为"追加一个伴随实体"，需判断该实体是否在**全图**存在——但数据未标注该实体位置，verifier 只能靠 target box + 全图推断。27 条为同一对象内部反义词冲突（standing/sitting 类），属第 3 节 safeguard 点名的"文本捷径"风险，应作 ambiguous。
- **attribute（ROH）**：920 为干净单属性替换（较可训练），967 为多 token 变化（可能引入非单因素、职业/不可见属性猜测），113 为纯追加属性。
- **relation（ROH）**：**结构性缺陷**。959 追加关系从句 + 469 引入关系 = 1,428/2,000 是"给正例加一个指向新 reference 的关系子句"，而**无 reference_box**；1,995/2,000 的负例相对正例引入了新名词 token（新 reference），却没有任何 reference 位置证据。真正的方向/角色变更仅 162（启发式上限，未经人工确认）。**这类样本无法有效训练 relation binding**。

## D. 哪些样本存在歧义

- 27 条 co_occurrence 内部反义词冲突 → `ambiguous`（自相矛盾语言捷径，非视觉判定）。
- 2,395 条 ROH（1,428 relation 加从句/引入 + 967 attribute 多变化）→ `unverified`：需 reference/属性可见性证据才能确定负例在图中确实错误。
- 关系方向类 162 条为启发式候选，未人工确认，不能直接当 clear。

## E. 哪些样本只是 construction label

- 7,781/8,000 `label_source=construction`，`ambiguity_status` 多为 `construction_only`。
- 其中 ROH construction_only = 1,518，ROH unverified = 2,395 —— 合计 3,913/4,000 ROH 缺乏独立可判定证据。
- 仅 219 条经同模型二次带图复核（`independent_model`，仍非人工金标）。

## F. 缺失的 hard-negative 类型（对照第 9 节）

| 类型 | 现状 |
|---|---|
| 9.1 同图同类多实例错误绑定 | **0**（train2000 无 instance/candidate 身份标签） |
| 9.2 关系方向反转 | 无带 reference_box 的干净样本（启发式候选≤162，未证实） |
| 9.3 target/reference role swap | **0** |
| 9.4 属性冲突且可观察 | 部分（920 单属性），但无可见性/遮挡标注 |
| 9.5 关系成立于另一对象 | **0** |
| 9.6 不可判定样本（DEFER 用） | 未显式隔离，混在强制二分类里 |

## G. 推荐新增数据格式

采用 guidelines 第 10 节 schema（`sample_id, base_image_id, candidate_box, reference_box, positive/negative_expression, hallucination_type, hallucination_group, challenge_type, construction_operation, positive/negative_target, label_source, ambiguity_status, evidence_scope, candidate_role, source_dataset, source_version, review_notes`）。新增写入版本化文件，不覆盖原始与本 corrected_v2。

**可用的人工证据源（重要）**：`golden_review/outputs/pilot_300.review_events.jsonl` = 352 条人工复核事件，含 `final_target_bbox_xyxy` 与 `final_reference_bbox_xyxy`（261 条有 reference box），`relation_status` verified/unverified/contradicted = 237/113/2。该图集与 train1996、dev500 **图像隔离为 0**（已核验 `train∩pilot=0`），是构造 9.1/9.2/9.3/9.5 clear hard-negative 的现成人工种子——但因图像不同，需作为独立扩展集，不能混入声称 train2000 已含。

## H. 推荐最小 pilot 规模

按第 12 节：每阶段 **50–200 个 ROH clear samples**。建议 P1 先做 100–150 条 clear ROH hard-negative（来源 pilot_300 人工 reference/target box），验证一致性与可分性后再扩。

## I. 每类新增数据的验证方式

- 同图同类多实例：A/B 皆为合理同类候选，禁止低 IoU/框大小/detector score 捷径；表达式仅对 A 成立须人工或独立可靠确认才标 clear。
- 关系方向 / role swap：显式记录 target/reference 角色与方向，图像中关系可辨，禁模板句法可判。
- 属性冲突：属性须可观察，遮挡/分辨率不足标 ambiguous。
- 全部：`challenge_label_source` 与 `ambiguity_status` 如实标注，保留 raw 证据与 before/after 对照；同模型复核 ≠ 人工金标，须声明。

## J. 不应加入当前强制二分类训练集的样本

- 27 条 co_occurrence 内部反义词冲突（ambiguous）。
- 2,395 条 ROH unverified（无 reference/可见性证据）。
- 所有 reference 不可见、方向不可判、多实例皆合法、属性被遮挡、candidate 未覆盖证据的样本 → `training_policy=exclude_from_hard_decision` 或 `use_for_defer`（第 9.6 节）。

---

## 隔离与去重核验（第 11 节）

- `train ∩ dev500 = 0`（COCO image id，已核验）。
- `train ∩ pilot_300 = 0`。
- 重复 sid = 0；重复 content_id = 0；重复 (img,bbox,pos,neg) = 0。
- **重复 (img,bbox,positive) = 5,803（2,003 组）**：同图的 4 类 pair 共享同一 candidate box 与同一 positive expression（1,835/2,000 图为单一共享 box+pos）。→ 名义 8,000 KEEP 正例的**有效独立正例 ≈ 2,000**。这必须在训练报告中声明（第 5 节 line 164）。

## 结论

- 图像隔离、pair 唯一性：通过。
- 名义规模 8,000 pairs 达标，但 **ROH 有效 binding 监督严重不足**：ROH clear 仅 87，unverified 2,395，同图错误实例/role-swap/带 reference_box 的方向样本为 0。
- BOH binding 若统一 `NA` 则只训练 decision，须在报告注明。
- 原始数据、corrected_v2、旧 run 均未修改；本审计仅新增派生标注与报告。

## 产物

- `train2000.binding_audit_v1.jsonl` — 8,000 条派生 provenance 标注（programmatic，非人工金标）。
- `BINDING_VERIFIER_DATA_AUDIT_v1.md` — 本报告。
