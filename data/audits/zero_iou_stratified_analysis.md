# IoU=0 分层失败分析与实验决策

生成时间：2026-07-31T14:24:30.681634+00:00

## 数据边界

- 有效框 IoU=0 目标组：114；模型成功复核：114；范围内人工完成：19。
- 关系表达：111/114（97.4%）；同时有 T2/T4 有效零框：66/114。
- 当前 manifest 没有独立的 BOH/ROH 标签，因此以下是 grounding failure strata，不把原因强行命名为 BOH 或 ROH。
- 模型统计仅来自视觉证据 + `qwen3.7-max-2026-05-17` 裁决；人工统计独立列出，不混合为同一标签源。

## 主要结论

1. 83/111 个关系样本被判为同类实例混乱；问题核心是完整表达式绑定，而不是 decoder 中是否出现某个颜色 token。
2. T2 的人工 `SWITCH` 为 4，T4 为 9；在已人工审核的子集，描述后 grounding 更常需要切换。
3. 现阶段最有价值的即插即用模块是 candidate-conditioned binding verifier：对每个候选框检查对象、关系锚点、动作/属性与空间一致性，再输出 KEEP/SWITCH/REJECT；不建议先做一个独立的 ROH/BOH 文本分类器。
4. 同类实例高度集中时应增加难度/拒答分支；query 与 GT 不成立或关系被视觉证据否定时，直接 REJECT 比盲目扩大候选池更符合 grounding 目标。

## 原因分层

| 分层 | n | 同类实例 | 角色交换 | 错类别 | 定位/框质量 | GT/标注 |
|---|---:|---:|---:|---:|---:|---:|
| `relation` | 111 | 83 (74.8%) | 4 | 9 | 3 | 11 |
| `attribute` | 2 | 2 (100.0%) | 0 | 0 | 0 | 0 |
| `object_only` | 1 | 0 (0.0%) | 0 | 0 | 0 | 1 |

### 目标类别（至少 2 个样本）

| 分层 | n | 同类实例 | 角色交换 | 错类别 | 定位/框质量 | GT/标注 |
|---|---:|---:|---:|---:|---:|---:|
| `person` | 47 | 40 (85.1%) | 0 | 0 | 1 | 6 |
| `chair` | 8 | 4 (50.0%) | 1 | 2 | 0 | 1 |
| `dining table` | 7 | 4 (57.1%) | 1 | 2 | 0 | 0 |
| `umbrella` | 5 | 5 (100.0%) | 0 | 0 | 0 | 0 |
| `laptop` | 5 | 3 (60.0%) | 0 | 1 | 0 | 0 |
| `couch` | 4 | 3 (75.0%) | 0 | 1 | 0 | 0 |
| `unknown` | 4 | 0 (0.0%) | 0 | 0 | 1 | 3 |
| `pizza` | 3 | 3 (100.0%) | 0 | 0 | 0 | 0 |
| `tv` | 3 | 1 (33.3%) | 0 | 0 | 0 | 2 |
| `suitcase` | 2 | 2 (100.0%) | 0 | 0 | 0 | 0 |
| `wine glass` | 2 | 2 (100.0%) | 0 | 0 | 0 | 0 |
| `tennis racket` | 2 | 2 (100.0%) | 0 | 0 | 0 | 0 |
| `truck` | 2 | 0 (0.0%) | 1 | 0 | 1 | 0 |
| `car` | 2 | 2 (100.0%) | 0 | 0 | 0 | 0 |

### 同类干扰实例数量

| 分层 | n | 同类实例 | 角色交换 | 错类别 | 定位/框质量 | GT/标注 |
|---|---:|---:|---:|---:|---:|---:|
| `1` | 46 | 36 (78.3%) | 2 | 5 | 2 | 0 |
| `3` | 29 | 26 (89.7%) | 0 | 2 | 0 | 1 |
| `2` | 24 | 21 (87.5%) | 1 | 0 | 0 | 2 |
| `0` | 11 | 1 (9.1%) | 1 | 2 | 1 | 6 |
| `10` | 2 | 0 (0.0%) | 0 | 0 | 0 | 2 |
| `11` | 1 | 0 (0.0%) | 0 | 0 | 0 | 1 |
| `4` | 1 | 1 (100.0%) | 0 | 0 | 0 | 0 |

## T2/T4 差异

| 任务 | 有效零框数 | valid_zero_unresolved | valid_zero_recovered | 同类自动类 | 其他类别/参照物自动类 |
|---|---:|---:|---:|---:|---:|
| `t2_vqa_grounding` | 89 | 61 | 28 | 68 | 17 |
| `t4_caption_grounding` | 91 | 57 | 34 | 63 | 27 |

模型原因按任务：

- `t2_vqa_grounding`：same_category_instance_confusion=67, annotation_or_gt_issue=12, wrong_category=4, localization_or_box_quality=3, target_reference_role_swap=2, visually_ambiguous_reference=1
- `t4_caption_grounding`：same_category_instance_confusion=65, annotation_or_gt_issue=11, wrong_category=8, target_reference_role_swap=3, localization_or_box_quality=3, visually_ambiguous_reference=1

## Query atom 证据

| 类型 | supported | contradicted | not_visible | not_applicable |
|---|---:|---:|---:|---:|
| `action_state` | 19 | 1 | 1 | 0 |
| `attribute` | 1 | 0 | 0 | 0 |
| `color` | 5 | 0 | 0 | 0 |
| `count` | 11 | 3 | 0 | 1 |
| `material` | 1 | 0 | 0 | 0 |
| `object` | 149 | 16 | 8 | 0 |
| `relation` | 91 | 13 | 5 | 1 |

颜色/材质 atom 很少，而关系 atom 占主导；属性描述应作为候选区分 cue，而不是独立 BOH/ROH 判别依据。

## 人工子集（独立证据）

### `t2_vqa_grounding` failure mode
| 类别 | 数量 | 比例 |
|---|---:|---:|
| `visually_ambiguous` | 7 | 36.8% |
| `same_category_wrong_instance` | 6 | 31.6% |
| `annotation_or_gt_issue` | 2 | 10.5% |
| `other` | 2 | 10.5% |
| `false_rejection` | 1 | 5.3% |
| `partial_or_oversized_region` | 1 | 5.3% |

动作：both_wrong=7, ambiguous=5, switch=4, keep=3
绑定证据：attribute=14, object_identity=14, action_or_state=10, left_right_or_depth=5, localization_tightness=1, none_visible=1

### `t4_caption_grounding` failure mode
| 类别 | 数量 | 比例 |
|---|---:|---:|
| `same_category_wrong_instance` | 11 | 57.9% |
| `visually_ambiguous` | 6 | 31.6% |
| `wrong_category` | 2 | 10.5% |

动作：switch=9, both_wrong=5, ambiguous=5
绑定证据：attribute=14, object_identity=14, action_or_state=12, left_right_or_depth=4, target_reference_relation=2, localization_tightness=1

## 建议的下一轮实验

1. **候选框级 binding verifier**：输入 query atoms、GT/候选框视觉证据和候选框间相对关系，分别打 object support、relation support、attribute/action support、box quality。
2. **难度分支**：用同类候选数、候选间相似度、关系 atom 数量、视觉证据冲突计数构造 difficulty score；高难样本提升拒答阈值，而不是固定扩大候选池。
3. **最小消融**：object-only、object+relation、object+relation+attribute/action、加 difficulty reject；分别报告 IoU=0 recovery、nonzero regression、REJECT precision、单样本延迟。
4. **审计优先级**：先人工复核 `query_binds_gt_target=no/uncertain`、`annotation_or_gt_issue` 和性别敏感线索样本；这些样本不应直接进入 verifier 正样本。

## 可复现来源

- manifest SHA-256: `0be219cdbc490103233d1912748e50b77c146357875aac2ff8538b13645669c1`
- model output SHA-256: `b5528edc0f8b408bcb51ad04eb571f49206c9d91966a97d6a891deabee2644fd`
- human review SHA-256: `58986b1e6b5f6c805e1aba30b5e1a52bbd8a7f6c702eb7376e64a6aba12b4da3`
