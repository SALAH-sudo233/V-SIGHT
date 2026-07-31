# IoU=0 目标属性与指代混乱复核

## 覆盖状态

- 有效 baseline 框且 IoU=0：114 个目标组，180 个任务样本。
- 人工已完成（本范围）：19 个。
- 两阶段模型已完成：114 个；与人工重叠 19 个。
- 合并覆盖：114/114；未完成 0 个。
- 视觉证据由 `qwen3-vl-plus` 提取；最终结构化裁决由 `qwen3.7-max-2026-05-17` 完成。Max 不直接接收图像。

## 失败原因

| 类别 | 数量 | 比例 |
|---|---:|---:|
| `same_category_instance_confusion` | 85 | 74.6% |
| `target_reference_role_swap` | 4 | 3.5% |
| `wrong_category` | 9 | 7.9% |
| `localization_or_box_quality` | 3 | 2.6% |
| `visually_ambiguous_reference` | 1 | 0.9% |
| `annotation_or_gt_issue` | 12 | 10.5% |

## 指代与风险

### Query 是否绑定 GT

| 类别 | 数量 | 比例 |
|---|---:|---:|
| `yes` | 89 | 78.1% |
| `no` | 19 | 16.7% |
| `uncertain` | 6 | 5.3% |

### 实例混乱风险

| 类别 | 数量 | 比例 |
|---|---:|---:|
| `high` | 93 | 81.6% |
| `medium` | 8 | 7.0% |
| `low` | 13 | 11.4% |

## 属性覆盖

| 属性族 | 有结果样本 | 属性项 | high | medium | low |
|---|---:|---:|---:|---:|---:|
| `colors` | 109 | 319 | 277 | 42 | 0 |
| `materials` | 109 | 212 | 105 | 103 | 4 |
| `other` | 109 | 214 | 207 | 5 | 2 |
| `actions_or_states` | 105 | 136 | 131 | 5 | 0 |

目标辅助属性是模型对可见区域的描述，不等于 query 明示属性，也不应直接作为 IoU=0 的因果标签。

### Query atom 核实

| 类型 | supported | contradicted | not_visible | not_applicable | 总计 |
|---|---:|---:|---:|---:|---:|
| `object` | 149 | 16 | 8 | 0 | 173 |
| `color` | 5 | 0 | 0 | 0 | 5 |
| `material` | 1 | 0 | 0 | 0 | 1 |
| `attribute` | 1 | 0 | 0 | 0 | 1 |
| `action_state` | 19 | 1 | 1 | 0 | 21 |
| `count` | 11 | 3 | 0 | 1 | 15 |
| `relation` | 91 | 13 | 5 | 1 | 110 |

## Baseline 框视觉归类

| 任务 | 同类错误实例 | 不同类别 | 关系锚点 | 背景/坏框 |
|---|---:|---:|---:|---:|
| `t2_vqa_grounding` | 79 | 6 | 2 | 2 |
| `t4_caption_grounding` | 72 | 12 | 3 | 3 |

## 性别呈现核实边界

模型输出：male_presenting=23，female_presenting=16，unclear=11，not_applicable=64。
自动风险标记：证据不足/自相矛盾=1，含服装、体型、姿态等刻板印象敏感线索=14。
这些标签仅表示外观呈现，不能解释为生理性别；带风险标记的样本必须回到图像人工复核。

## 未完成样本

| base_sample_id | 阶段 | 错误类别 |
|---|---|---|

## 解释限制

- 当前报告已覆盖全部有效框 IoU=0 目标组；后续若人工记录继续追加，应重新生成汇总并记录新的源文件哈希。
- 视觉模型可能把可见纹理扩展成具体材质；应优先使用 high confidence 且有局部证据的项，medium/low 只作为候选。
- 人工审核与模型输出分开保存；本轮对 19 个已完成人工审核目标做了显式交叉复核，但未计算人机一致性，不能把模型裁决当成人工真值。
- `annotation_or_gt_issue`、query contradiction 和性别风险标记建议优先人工二审。

Token 记录：视觉阶段 273829，Max 裁决阶段 536662，合计 810491。
