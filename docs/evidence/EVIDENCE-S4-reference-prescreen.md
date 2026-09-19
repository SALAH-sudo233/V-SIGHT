# 证据资产：S4 reference 复核第一遍（旗舰 Qwen3-VL-235B）— 全量完成

> 用旗舰视觉大模型 **PPIO qwen3-vl-235b-a22b-instruct** 对 2728 个关系 query 的 GroundingDINO reference 候选做第一遍筛选，
> 供项目负责人二轮确认。质量经独立核验远超之前失败的 8B（8B 对每个框盲目 yes，235B 能选/能拒/能识别关系方向/能发现坏候选）。

## 运行
- 全量：2728 units / 6735 candidate calls，**2,905,751 tokens**，68 min（8 路并发，~1.5s/unit），解析零错误。
- 关系约定（已验证并用于 prompt）：relation = "target is \<relation\> reference"。
- 合规运行：token 在本地进程内从 auth.json 读、直接调 PPIO，不落盘/不传远程/不回显；图像本地处理。脚本 `Temp/s4_run/review_full.py`。

## 三态 triage 结果（2727 有效 + 1 缺图）
| triage | 数量 | 含义 | 人工动作 |
|---|---|---|---|
| **auto_accept** | 874 | VLM 确信恰好一个候选是有效 reference 且关系成立 | 抽检 |
| **conflict_multi** | 770 | 多个候选都 plausibly 成立（多为同类多实例真难例） | 二轮消歧 |
| **none_valid** | 1083 | 无候选正确框住 reference 或关系不成立 | 二轮确认是否丢弃 |

- 单候选 893：auto_accept 478 / none_valid 415。
- 多候选 1834：auto_accept 396 / conflict_multi 770 / none_valid 668。
- 选出 reference（chosen≠-1）：1644 units。

## 质量核验（独立抽检，确认 235B 可信）
- **关系方向判别**："person on yacht" 正确判"boat is not on person; person is on boat"，只选方向成立的候选；"school bus behind another" 判 is_ref=True 但 rel=False（"green bus is in front, not behind"）——8B 会全 yes。
- **坏候选过滤**："straw" 案例 none_valid，我用 vision_analyze 独立核验：图中根本没有吸管、DINO 候选是三明治和杯子——**判定正确，非过度严格**。
- **none_valid 多为合理拒绝**：其一是 DINO 候选确实不含 reference；其二是 relation 方向/绑定不成立。

## 发现的数据质量问题（影响下游，需注意）
reference queue 的 phrase 抽取产生了一批**非干净物体短语**（动词/子句片段），如
"is in the middle of the two cars"、"bench holds his bag"、"/ his wrist taking a picture"。
这类 phrase 无法对应任何单一物体框，被正确判为 none_valid。启发式初筛出 ~385 个疑似退化短语，
建议二轮时对 none_valid 中的退化短语单元直接丢弃、或回到 phrase 抽取环节修正。

## 产物（供二轮）
根目录 `s4_review/`：
- `review_full_out.jsonl`（全量原始判定，每候选 is_ref/rel_holds/conf/note）
- `worksheet_auto_accept.jsonl`（874，抽检）
- `worksheet_conflict_multi.jsonl`（770，重点二轮）
- `worksheet_none_valid.jsonl`（1083，确认丢弃）
服务器副本：`roh_boh_gate_a_500dev/results/s4_vlm_review_full.jsonl`。

## 二轮建议
人工只需重点看 **conflict_multi（770）**（同类多实例消歧，模型选了 chosen_idx 但需确认），
none_valid 抽查是否有漏掉的好 reference，auto_accept 抽检即可。这把原本 2728 全人工降到"重点 770 + 抽检"。

## 下一步（解锁 SWITCH/RELOCALIZE）
二轮确认后，auto_accept + 二轮通过的 conflict_multi 构成带 reference box 的 blind edge 集，
即可评 SWITCH/RELOCALIZE 的完整 edge 正确性（target✓ ∧ reference✓ ∧ 角色未交换 ∧ relation 成立），把 verifier 动作从二态扩到四态。

---

*生成 2026-09-05。全量 2728、2.9M tokens、235B、零解析错误、独立抽检通过。*
