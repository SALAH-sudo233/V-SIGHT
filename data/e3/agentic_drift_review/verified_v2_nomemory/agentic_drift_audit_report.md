# Agentic Drift Loop 审计报告

生成时间：2026-08-11T03:06:19.508431+00:00
数据角色：`development_agentic_audit_not_training`

> 审核队列按 agent 风险和控制带富集，以下 precision/recall 仅为审核队列诊断结果，不代表自然错误率。

## 审核完整性

- 队列：50 条；最新审核版本：50；已完成：49。
- 完成率：98.0%；未完成编号：agentic-review:114273d573c2a5fcc099。
- 队列哈希校验：通过。单次 `项目负责人` 审核有效。
- Agent effective config：`a57eea39e6e3a20edcd63a4cca61101f304da76a5c50d3da713aed552f1b6139`（verified_unique）。

## 诊断结果

- Parser incorrect：9（18.4%）；uncertain：16（32.7%）。
- evidence state 精确一致率：24.5%；分歧：37 条。
- WRONG_INSTANCE precision：0.0；recall：0.0。
- WRONG_INSTANCE 排序诊断（风险富集诊断队列）：drift risk AUROC/AUPRC=0.6122448979591837/0.33125707486609735；review priority AUROC/AUPRC=0.5748299319727891/0.19270870394149492。
- 目标 Top-K 覆盖：17/23（73.9%）；参照 Top-K 覆盖：21/36（58.3%）。
- 人工目标校正框：0；人工参照框：41。

## 策略推导

- 动作计数：{"ABSTAIN": 32, "ACCEPT": 17}。
- oracle attribution（保守 proxy）：{"NONE": 10, "PARSE_ERR": 9, "SEE_ERR": 13, "UNRESOLVED": 17}。
- 可进入 RELOCALIZE 的记录：0；当前没有人工目标校正框时，该数值必须为 0。
- 关系候选证据没有 typed edge score；RELOCALIZE 使用 target/reference Top-K 覆盖作为 witness proxy，不能宣称因果修复。

## 数据边界

verified memory 独立写入，`training_eligible=false`，不回写 self-memory，不进入主训练集，也未访问 sealed heldout。
