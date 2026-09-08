# V-SIGHT Progress

**Updated:** 2026-09-05

**Phase:** 即插即用 verifier 在真实上游模型上验证 + reference-box 复核（解锁 SWITCH/RELOCALIZE 四态动作）

> 当前权威叙事与结论见 `docs/updates_2026_09/DIRECTION_UPDATE.md` 与 `FINDINGS.md`；
> 可复现脚本见 `experiments/roh_boh/`。本文件早期内容（IoU=0 审计、TRACE 诊断、E2 负结果）
> 作为决策历史保留在下方与 Git 历史中，不再是当前方法。

## 本轮完成（2026-09）

- **问题层因果化**：ROH>BOH 与 VQA↔grounding 解耦（P(定位对|判别对)=0.35、φ=0.12）确立，并用 11 篇
  上游模型论文原文核实其成因——"正样本-only RL 训练 + 坐标 token 无独立梯度"两个范式选择的必然产物
  （Visual-RFT 作为剂量-反应反例）。
- **Gate A 通过**：train-free 信号近随机（BOH/ROH separation AUROC ~0.49），VLM 语义验证是解法
  （BOH 0.791 / ROH 0.670，500-dev）；relation 类最难（0.656）。
- **方法主结果**：即插即用 verifier（S5 → B1a 融合 → B1b 弱标签 LoRA → B1c 集成）在 4 个真实上游模型上，
  以报告自身 T2 指标衡量、正样本 mIoU 损失 ≤0.005 约束下，高幻觉上游 FG@Neg 下降 22–30pp；
  跨未见上游泛化（leave-one-model-out）。
- **成本**：便宜检测器信号抓 BOH 不抓 ROH → 分层触发级联（~75% VLM 调用率下 catch 超纯便宜头与纯 VLM）；
  3B 骨干保住 7B ~90% ROH 判别力。
- **数据工程**：GroundingDINO reference 候选（99% 覆盖）+ 旗舰视觉模型第一遍预筛（三态 triage），
  人工只做二轮确认。发现并定位某归一化坐标上游模型的评测坐标 bug（仅影响单一模型）。
- **诚实负结果**：TRACE decoder-trajectory（AUROC≈0.50）、train-free 注意力信号（AUROC<0.5）、
  负例工程重训退化——均保留为 ablation。

## 进行中 / 下一步

- reference-box 人工二轮复核（VLM 预筛已完成，产物待人工确认）→ 冻结带 reference box 的 blind edge 集。
- 完成后评估 SWITCH/RELOCALIZE 的完整 edge 正确性（target✓ ∧ reference✓ ∧ 角色未交换 ∧ relation 成立）。
- 修复上游坐标 bug 后只重跑该模型的 T2/T4。

---

## 决策历史（早期阶段，保留）

以下为 2026-08 及更早的状态，已被上面的 2026-09 工作取代，仅作历史参考：

- E1 image-disjoint positive source 283,249 queries；P1 生成 14,000 queries 的 baseline+challenger（零推理错误）。
- E2 训练 CLIP 与 Qwen-LoRA candidate verifier，无学习式 selector 通过 oracle-gap 与 repaired-500 迁移门
  → 保留为负结果（`docs/E2_RESULTS.md`）。
- IoU=0 属性审计：114 valid-box zero-IoU groups，85/114 判为同类实例混淆、93/114 高混淆风险。
- 动作空间早期固定为 `KEEP / SWITCH / REJECT`，后扩展为四态（加入 DEFER）。
- CCV/CABLE 作为实现资产保留；formal 500-group relation proxy 完成，静态 typed geometry 是当时最强诊断
  （AUROC 0.6135），trajectory-only 近随机（0.5009）。
