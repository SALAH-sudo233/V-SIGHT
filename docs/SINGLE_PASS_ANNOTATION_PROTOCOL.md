# 单次推理后处理标注协议

## 目标

标注用于训练与上游模型无关的 `ACCEPT / REJECT / RELOCALIZE` 后处理器。
部署输入只允许包含图像、query、上游模型的一次输出、本地检测候选与可复算视觉特征。
标注不得把 GT、IoU、正负类型、模型名称或最终指标写入推理输入。

## 三阶段标注

1. **目标存在性**：标注者只看原图与 query，判断完整指代是否有唯一、可见且证据充分的目标，输出
   `supported / contradicted / ambiguous`。
2. **原框一致性**：在第一阶段答案隐藏的条件下展示原框，核对框内对象身份、属性、动作和
   target-reference relation，输出逐原子的支持与矛盾证据，不直接选择 IoU 更高的框。
3. **候选动作**：项目负责人依据前两阶段的结构化证据，输出 `ACCEPT / REJECT /
   RELOCALIZE / UNCERTAIN`。`UNCERTAIN` 不进入训练。

`REJECT` 只表示完整目标在图像中没有充分视觉支持；若目标存在但原框身份、
属性或关系不一致，必须标为 `RELOCALIZE`，并独立绘制/选择一个可支持目标的
校正框。不能把这类样本折叠为负例，否则后处理器会通过拒绝正样本掩盖指代漂移。

## API 标注门槛

- 用于 learned head 的 API/人工队列只处理 train 图像；repaired-500 仅用于
  development/evaluation 和补充 binding 标注，不能成为最终 held-out；sealed
  repaired-1996 不进入标注队列。
- 每批至少混入 40 条隐藏控制样本，覆盖 positive、object、co-occurrence、attribute、relation。
- 控制组的高置信标签精度必须不低于 90%，且每类至少有 20 条被接受控制样本。
- 置信度低于 0.90、视觉证据为空、自动证据冲突或包含敏感属性推断的样本标为
  `UNCERTAIN` 并进入项目负责人复核；它们不进入训练。
- API 原始输出不可覆盖；晋级记录追加模型、prompt hash、时间、复核人和来源哈希。
- 当前首批人工输入队列为
  `data/e3/binding_annotation_queue/e3_binding_annotation_queue.train.jsonl.gz`；
  它只含 train 图像、query 和原框，审核完成前不得进入训练。

## 人工复核逻辑

人工只回答可观察事实：目标是否存在、原框是否框住该目标、属性/关系是否可见。人工不知道
模型来源和 GT 框，不根据算法期望修改答案。项目负责人单人完成的记录就是权威真值，固定为
`annotation_protocol=single_project_owner`；不设置第二审核人、不计算 inter-reviewer
agreement。晋级训练仍要求置信度不低于 0.90、queue SHA-256 匹配、无敏感属性、非
`UNCERTAIN`，并保留 reviewer ID、时间和来源哈希。`RELOCALIZE` 必须有一个校正目标框，
可观测 relation 必须有一个参照框；不做成对 bbox IoU。同图样本始终位于同一 split。

## 数据是否足够

不以总条数判断，而以学习曲线和分层置信区间判断。若新增一批后验证收益仍上升，或某个
`模型族 x 负例类型` 的有效样本少于 200 条，则继续标注。若连续两批新增 500 个 image-group
后 HR/FG@Neg 改善小于 1 个百分点，或改善只能通过超出预算的 FNR/mIoU 损失获得，则停止扩充，
转而修改表征或损失。

## 联合验收

- T1：HR 下降，同时新增 FNR 不超过 3 个百分点。
- T2：FG@Neg 下降，正样本 mIoU 损失不超过 0.005，新增 false reject 不超过 3 个百分点。
- T4：Neg HR 下降，正样本 mIoU 损失不超过 0.005，caption 不被静默改写。
- 分别报告基础模型、RL/推理模型、BOH 与 ROH；任何一类的提升不能由全局拒答掩盖。
