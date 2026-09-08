# V-SIGHT 实验结论汇总（2026-09）

评测集：RefCOCOg-500 修复版（500 图组 × object/co_occurrence/attribute/relation 反事实，
每模型 7500 条 T1/T2/T4）。BOH = {object, co_occurrence}，ROH = {attribute, relation}。
数值均为服务器实测，脚本与产物在实验工作区（本文件为可公开的结论摘要，隐去内部路径/凭据）。

## 1. 问题层：ROH 系统性比 BOH 难，且与 VQA 解耦

**模型级（11-model 评测报告）**：ROH−BOH gap 广泛存在——T1 判别幻觉率 gap 11–31pp、
T2 前景误定位 gap 15–33pp、T4 gap 27–34pp。

**VQA↔grounding 解耦**：逐 item join T1（判别 VQA）与 T2（定位），池化 5,500 正样本，
**P(定位正确 | 判别正确) = 0.35，φ = 0.12**。模型答对"表达是否匹配"后仍只有 35% 能把框定对。
极端个案 TreeVGR：判别 0.99 / 定位 mIoU 0.281。

**verifier 级（train-free vs VLM 验证，500-dev separation AUROC）**：
| | train-free | VLM verifier | 
|---|---|---|
| BOH | 0.497 | 0.791 |
| ROH | 0.495 | 0.670 |
train-free 几乎随机；VLM 语义验证是解法；ROH 残余更难（relation 类 0.656 最难）。

## 2. 因果机制（11 篇上游模型论文原文核实）

**成因一 — 正样本-only RL → 系统性"必画框"**：RL grounding 模型训练几乎全为正样本，奖励=IoU，
不惩罚"给不存在目标画框"，无拒答动作 → 策略收敛到无条件出框，FG@Neg 飙高。
**剂量-反应证据**：唯一在训练中处理负样本 + 惩罚自信错框 + 有 "No Objects" 弃权的模型（Visual-RFT）
FG@Neg 最低（28.2%）；纯正样本/纯 IoU/必出框模型是极端幻觉端（UniVG-R1 99.8%、Seg-zero 97.1%、LENS 94.2%）。

**成因二 — 坐标 token 与判别不共享有效梯度 → 解耦**：上游模型都是 VLM 把 bbox 当文本坐标 token
自回归输出、无独立检测头；判别走海量优化的识别通路，坐标 token 在 LM 损失下几乎不受罚，且 RL 的
IoU 奖励仅在超阈值时给正梯度，故"物体对但框粗"拿不到收紧梯度。

**由此**：BOH（存在性）落在被重度训练的判别维度、外部信号可抓；ROH（物体在场、仅绑定错）无任何上游
监督覆盖"关系不成立就拒绝"，检测器一阶信号对其近乎盲视。

## 3. 方法层：即插即用 verifier 在真实上游模型上降幻觉

**报告自身 T2 指标 before→after（B1a+B1b 集成，leave-one-model-out，正样本 mIoU 损失 ≤0.005）**：
| 上游模型 | 正样本 mIoU | FG@Neg（幻觉率）↓ | BOH FG↓ | ROH FG↓ |
|---|---|---|---|---|
| Seg-zero | 0.522→0.517 | 97.1%→66.7%（−30.4） | 95.3→51.3 | 98.9→82.1 |
| LENS | 0.487→0.484 | 94.2%→71.8%（−22.4） | 90.4→57.3 | 98.0→86.2 |
定位质量几乎不变，负样本幻觉率大幅下降；"False reject 上升"多为拒绝已经错的框（LENS 丢失 25 框中 24 个本就 IoU<0.5）。

**verifier 判别力（correct-vs-halluc separation AUROC，跨未见上游泛化）**：集成后 LENS 0.911/0.779、
Seg-zero 0.904/0.765（BOH/ROH），全部上游 ROH<BOH 一致保持。

## 4. 成本：分层触发 + 更小骨干

- **便宜信号侦察**：检测器一阶信号抓 BOH（AUROC 0.66–0.71）但抓不到 ROH（0.51–0.58 ≈ 随机）。
  这解释了早期学习式 MLP verifier 为何失败——信号本身对 ROH 无判别力，非模型问题。
- **3B 骨干**：同弱标签/契约下，3B 保住 7B ~90% 的 ROH 判别力（LENS 0.68 vs 0.74），远超便宜头。
- **分层级联**：检测器判 BOH、物体在场才升级 VLM。~75% VLM 调用率下，池化 catch@FNR≤3pp 达
  BOH 0.309 / ROH 0.203，**超过纯便宜头（0.251/0.161）与纯 VLM（0.227/0.147）两个端点**。
- **成本护城河表述**：不是"零额外开销"，而是自适应分层预算（平均 ~0.75 次 3B VLM + 1 次检测器前向），τ 可调。

## 5. 干净的负结果（诚实记录）

- **decoder-trajectory TRACE**：trajectory-only separation AUROC ≈ 0.50，已弃用为诚实 ablation。
- **train-free 注意力信号**：4 个 RAFT 注意力信号 AUROC 全部 <0.5（对绑定正确性系统性反相关），
  作为即插即用信号直接失败（对 MTLA 类"注意力内证据"方法的正面 rebuttal）。
- **负例工程重训（真实语义幻觉负例）**：整体退化、未救回最难上游，说明干净的空间负例是更优训练配方。

## 6. 数据工程

- reference-box 复核：GroundingDINO 生成 reference 候选（99% 覆盖）后，用旗舰视觉模型做第一遍预筛
  （三态 triage：auto_accept / conflict_multi / none_valid），人工只做二轮确认，把全人工降到重点复核。
- 发现评测侧一个坐标空间 bug：某归一化坐标上游模型的定位被评测 harness 系统性低估（修正后 mIoU 提升数倍），
  只影响该单一模型，主结论不受影响。
