# V-SIGHT 方向更新与叙事重构（2026-09）

本文件记录 V-SIGHT 的定位重构与本轮实验结论，作为当前权威叙事，取代早期
README/PROGRESS 中"学习式联合验证器 + E2 负结果"的旧框架表述。

## 0. 一句话定位

现有 RL 增强的 grounding 模型靠提高 IoU 刷分，但同时引入了物体幻觉——分数的提升有一部分来自
不可靠的 bbox，分高而无意义。V-SIGHT 不重训上游大模型，而是给出一个**即插即用**的后处理框架：
先把幻觉缓释与定位能力解耦并重新刻画问题，再用一个低数据 verifier 在 `KEEP / SWITCH / REJECT / DEFER`
上修正上游的单次 grounding 结果，最后用 agentic 飞轮持续产出可解释的关系型难例数据。

**命题**：VLM 的 grounding 能力与 hallucination mitigation 能力不是同一个目标。对容易的目标级
grounding，模型已接近饱和；对关系场景中的 target-reference binding，模型仍普遍出现关系型幻觉。
重复调用 VLM 能缓解一部分基础物体幻觉，但代价是推理开销，且不能稳定解决关系型幻觉。

## 1. 术语：BOH vs ROH

- **BOH（basic-object hallucination）**：目标对象本身不存在或类别不成立。瓶颈在于目标是否存在、候选池是否覆盖。
- **ROH（relation-object hallucination）**：目标与参考对象可能都存在，但属性、关系或角色绑定错误。
  瓶颈在于 target 与 reference 是否正确绑定、关系是否在正确实例对上成立、target/reference 角色是否保持一致。

核心发现不是"ROH 比 BOH 难"这么简单，而是二者瓶颈本质不同。即使 target mIoU 很高，模型仍可能
交换 target/reference、选错参考物、误判关系方向。**只报告 target mIoU 或 object existence accuracy
会系统性高估模型对关系表达的理解。**

## 2. 三阶段叙事

**阶段一（问题重刻画）**：IoU=0 样本几乎都是物体幻觉导致的定位偏移；MLLM logit 不携带可直接提取的
属性/空间特征（对比解码等失效）；输出是 bbox 坐标而非语言 token，故"干预注意力影响语言解码"的
物体幻觉工作在本任务机制错位。任务链路：弥合 ROH↔BOH gap → 引入拒答机制 → 候选池+打分缓解视觉漂移。

**阶段二（方向抉择）**：训练记录里 LoRA+CLIP+教师模型可刷到 80+ IoU，但抹掉了 train-free 的低成本优势、
抬高部署成本。决策：基于 verifier 模块做即插即用修正，而非引入更多更大成本的视觉证据。

**阶段三（四贡献）**：问题发现与重刻画 / Trace-Bind 结构化绑定决策 / 低数据 verifier（KEEP·SWITCH·REJECT·DEFER）/
agentic 飞轮定向产出 ROH 难例。

## 3. 本轮实验结论（详见 FINDINGS.md）

- **贡献一有因果机制**：ROH>BOH 与 VQA↔grounding 解耦，是"正样本-only RL 训练 + 坐标 token 无独立梯度"
  两个训练范式选择的必然产物（11 篇上游模型论文原文核实，含 Visual-RFT 剂量-反应反例）。
- **即插即用 verifier 在真实上游模型上降幻觉**：以报告自身 T2 指标衡量，在正样本 mIoU 损失 ≤0.005 约束下，
  高幻觉上游 FG@Neg（负样本前景率）下降 22–30pp。
- **分层成本设计**：BOH 可用便宜检测器信号；ROH 内在需要 VLM 级语义验证。级联在 ~75% VLM 调用率下
  catch 超过纯便宜头与纯 VLM 两个端点。verifier 骨干可用 3B（保住 7B ~90% ROH 判别力）。

## 4. 术语与动作映射（新旧对照）

| 旧序列化（代码保留） | 当前语义动作 | 含义 |
|---|---|---|
| `KEEP` | `ACCEPT` | 保留有支持的上游框 |
| `REJECT` | `ABSTAIN` | 无支持或无安全替代 → 拒答，不返回框 |
| `SWITCH` | `RELOCALIZE` | 对可恢复的错误实例，用验证过的 Top-K 提案改定位 |
| — | `DEFER` | verifier 不确定 → 转人工复核优先级 |

## 5. 诚实边界

- 关系型难例的 reference box / edge 真值仍需人工复核回流（已有 VLM 第一遍预筛 + 人工二轮流程）。
- 单个上游模型（多任务统一 RL 训练者）的幻觉框最难判别，是 verifier 能力边界。
- verifier 弱标签实验为 accept/reject 层结论；SWITCH/RELOCALIZE 的因果正确性需 reference-box held-out 集完成后评估。
