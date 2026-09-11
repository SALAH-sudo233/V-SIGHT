# E6：监督训练与 500-dev 阶段性归档

日期：2026-09-09。本次为结果归档，不表示完整十一模型实验已完成。

## 1. 训练与 holdout

prepared fix6 数据包含 pilot 人工数据与 paired240。VLM rows 使用 image-group split：train 4073、calibration 840、holdout 854。3B LoRA rank=8，flat/struct 两种目标，平衡后训练 855 行、2 epochs。

| verifier | candidate AUROC | ROH 子集 AUROC |
|---|---:|---:|
| zero-shot 3B | 0.856 | 0.780 |
| flat LoRA | 0.897 | 0.854 |
| structured LoRA | 0.868 | 0.811 |

混合候选 holdout 上 flat 优于 zero-shot 与 structured；不能据此证明完整关系理解，也不能与不同数据集直接比较。线性头 relation macro-F1 0.598、attribute macro-F1 0.382；这些任务/指标不同，不能与 AUROC 直接相减。

## 2. E2 wrong-instance 诊断

历史运行报告的 AUROC：zero-shot 0.595、旧 b1b LoRA 0.656、E6 flat 0.624；E6 flag<0.5 为 0.297。该次运行没有显示提升，但不支持“3B 能力天花板”的结论。

**重要限制与对先前解读的更正：**

- E6 源数据包含 pilot，而 E2 也来自 pilot。必须核验实际训练 image IDs 与 E2 评估 IDs 的交集；在此之前，不能将全 pilot 复评称作独立泛化测试。
- `relation_status=verified` 不证明 agent 错框也是合法目标；该状态可能描述人工修正后的 target/reference。`unverified` 也不等于关系错误。
- 历史分组计数为 verified 100、unverified 10、contradicted 1，不能据此宣称“90% 双合法目标”或“只有 11 条真错误”。
- 单条 contradicted 样本相对于共享正例池的 AUROC 不能说明该类别已被可靠解决。
- 负例难度、标签含义、训练覆盖与模型判别能力均是待检验解释；尚未确证几何捷径或数据饥饿是根因。

## 3. 500-dev：四个上游的已有输出评测

将 E6 flat 与旧 b1b-3B 分别用于已有上游 T2 prediction boxes。以下为此前同一指标实现重算的全 500-dev 描述性结果；尚未完成逐 key/候选框一致性审计与 E0 独立 calibration/development 路由复算。

| 上游 | ROH AUROC 旧→E6 | BOH AUROC 旧→E6 | catchROH@3pp 旧→E6 |
|---|---:|---:|---:|
| LENS | 0.6792→0.7714 | 0.8057→0.8967 | 0.142→0.259 |
| Seg-zero | 0.6515→0.7511 | 0.7779→0.8845 | 0.098→0.214 |
| Orsta-7B | 0.5830→0.6229 | 0.6052→0.6751 | 0.055→0.066 |
| Qwen3-VL-8B | 0.7126→0.5706 | 0.7354→0.6092 | 0.206→0.222 |

E6 原始运行汇总见 `e6_500dev_eval_analysis.json`。

### 解释与协议限制

- 三个上游的描述性 AUROC 上升，支持监督训练在当前候选—表达一致性任务上具有价值；未计算置信区间，不称统计显著。
- catch@3pp 是使用评估正例分布选阈值的探索性数值，不代表部署时或独立 development 上保证 FNR≤3pp。论文须使用 E0 冻结 calibration 阈值后评估 development。
- Qwen3-VL 的 AUROC 下降必须保留。该运行 n_correct=12，历史上存在坐标协议问题，但本次未验证此输出版本是否修复，不将退化直接归因于 bug。
- 这轮读取四个模型的既有预测，不等于重新加载全部十一模型权重，也不等于完成 leave-one-upstream-out 训练协议。训练源身份/图像隔离仍需审计后才能主张未见上游迁移。
- ROH 合并 attribute 与 relation；聚合提升不能独立证明关系子类改善。需要分别报告两类指标。

## 4. 研究范围

当前优先目标是有限预算下筛查候选—表达不一致，而非重做完整 grounding。500-dev 的正面信号值得作为主线继续验证。

多实例定位、候选恢复与完整指称消解可以作为扩展/边界任务；但若已有候选明确违反表达的关系约束，判断它错误仍属于 verifier 范围，不能因表现弱就事后排除。E2 保留为困难诊断，不据其当前结果断言模型规模上限，也不声称已证实多合法目标歧义。

## 5. 尚未完成

- [ ] E6 train 与 E2/500-dev 的实际 image-ID 交集及来源审计。
- [ ] 新旧输出逐 key、候选框、query 和失败样本一致性核验。
- [ ] 按 E0 fusion-fit/calibration/development 边界重算阈值指标与分组置信区间。
- [ ] relation/attribute 分开报告，保留 Qwen3-VL 退化与协议版本说明。
- [ ] 完整十一模型比较、实际权重替换控制、report-native pos mIoU / FG@Neg 与成本报告。

不提交权重、图像、凭据或与本轮无关的 GRPO 在途文档。
