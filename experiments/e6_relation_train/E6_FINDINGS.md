# E6 — 关系训练：线性头基线 vs VLM LoRA（首轮）

**日期：2026-09-09　执行：goldenapple（数据）+ vlm1（3B LoRA 训练/评估）**
**对应路线图：§6.1 三训练假说、§6.2 主训练方案、§6.3 小头 vs VLM**
**方案 C：先 A 线性头基线（几秒 CPU）→ 再 B VLM LoRA，形成小头 vs VLM 对照**

---

## 0. 一句话结论

训练**有效**：在 group-split holdout 上，训练后的 3B LoRA verifier（ROH 正例 vs 错框 AUROC **0.854**）显著超过 zero-shot（0.780）和线性头基线（relation macro-F1 ≈0.60，靠预测多数类）。这是路线图 §1.1 主实验命题「方法补足这类模型」的首个正面证据。但与 E2 的 0.656 天花板**尚非严格同底**（见 §4 caveat），需在 E2 纯 wrong-instance 配对上复评确认。

**H-objective 未兑现**：结构化联合目标（relation+attribute+verdict）AUROC 0.811 < 纯 flat yes/no 0.854。这个规模上，联合目标没胜过二分类。

---

## 1. 数据（用户已准备好的 fix6 资产，非新标注）

`goldenapple:~/anyibo/flywheel_pretrain/qwen7b_round_20260901/`
- 源：D0 = pilot 334 人工核验 + D1 = paired240 1199（fix6 audit `valid:true`，7521 行零重复零泄漏）
- label schema：`candidate_correct`（0/1）、`relation_label`/`attribute_label`（VERIFIED/UNVERIFIED/CONTRADICTED）、`edge_label`、`action_label`（KEEP/SWITCH/REJECT/DEFER）
- **类不平衡**：relation UNVERIFIED 74%，candidate_correct 仅 441/7521 正 → 这是线性头躺平预测多数类的直接原因
- **group split（fix6 seed=17，内建）**：train 4073 / calibration 840 / holdout 854（按 coco 图像组隔离）
- **泄漏警告**：D0 = E2 诊断用的 pilot 图 → B **只在 holdout 评估**，绝不在全 pilot 上

VLM 训练 rows 由 `build_vlm_rows.py` 从源数据 join 出（image+query+pixel box+labels），533 图全部在 vlm1 refcoco 目录，无需传输。

## 2. A — 线性头基线（§6.3 复用特征的关系头）

`train_qwen7b_verifier.py`（冻结特征 + logistic 多分类头，box_features 8 几何量 + CCV 观测）。fix6 重跑，V1 holdout（n=228 记录）：

| head | accuracy | macro-F1 |
|---|---|---|
| relation | 0.930 | **0.598** |
| attribute | 0.360 | 0.382 |
| edge | 0.706 | 0.360 |
| action | 0.702 | 0.428 |

relation acc 0.93 是假象：confusion 显示几乎全预测多数类 UNVERIFIED（194 里对 180），CONTRADICTED 0/1 全错。**印证 cheap-signal 老教训：几何/浅层特征做关系判别 ≈ 随机偏多数类。** 这是 B 要打败的基线。

## 3. B — VLM LoRA（§6.2 主训练方案）

3B backbone + LoRA r=8（q/k/v/o_proj），train split 4073 行、正例平衡到 neg=2×pos（855 行）、2 epoch。两个监督模式（H-objective 对照）：
- **flat**：单 token yes/no（from candidate_correct）
- **struct**：`relation=<>; attribute=<>; verdict=<>` 联合目标

holdout（854 行）区分正确框 vs 错框：

| 模型 | AUROC(candidate) | AUROC(正例 vs ROH错) | mP(y)正 | mP(y)负 | negflag<0.5 |
|---|---|---|---|---|---|
| base-3B zero-shot | 0.856 | 0.780 | 0.823 | 0.355 | 63.6% |
| **LoRA flat** | **0.897** | **0.854** | 0.661 | 0.198 | 82.6% |
| LoRA struct | 0.868 | 0.811 | 0.421 | 0.161 | 94.7% |

**结论**：
1. 训练有效：ROH 正例 vs 错框 AUROC zero-shot 0.780 → flat LoRA 0.854（+0.074），远超 A 线性头和 E2 0.656。
2. flat > struct（0.854 vs 0.811）：H-objective 联合目标此规模未兑现；struct 过度保守（正例 P(yes) 塌到 0.42）。
3. 训练 loss：flat 0.39 / struct 0.11（struct 低因目标模板化，不代表判别更好）。

## 4. Caveat：与 E2 天花板尚非严格同底（下一步复评）

本 holdout 的「负例」是**混合**的：多数是 detector 普通错候选（几何可分，故 zero-shot 已 0.856），仅一部分是真 ROH 错实例。E2 的 0.656 测的是**纯 wrong-instance**（同图同表达、类别对但绑定错，最难一类）。故「0.854 vs 0.656」不是 apples-to-apples。

**待做（决定性）**：在 E2 的 111 条纯 wrong-instance 配对上评估 e6_lora_flat，直接回答「训练是否把 ROH 错实例判别拉过 0.656」。

## 4b. 决定性复评 + relation_status 子集分析（已跑）

e6_lora_flat 在 E2 的 111 条纯 wrong-instance 上（同底 vs E2 0.656）：

| verifier | AUROC(pos vs wrong-instance) | flag<0.5 |
|---|---|---|
| base-3B zeroshot | 0.595 | 0.126 |
| LoRA-3B b1b（E2 基准） | 0.656 | 0.279 |
| **e6_lora_flat（关系训练）** | **0.624** | 0.297 |

**关系训练没有打破 E2 天花板**（0.624 ≲ 0.656）。holdout 的 0.854 是假象：那里负例多为 detector 普通错候选（几何可分），E2 wrong-instance 才是「类别对、绑定错」的真难例。

按 wrong-instance 的 `relation_status` 拆分（关键诊断）：

| 子集 | n | agent错框 mP(yes) | AUROC(pos vs this) | flag<0.5 |
|---|---|---|---|---|
| verified（可能双合法目标/歧义） | 100 | 0.576 | 0.610 | 0.28 |
| unverified（真·绑定错） | 10 | 0.473 | 0.735 | 0.40 |
| contradicted（真·关系矛盾） | 1 | 0.148 | 0.98 | 1.0 |

**双重结论**：
1. **任务歧义拖累**：占 90% 的 verified 子集 AUROC 仅 0.610；verifier 在真·绑定错子集（unverified/contradicted）分得明显更开（0.735/0.98）。E2 的 0.656「天花板」部分是评测口径问题——大量 wrong_instance 实为「换了个也合法的目标」，给高分未必错。
2. **VLM 关系理解确实未到位**：即便真·绑定错子集，unverified 也只 0.735 且 40% 漏抓。
3. **根因 = 关系矛盾样本饥饿**：pilot 里真·矛盾极少（contradicted 1、unverified 10），fix6 训练集 CONTRADICTED 仅 19/7521。**从训练到评测，"关系被违反"的信号都极度稀缺** → 现规模下无法区分「能力上限」还是「数据饥饿」（真·矛盾 n=11，统计不足）。指向 §5.4「优先构造关系被违反的同图困难负例」+ §6.1 H-data 失败分支「先查标签/难度/源分布」。

## 5. 500-dev 真实上游评测与迁移验证（已跑）

将 e6_lora_flat（3B + LoRA，训练未使用 500-dev）插入 4 个上游的真实 T2 prediction boxes，在同一管线下与旧 b1b-3B LoRA 比较。每个上游均为训练后 verifier 未见过的输出分布；报告 ROH/BOH AUROC 与 FNR≤3pp 下的 catch。

| 上游 | ROH AUROC 旧→e6 | BOH AUROC 旧→e6 | catchROH@3pp 旧→e6 |
|---|---:|---:|---:|
| LENS | 0.679→**0.771** | 0.806→**0.897** | 0.142→**0.259** |
| Seg-zero | 0.652→**0.751** | 0.778→**0.885** | 0.098→**0.214** |
| Orsta-7B | 0.583→**0.623** | 0.605→**0.675** | 0.055→**0.066** |
| Qwen3-VL-8B | 0.713→0.571 | 0.735→0.609 | 0.206→0.222 |

**3/4 个上游提升**，其中 LENS/Seg-zero 的 ROH AUROC 提升约 +0.09~+0.10；说明训练后的 verifier 能迁移识别一类真实 ROH：表达被改写后与上游框不一致（query/expression-side mismatch）。Qwen3-VL 下降且 n_correct=12，受其已知坐标/协议异常与极小正例分母影响，不能作稳定失败结论。

### 5.1 500-dev 与 E2 的互补含义

- 500-dev ROH：主要测**表达—框不匹配**：表达/反事实改变后，当前框不再满足表达。
- E2 wrong-instance：主要测**同图多实例中的指向/绑定消歧**：表达不变，候选框换成另一个实例；其中 100/111 的 `relation_status=verified`，还混有多合法目标歧义。

因此不能把 500-dev 的提升写成“3B 已解决 ROH 关系理解”，也不能把 E2 的弱结果写成“3B 完全不理解关系”。更准确的结论是：**3B 已能学到表达—框一致性判别，但对最难的实例级关系绑定/消歧仍缺乏充分证据，且当前 E2 真正矛盾样本过少。** 这正好支持 V-SIGHT 把 BOH/ROH 与 query/box 两个轴拆开的设计。

## 6. 产物

```
experiments_v2/e6_relation_train/
  build_vlm_rows.py                # 从 fix6 源数据 join VLM 训练 rows（image+query+box+labels+split）
  b_train_lora.py                  # B 训练（flat/struct，正例平衡，3B+LoRA r=8）
  b_eval_lora.py                   # B 评估（仅 holdout，读 P(yes)，AUROC）
  e6_vlm_rows.jsonl                # train4073/calib840/holdout854
  A_linear_head_baseline.json      # A 线性头结果
  results/eval_{flat,struct,zeroshot}_analysis.json
远程 LoRA:
  vlm1:.../roh_boh_gate_a_500dev/results/e6_lora_{flat,struct}
```

## 6. 待办

- [ ] **决定性复评**：e6_lora_flat 在 E2 111 条纯 wrong-instance 上 → 与 0.656 同底比较
- [ ] **500-dev 评测**：将训练后 verifier 插到 11 模型真实 T2 pred boxes 上（report-native gain：pos mIoU / FG@Neg），与既有 B1c ensemble 对比
- [ ] **换上游权重泛化**：leave-upstream-out，验证 verifier 对未见上游有效（§4.6 E4 迁移）
