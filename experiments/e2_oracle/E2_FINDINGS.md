# E2 — 证据 oracle 诊断（首轮结果）

**日期：2026-09-08　执行：vlm1，3B verifier（Qwen2.5-VL-3B + B1b LoRA）固定**
**数据：pilot_300 人工金集（334 verified，E0 已验证与 dev500/bench1996 不相交）**
**对应路线图：4.4（E2）、12.2 Gate 1 决策分叉**

---

## 0. 一句话结论

给对目标框只能部分缩小缺口，参照物证据（当前呈现方式）无效 → **ROH 瓶颈主要在关系判别能力本身，不是纯粹的证据获取**。这指向路线图 §4.4 的「E2-e 仍弱 → 关系能力/监督是瓶颈，多轮编排救不了」分支。

**重要边界**：pilot 金集的正例全部来自 ROH 组，本轮实质是**全 ROH 绑定诊断**（BOH 无正例，AUROC=nan）。BOH 侧结论沿用既有 cheap-signal 证据，不在本轮。

---

## 1. 三个证据条件（固定 verifier，只改画在图上的框）

| 条件 | 画什么 | 测什么 |
|---|---|---|
| E2a_agent_target | agent 原始 target 框（红） | verifier 原始工作：能否抓错实例 |
| E2c_human_target | 人工正确 target 框（红） | 目标选择 oracle 上界 |
| E2d_joint_boxes | 人工 target（红）+ 人工 reference（蓝） | 参照物证据对关系绑定的贡献（S4 价值） |

从不把正误答案喂给 verifier，只给对象位置。人工框是金标，可能超出检测器候选池 → 本轮测的是**感知/证据上界**，非固定候选池可恢复性（§4.4）。

## 2. 关键数字（全 ROH）

### 2.1 wrong-instance 配对（n=111，同图同表达，agent 错框 vs 人工正确框）

- agent 错框 mean P(yes) = **0.695**
- 人工正确框 mean P(yes) = **0.884**（+0.189）
- 换成正确框后 P(yes) 明显上升(>0.05) 的：35/111；下降的：6/111

### 2.2 verifier 抓错框的能力（E2a 在 wrong-instance 上）

| 阈值 | 错框被压到阈值下的比例 |
|---|---|
| P(yes)<0.5 | 31/111 = **27.9%** |
| P(yes)<0.7 | 35/111 = 31.5% |
| P(yes)<0.9 | 45/111 = 40.5% |

### 2.3 正例保护（pos_correct，n=123）

- 正确框 mean P(yes) = **0.92**（应高，符合预期）

### 2.4 参照物证据贡献（E2d vs E2c 配对，n=253）

- mean delta（joint − target_only）= **−0.025** → 画参照物蓝框**没帮助，反而略降**

## 3. 解读与 Gate 1 影响

1. **错实例是 ROH 幻觉的核心难点**：verifier 在 agent 错框上平均还给 0.695 的 yes 概率，只有 27.9% 被压到 <0.5。错实例框落在正例流形附近，与「BOH 易、ROH 难」的既有判断一致。
2. **目标选择是部分瓶颈但不足**：换人工正确框把 wrong-instance 的 P(yes) 抬到 0.884，但这主要是「正确框本就该 accept」，并没有证明 verifier 能主动**拒掉**错框。真正的失效是错框上给高分。
3. **参照物证据当前呈现无效（负信号）**：画蓝框 reference 反而略降 P(yes)。这弱化了「S4 参照物 → 提升关系绑定判别」的直接路径**在这种输入呈现下**。不等于参照物真值无用（§6.3 备选：缓存特征上的 pair scorer、角色条件化输入仍待测），但「整图 + 画框」的朴素呈现不 work。

**决策分叉落点（§4.4）**：E2-e 类（正确目标 + 参照物 + 全图）未能显著救回 → 偏向「关系能力/监督是瓶颈」，指向 §6 的关系训练假说（H-role 角色条件化输入、H-objective 联合目标），而非优先扩 Agent 轮次或 backbone。

## 4. 混淆排除：zero-shot 3B 对照（已跑）

担心「B1b LoRA 训练负例是空间抖动/低 IoU，对语义实例错误天然不敏感」→ 用 **zero-shot 3B（不加 LoRA）** 跑同样三条件对照。

| 指标（全 ROH） | zero-shot 3B | B1b LoRA 3B |
|---|---|---|
| 错框 mean P(yes)（越低越好） | 0.760 | **0.695** |
| 错框被压到 P(yes)<0.5 | 12.6% | **27.9%** |
| pos-correct vs wrong-instance AUROC @agent框 | 0.595 | **0.656** |
| oracle delta（正确框−错框 P(yes)） | 0.032 | **0.189** |
| 参照物贡献（E2d−E2c） | −0.028 | −0.025 |

**结论（钉死 Gate 1）**：
1. **LoRA 比 zero-shot 更好，不是更差** → 「关系判别弱」**不是** LoRA 训练配方错配造成的；LoRA 是当前最优配置。排除「重做训练配方即可解决」的乐观假设。
2. **两模型都抓不住错实例**（zero-shot 更差，只压住 12.6%）→ 这是 3B verifier 在 ROH 错实例上的**能力上限**，非偶然 checkpoint。
3. **参照物画框两模型都是负贡献**（−0.025/−0.028）→ 呈现方式问题与 LoRA 无关。

→ 落在 §4.4「E2-e 仍弱 → 关系能力/监督是瓶颈，多轮编排难独立解决」；指向 §6 关系训练（H-role 角色条件化输入、H-objective 联合目标），而非优先扩 Agent 轮次或 backbone。

## 5. 混淆排除：参照物文字形式对照（已跑，E2 收尾）

担心 E2d 无效是「蓝框视觉干扰」造成 → 加 **E2e_reftext**：同样画人工正确 target 框（红），但参照物用**文字**写进 prompt（`the reference object is: "<ref phrase>"`），不画蓝框。170 条同时具备三条件的配对：

| 条件 | mean P(yes) | vs E2c delta | AUROC(pos-correct vs wrong-instance) |
|---|---|---|---|
| E2c 仅目标框 | 0.901 | — | 0.572 |
| E2d 蓝框参照物 | 0.874 | −0.027 | 0.593 |
| **E2e 文字参照物** | 0.889 | −0.012 | **0.566** |

**结论**：三条件判别力都在 0.57–0.59，**文字形式不优于画框，也不优于完全不给参照物**。排除「蓝框视觉干扰」——问题不在呈现方式，而是 3B verifier **用不上参照物证据**来做关系绑定判别。「参照物证据（当前任何呈现方式）无效」完整成立。

→ **E2 结论定稿**：ROH 瓶颈是关系判别能力本身，证据侧（正确目标框 / 画框参照物 / 文字参照物）都救不回来。进 §6 关系训练。

（可选后续）固定候选池 oracle-T/R：本轮人工框是感知上界；如需可补一版限制在检测器候选内的可恢复性对照（§4.4 二者分列）。

## 5b. 补充对照：引导式关系推理 + 上游自复核（不加新模型/显存）

用户问：不加 3B、不做路由，只用**更强引导让 VLM 做一次前向关系推理**是否可行？并预判「对 positive-only-RL 上游自复核效果差」。约束：不引入新参数/显存。

设计：verifier 仍是已在显存里的模型，只把 flat yes/no prompt 换成强制 **Step1 target → Step2 reference 定位 → Step3 relation 方向/角色判定 → Step4 verdict** 的引导 CoT，生成后在 `Final answer:` 读 yes/no logit（连续 P(yes|推理)）。矩阵：base-3B(guided/flat)、SegZero-7B 自复核(guided/flat)。

**关键前置发现**：LoRA-3B 在 flat yes/no 上训练过头，**完全不follow引导格式**（只吐表达残句，如 "the giraffe"）→ 引导推理必须用保留指令跟随的 **base 3B**，不能用 LoRA。这本身是个结论：flat 专用微调会牺牲引导推理能力。

结果（ROH，E2a=在 agent 自己的框上区分正确 vs 错实例）：

| verifier | AUROC(正确 vs 错实例) | flag<0.5 | mP(yes)正例 | mP(yes)错框 | avg_gen tok |
|---|---|---|---|---|---|
| **LoRA-3B flat（E2 基准）** | **0.656** | 0.279 | 0.92 | 0.695 | 0 |
| base-3B flat | 0.595 | 0.126 | 0.847 | 0.760 | 0 |
| base-3B GUIDED | 0.601 | 0.297 | 0.695 | 0.600 | 129 |
| SegZero 自复核 flat | 0.599 | 0.288 | 0.853 | 0.677 | 0 |
| SegZero 自复核 GUIDED | 0.632 | 0.261 | 0.937 | 0.737 | 162 |

**结论**：
1. **引导推理没有解锁 ROH 判别**。AUROC：base 0.595→0.601（+0.006 噪声级）、SegZero 0.599→0.632（+0.033 小幅）。三个引导配置全部 **≤ LoRA-3B flat 基准 0.656**。引导整体拉低 P(yes)（更爱说 no）→ 低阈值抓取率虚高，但 AUROC 几乎不动 = **判别力没变强，只是变保守**。部署上会直接违反 FNR≤3pp 正例保护。
2. **用户预判 1 部分成立（方式不同）**：SegZero 自复核 flat 没崩（追平外部 base-3B），但**引导版把正例 P(yes) 冲到 0.937** = 自我辩护/信心虚高 → IVT self-correction mirage 的实证。自复核不 rubber-stamp 一切，但引导会让它给自己背书。
3. **用户预判 2 成立**：3B/7B 层级「更强引导 + 前向推理」无效，E2「ROH=判别能力瓶颈」结论在引导下依旧成立。
4. **成本账（§4.3 预算诚实）**：引导每条多烧 130–160 输出 token，换 AUROC +0.006~0.033，远撑不起额外 compute，且尚未跟 M3-repeat 等预算重复采样比就已落后。

→ **不改结论**：在「不加新模型」约束下，引导推理与上游自复核都救不回 ROH 判别 → 仍指向 §6 关系训练。真正可能有跳变的是 flagship 层级，但那违反「不引入新模型/显存」的约束，不在本轮。

## 6. 产物

```
experiments_v2/e2_oracle/
  e2_items.jsonl                       # 334 item（金集构造，pos_correct123/wrong_instance133/target_absent78）
  e2_evidence_oracle.py                # runner（固定3B，三证据条件，读 P(yes)）
  results/e2_oracle_out.jsonl          # 逐项 P(yes)
  results/e2_oracle_out_analysis.json  # 分条件/BOH-ROH AUROC
```
