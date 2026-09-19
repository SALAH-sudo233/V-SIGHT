# GRPO Verifier 训练文档（Route A：3B KEEP/REJECT 绑定 verifier）

用 GRPO 把 Qwen2.5-VL-3B 优化成可验证奖励驱动的 KEEP/REJECT 绑定 verifier。
本文件记录完整可复现的训练流程、环境、命令、评测口径与结果。

---

## 1. 目标与判据

- **优化目标**：给定 (图像, 指称短语, bbox)，让 3B 输出 typed-JSON 决策 `KEEP`（短语正确描述该框）/ `REJECT`（短语被幻觉/绑定错误），
  用 **完全可验证的奖励**（无 reward model）推动决策准确率，重点抬升 ROH。
- **成败判据**（训练前定死，防事后找补）：
  - 正向：ROH decision 准确率从零样本 **0.640** 显著上行（目标 >0.70），且 BOH 不退化。
  - 诚实负向：若只有 BOH 涨、ROH 不动 → 印证 "ROH 是关系判断能力瓶颈、RL 也难救"（同样有论文价值，导向 sec6 relation training）。

## 2. 环境（隔离，不污染生产评测环境）

- 独立 conda env `grpo_ayb`（克隆自 `mllm_ayb`）。
- torch 2.12.1+cu126 / transformers 5.12.1 / **ms-swift 4.5.3** / trl 0.29.1（swift 自带 GRPO trainer）/ peft 0.19.1 / msgspec 0.21.1。
- 必设 `export LD_LIBRARY_PATH=$HOME/.miniconda3/envs/grpo_ayb/lib:$LD_LIBRARY_PATH`（修 GLIBCXX_3.4.29）。
- 机器：vlm1，8×RTX4090-24G。基座本地缓存 `Qwen/Qwen2.5-VL-3B-Instruct`。

### 为什么用 ms-swift 而非 TRL / vllm

TRL 1.12 的 GRPO rollout 在 Qwen2.5-VL 上崩（`get_rope_index` position_ids 形状不匹配，
经隔离实验证明是 TRL 手写 position_ids 拼接的 bug，非 transformers）。绕过方案对比：

| 方案 | 代价 | 采用 |
|---|---|---|
| vllm 任意版本做 rollout | **强制重装 torch**（0.11.2 连 transformers 都降到 4.57） | ✗ |
| 手动 patch TRL 库代码 | 不可移植 | ✗ |
| **ms-swift 4.5.3** | 不动 torch/transformers（仅 trl→0.29.1），用自己的 GRPO trainer | ✓ |

## 3. 数据构造

- 源：`refcocog_1996_heldout.manual_v2.json`（1996 图 × 4 反事实 = 7984 记录）。
- 每条反事实对 → 2 个 verifier prompt：正文本→`KEEP`，负文本→`REJECT`。
- 产出 **15968 prompt**，KEEP/REJECT 完美平衡（各 7984），四类幻觉各 3992，零缺图。
- **合法性前提（已核验）**：`500-dev`（500 图）∩ `1996`（1996 图）= **0 图像重叠**，
  故 1996 训练 + 500-dev 评测不构成 train-on-test。**约束**：训练用 1996，评测只用 500-dev。

swift 数据格式（`prep_1996_swift.py`）：
```json
{"messages":[{"role":"user","content":"<image>Look at the region [x0,y0,x1,y1]. Does the phrase \"...\" correctly describe... Answer ONLY JSON {\"decision\":\"KEEP\"|\"REJECT\",\"confidence\":0-1}"}],
 "images":["/abs/path.jpg"],
 "solution":"KEEP"}
```

## 4. 奖励函数（`vsight_reward_plugin.py`，完全可验证）

外部 plugin 注册两个 ORM，dataset 的 `solution` 列自动作 kwarg 传入：

| reward | 权重 | 规则 |
|---|---|---|
| `vsight_format` | 0.3 | 输出合法 typed-JSON（含 decision+confidence）→ +0.3 |
| `vsight_decision` | 1.0 | decision 命中 `solution` → +1.0；解析失败 → −0.2；不命中 → 0 |

## 5. 训练命令

```bash
export LD_LIBRARY_PATH=$HOME/.miniconda3/envs/grpo_ayb/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,2,3,4 NPROC_PER_NODE=4
swift rlhf \
  --rlhf_type grpo \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --external_plugins vsight_reward_plugin.py \
  --reward_funcs vsight_format vsight_decision --reward_weights 0.3 1.0 \
  --tuner_type lora --lora_rank 8 --lora_alpha 16 \
  --target_modules q_proj k_proj v_proj o_proj \
  --dataset grpo_1996_swift.jsonl \
  --num_generations 8 --max_completion_length 64 \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 1 \
  --num_train_epochs 2 --learning_rate 1e-5 --temperature 1.0 --beta 0.001 \
  --use_vllm false --torch_dtype bfloat16 --gradient_checkpointing true \
  --log_completions true --logging_steps 1 --save_steps 200 \
  --output_dir runs/swift_full
```

- 4 GPU DDP，全局 32 prompt/step ×8 gen = 256 rollout/step，~7.5 s/it，单卡显存 ~12GB。
- 全量 2 epoch ≈ 16h（无 vllm 加速）；checkpoint 每 200 步保存。

## 6. 评测口径

`eval_decision_acc.py`：从 **500-dev** 同样构造 KEEP/REJECT prompt（图像与训练集不相交），
贪心生成、解析 decision、对金标算准确率，按 BOH/ROH 分组。`--lora ""` = 零样本基线。
- 全量评测 = 2000 记录 × (KEEP+REJECT) = **4000 prompt**。
- 500-dev 图像根：`$DATA/refcoco/train2014`（服务器上的 COCO train2014 镜像）。

## 7. 结果

### 7.1 早期锚点（400 抽样，快速判读）

| 组 | 零样本 | GRPO ckpt-200 (~0.4 epoch) | Δ |
|---|---|---|---|
| ALL | 0.710 | 0.730 | +2.0pp |
| BOH | 0.780 | 0.790 | +1.0pp |
| **ROH** | **0.640** | **0.670** | **+3.0pp** |

早期信号：提升主要落在 ROH，方向正确；但 400 样本有 ±3pp 噪声，需全量确认。

### 7.2 全量评测（500-dev 全量 = 4000 prompt，权威数字）

| checkpoint | 训练步 | ALL | BOH | ROH | BOH−ROH gap | parse_rate |
|---|---|---|---|---|---|---|
| 零样本 | 0 | 0.6893 | 0.7480 | 0.6305 | 11.8pp | 1.000 |
| ckpt-200 | 200 | 0.7043 | 0.7675 | 0.6410 | 12.7pp | 1.000 |
| ckpt-400 | 400 | 0.7023 | 0.7385 | 0.6660 | 7.3pp | 1.000 |
| ckpt-600 | 600 | 0.7137 | 0.7495 | 0.6780 | 7.2pp | 1.000 |
| ckpt-1000 | 1000 | 0.7163 | 0.7535 | 0.6790 | 7.5pp | 1.000 |
| **ckpt-1400** | 1400 | **0.7228** | 0.7655 | **0.6800** | 8.6pp | 1.000 |
| ckpt-1600 | 1600 | 0.7173 | 0.7560 | 0.6785 | 7.7pp | 1.000 |
| ckpt-7984 (末) | 7984 | 0.7100 | 0.7420 | 0.6780 | 6.4pp | 1.000 |

**结论（正向但快速饱和，趋势诚实）：**
1. **ROH 提升集中在前 600 步**：0.6305→0.641→0.666→0.678，**+4.75pp**；此后 600→7984 步 ROH 几乎不动
   （0.678→0.679→0.680→0.679），**在 ~0.68 撞墙**。关系幻觉难度被低估——GRPO 能救但非线性无限涨。
2. **BOH 全程守住** ~0.75，gap 从 11.8pp 收窄并稳定在 7–8pp。
3. **最优点是 ckpt-1400**（ALL 0.7228 峰值），**不是**最终 ckpt-7984 —— 后期长训对 ROH 边际递减、有轻微过训。
   已 cp 保护 `saved_ckpts/ck1400` 与 `ck7984_final`。
4. parse_rate 全程 1.0，预算不浪费在格式。
5. **诚实提示**：400-抽样早期锚点（§7.1）ROH +3pp，全量 4000 校正后仅 +1pp；**以全量数字为准**。

**对论文的意义**：E2 曾判定 "ROH=关系判断能力瓶颈，evidence 侧救不了，指向 sec6 relation training"。
本实验既给出 sec6 的**第一个正向证据**（可验证奖励 GRPO 让 3B ROH 真实 +5pp，推翻"小 verifier 对 ROH 无望"），
又标出了**天花板 ~0.68**：单纯堆 RL 步数在 600 步后饱和，进一步提升需 relation-training 层面的结构改进
（role-conditioned 输入 / 关系联合损失），而非更多 GRPO 步数。提升→饱和曲线比单调直线更可信。

### 7.2-old 早期 400-抽样锚点（见 §7.1，仅快速判读，勿作正式数字）

### 7.3 训练曲线要点

- smoke（256×40 步）：decision reward 0.664→0.715，KL~0.0005，format 全程满分。
- 全量早期：decision reward 前 73 步 0.557→0.617。format 全程 0.3 满分、parse_rate 1.0（预算不浪费在格式）。

## 8. 复现步骤

```bash
# 1. 造 swift 数据（在有 1996 json + 图像的机器）
python prep_1996_grpo.py <1996.json> <img_dir> grpo_1996_verifier.jsonl 0
python prep_1996_swift.py grpo_1996_verifier.jsonl grpo_1996_swift.jsonl <img_dir> 0
# 2. 训练（见 §5）
# 3. 评测各 checkpoint（见 §6）
python eval_decision_acc.py --dev <500dev.json> --img_dir <root> --model <base> --lora <ckpt|"">
```
