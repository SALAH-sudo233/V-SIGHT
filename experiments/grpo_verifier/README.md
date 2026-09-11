# GRPO Verifier Training (Route A) — 探索分支

用 GRPO 把 Qwen2.5-VL-3B 优化成 **KEEP/REJECT 绑定 verifier**（不是定位器），
奖励完全可验证、无 reward model。这是飞轮训练引擎的候选实现，仍在验证 pipeline 阶段。

> 状态：**WIP / 未合并**。数据与基座已就绪，卡在一个 TRL 框架层 bug（见下）。
> 不要合并进 main，直到 smoke test 通过并给出 reward 上升曲线。

## 方法定位

- **Route A（本分支）**：3B 当 verifier / 选框器，接续 B1b LoRA 的贡献主线。
  奖励可验证：格式合法性 + 决策是否命中金标。
- Route B（未采用）：3B 当定位器（IoU reward）——与上游 grounding 模型撞车，非本项目贡献点。

## 数据构造（`prep_1996_grpo.py`）

- 源：`refcocog_1996_heldout.manual_v2.json`（1996 图 × 4 反事实 = 7984 记录）。
- 每条反事实对展开成 **2 个 verifier prompt**：
  - `(image, positive_text, gt_bbox)` → 金标 `KEEP`
  - `(image, negative_text, gt_bbox)` → 金标 `REJECT`
- 产出 **15968 prompt**，KEEP/REJECT 完美平衡（各 7984），四类幻觉各 3992，零缺图。

### 关键：为什么 1996 可以做训练集

`500-dev`（500 图）与 `1996-heldout`（1996 图）经 COCO 文件名核验 **图像级零重叠**。
因此 1996 训练 + 500-dev 评测 **不构成 train-on-test**（held-out 身份转移到 500-dev）。
**约束**：若在 1996 上训练，则最终评测只能用 500-dev，绝不能再在 1996 上报数。

## 奖励函数（`grpo_verifier_smoke.py`，完全可验证）

| 组件 | 取值 |
|---|---|
| `reward_format` | 输出合法 typed-JSON（含 decision + confidence）+0.3 |
| `reward_decision` | decision 命中金标 +1.0；解析失败 −0.2；不命中 0 |

后续正式版可加：target_id 命中 +0.5、relation_check 与 hallucination_subtype 一致 +0.5。

## 环境

- 隔离 conda env `grpo_ayb`（克隆自 `mllm_ayb`，不污染生产评测环境）。
- torch 2.12.1+cu126 / transformers 5.12.1 / trl 1.12.0 / peft 0.19.1。
- `export LD_LIBRARY_PATH=$HOME/.miniconda3/envs/grpo_ayb/lib:$LD_LIBRARY_PATH`（修 GLIBCXX）。
- 机器：vlm1（8×RTX4090-24G，模型已缓存）。

## VLM 数据格式陷阱（已解决）

不要把图像嵌进 Arrow 序列化的 `prompt` content：混放 `{type:image,image:path}` 与
`{type:text,text:..}` 会让 Arrow 统一 key（给每个 dict 补 `image:None`/`text:None`），
破坏 `apply_chat_template` 的占位符计数（`StopIteration`）。
正确做法：prompt content 的 image block 留空 `{type:image}`，图像放**独立 `image` 列**并
`ds.cast_column("image", datasets.Image())`；TRL 1.12 会读该列并注入。

## 已知 BLOCKER（诊断完成，未修复）

TRL 1.12 的 rollout/score forward 在 Qwen2.5-VL 上崩：
```
get_rope_index ... position_ids ... value tensor of shape [3,551]
cannot be broadcast to indexing result of shape [3,474]
```
**隔离实验证明是 TRL 的 bug，不是 transformers**：base `Qwen2_5_VLForConditionalGeneration`
的 batched forward（带 padding，2 图）正常，单条 generate 也能输出正确 JSON。
transformers 已知回归 #44479（v5.3 rope）已由 #44474 修复，5.12.1 含该修复。
问题在 TRL 手写的 prompt_ids / position_ids 拼接。

### 绕过方案（决策中）

- **用较旧的 vllm（0.10/0.11.x，兼容 torch 2.12）做 rollout**，绕开 TRL 手写 position_ids 路径，
  且不升 torch/CUDA（vllm 0.28 会强升 torch→2.13 + CUDA13 全套，代价过大）。← 当前选择
- 备选：手动 patch TRL 的 `_tokenize_prompts`/`_generate`；或降 TRL 版本。

smoke 脚本当前 `use_vllm=False` 以隔离框架；vllm 装好后翻开。奖励函数已手工验证正确。

## 复现

```bash
# 1. 造数据（在有 1996 json + 图像的机器上）
python prep_1996_grpo.py <1996.json> <img_dir> grpo_1996_verifier.jsonl 0

# 2. smoke（vllm 就绪后加 --use_vllm）
export LD_LIBRARY_PATH=$HOME/.miniconda3/envs/grpo_ayb/lib:$LD_LIBRARY_PATH
python grpo_verifier_smoke.py --n 200 --steps 30 --num_gen 8 --out runs/smoke
```

**PASS 判据**：reward 均值随 step 上升 + format_ok>90% + 无长度爆炸。仅验证 pipeline，不追求增益。
