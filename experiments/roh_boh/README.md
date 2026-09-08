# ROH/BOH 实验脚本（2026-09）

本目录是 `docs/updates_2026_09/FINDINGS.md` 所有结论的可复现脚本。脚本原在 GPU 工作区开发，
已把硬编码的服务器绝对路径替换为**环境变量 + 相对默认值**，可移植。

## 路径配置（环境变量，均有默认值）

| 变量 | 含义 | 默认 |
|---|---|---|
| `VSIGHT_IMAGE_ROOT` | RefCOCOg 图像目录（COCO train2014） | `data/refcoco/train2014` |
| `VSIGHT_EVAL11_ROOT` | 11-model 修复版评测输出根（每模型 `records.jsonl`） | `data/eval_11models_500_repaired` |
| `VSIGHT_HELDOUT_1996` | repaired-1996 held-out 反事实集 | `data/refcocog_1996_heldout.manual_v2.json` |
| `VSIGHT_RESULTS` | 中间产物/结果输出目录 | `results` |
| `VSIGHT_EXP` / `VSIGHT_REPO` | 实验/仓库根 | `.` |
| `QWEN25VL_7B` | verifier 骨干（7B） | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `QWEN3VL_8B` | 旗舰 8B（S4 预筛可选，视觉判别） | `Qwen/Qwen3-VL-8B-Instruct` |
| `GROUNDING_DINO` | 检测器（便宜信号 / reference 候选） | `IDEA-Research/grounding-dino-base` |

模型默认走 HF Hub id；本地权重可用绝对路径覆盖。GPU 环境注意 `LD_LIBRARY_PATH` 指向 conda env lib。

## 流水线（按依赖顺序）

**问题层**
- `vqa_grounding_decouple.py` — join T1(判别)×T2(定位)，算 P(定位对|判别对)、φ。
- `s1_roh_boh_separation.py` — train-free 基线的 BOH/ROH separation。
- `s2b_vlm_verify.py` — 画框 + graded yes/no 的 VLM 验证信号（S2b）。
- `s3_attention_ablation.py` — 4 个 RAFT 注意力信号消融（MTLA rebuttal，AUROC<0.5）。
- `gate_a_analysis.py` — grouped-bootstrap 配对 delta + Gate A 判定。

**方法层（即插即用 verifier）**
- `s5_plugin_filter.py` / `s5_analysis.py` — verifier 作用于真实上游模型 T2 预测框，correct-vs-halluc 分析。
- `b1a_fusion_cv.py` / `b1a_calibrate.py` — 零训练融合校准（leave-one-model-out），打破 FNR≤3pp。
- `b1b_prep_data.py` / `b1b_train_lora.py` / `b1b_eval.py` — 弱标签 LoRA verifier（训练 + 评测）。
- `b1b_prep_v2_realneg.py` — 负例工程重训数据（负结果对照）。
- `b1c_ensemble.py` — B1a 特征 + B1b 连续分集成（全上游一致正向主结果）。
- `report_gain.py` — 用 11-model 报告自身 T2 指标（mIoU/FG@Neg）衡量 verifier before→after 增益。

**成本层**
- `cheap_signal_scout.py` — 检测器一阶信号对 BOH/ROH 的判别力侦察。
- `boh_cheap_head.py` — BOH 便宜头（检测器信号逻辑回归）。
- `tiered_cascade.py` — 分层触发级联（便宜头判 BOH，物体在场升级 VLM）。

**数据工程（reference-box 复核）**
- `s4_bucket_render.py` — reference 候选分桶 + 渲染。
- `s4_vlm_prescreen.py` — VLM 第一遍预筛（三态 triage：auto_accept / conflict_multi / none_valid）。

## 复现说明
- 所有 verifier 泛化实验用 **leave-one-model-out**（在其他上游模型上校准、留出模型上评测）。
- 弱标签 = `label_exists ∧ IoU≥0.5`，零人工；SWITCH/RELOCALIZE 的 edge 真值需 reference-box held-out 集（人工二轮）。
- 数值口径以 500-dev（与 11-model 评测同集）为主，1996-heldout 为冻结后确认。
