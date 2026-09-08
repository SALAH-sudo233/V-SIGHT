# 数据 schema 与跨脚本主键契约

复现链路依赖脚本间的记录对齐。本文件固定字段与 join key，避免"公开代码与服务器实验不一致"。

## 主键（join key）
所有跨模型/跨阶段的记录对齐统一用 **`(model, sample_id)`**。
- `sample_id` 形如 `hallu_000249_COCO_train2014_000000310457`，负样本反事实带后缀 `__obj/__rel/__attr`。
- 图像文件名由 sample_id 还原：取 `COCO_train2014_<numeric>`（剥离反事实后缀）。

## 每阶段输出字段（committed 脚本已对齐）

### 上游评测记录（输入，`$VSIGHT_EVAL11_ROOT/<model>/records.jsonl`）
`task`(t1_discriminative_vqa|t2_vqa_grounding|t4_caption_grounding), `sample_id`, `model`,
`label_exists`, `hallucination_type`(object|co_occurrence|attribute|relation),
`pred_exists`(T1), `pred_found`/`pred_bbox_xyxy`/`gt_bbox_xyxy`/`iou`(T2)。

### cheap_signal_scout.py → `cheap_signals_full.jsonl`
每行：`model, sample_id, htype, label_exists, iou, det_max, det_best_iou, det_n, agree`。
**注意**：文件名必须是 `cheap_signals_full.jsonl`（consumers `boh_cheap_head.py` / `tiered_cascade.py` 读它）；
`sample_id` 必须存在（否则 tiered_cascade 与 b1b 分数 join 失败，overlap=0）。

### b1b_eval.py → `b1b_eval.jsonl`（7B）/ `b1b_3b_eval.jsonl`（3B）
每行：`model, sample_id, htype, label_exists, iou, p_correct`（连续置信度，读 P(yes) logit softmax）。

### s5_plugin_filter.py → `s5_<model>.jsonl`
每行：`model, sample_id, htype, label_exists, upstream_drew_box, iou, vlm_score, verdict, confidence`。

## 消费者的 join
- `b1c_ensemble.py`：join `s5_*`（B1a 特征）× `b1b_eval`（B1b p_correct）on `(model, sample_id)`；缺任一侧则跳过该框。
- `report_gain.py`：同上 + 回 join 上游 T2 记录取 `pred_bbox_xyxy`。
- `tiered_cascade.py`：join `cheap_signals_full` × `b1b_3b_eval` on `(model, sample_id)`。

## 弱标签
`y = label_exists AND iou>=0.5`（"存在且定位对"）。零人工。SWITCH/RELOCALIZE 的 edge 真值需 reference-box held-out（人工二轮）。

## 已知复现陷阱（P0 修复记录）
- cheap scout 早期缺 `sample_id` 且输出名为 `cheap_signals.jsonl` → 与 consumers 的 `cheap_signals_full.jsonl` 不一致，
  导致 join overlap=0。**已修**：scout 现输出 `sample_id` + 正确文件名。
- b1b_eval 的 3B 需 `--base $QWEN3VL_8B`... 实为 `Qwen2.5-VL-3B`；train/eval 的 `--model`/`--base` 必须指同一骨干。
- 评测泄漏（P1 待办）：leave-one-model-out 仍在同一批 500 图上，存在图像级泄漏；校准阈值碰了评测图像。
  真·held-out 需图像×模型双隔离 + 在 train/calib split 上定操作点（见 P1）。
