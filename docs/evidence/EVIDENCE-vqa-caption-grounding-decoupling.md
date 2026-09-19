# 证据资产：VQA / caption 与 grounding 的解耦（贡献一支撑）

> 问题：11-model 评测报告能否证明"grounding 与 VQA/caption 无关（解耦）"？
> **结论：能，且证据很强。** 关键在于正确理解三个任务是同一批 500×4 item 上的三条并行探针：
> - **T1 = 判别式 VQA**（yes/no：表达是否匹配可见目标）
> - **T2 = VQA + grounding**（存在性判断 + 输出 bbox）
> - **T4 = caption + grounding**（先描述图像，再定位）
>
> 在**同一批 item** 上比较 T1（纯 VQA 对错）与 T2（grounding 对错），得到直接的 VQA↔grounding 解耦证据。

数据：`benchmark/refcocog_eval_11models_500_repaired/run_500_semantic_strict/*/records.jsonl`，
每模型 7500 条（T1/T2/T4 各 2500 = 500 正 + 2000 负）。任务定义见 `eval_11models_refcocog_500_run.py` 第 5 行。
复算脚本：`vqa_grounding_decouple.py` → `roh_boh_gate_a_500dev/results/vqa_grounding_decouple.json`。

---

## 1. 核心结果：VQA 正确 ≠ grounding 正确（正样本）

对每个正样本，join T1 与 T2：`vqa_correct` = T1 判 yes；`ground_correct` = T2 pred_found 且 IoU≥0.5。
关键量：**P(ground_correct | vqa_correct)** 与 **φ 系数**（0=独立，1=完全一致）。

| 模型 | VQA 准确率 | grounding 准确率 | **P(ground对\|VQA对)** | **φ(VQA,ground)** |
|---|---:|---:|---:|---:|
| Qwen3-VL-8B | 0.78 | 0.522 | 0.554 | 0.120 |
| Seg-zero | 0.966 | 0.550 | 0.555 | 0.052 |
| VisionReasoner | 0.958 | 0.506 | 0.505 | **−0.007** |
| qwen2.5-vl-7b | 0.842 | 0.484 | 0.520 | 0.167 |
| Vision-R1 | 0.926 | 0.296 | 0.315 | 0.150 |
| TreeVGR | 0.99 | 0.262 | 0.263 | **0.014** |
| visual-rft | 0.876 | 0.052 | 0.055 | 0.033 |
| UniVG-R1 | 0.44 | 0.06 | 0.086 | 0.098 |
| **pooled (5500 正样本)** | — | — | **0.348** | **0.124** |

**读法**：
- 池化 **P(grounding 正确 | VQA 正确) = 0.35**。模型正确判断"是的，这个表达匹配图中目标"（VQA 答对）后，
  仍只有 **35%** 能把框定对。**VQA 答对根本不蕴含 grounding 定对。**
- 池化 **φ = 0.12**（跨模型均值 0.08）——VQA 对错与 grounding 对错的关联极弱。VisionReasoner 甚至 φ≈−0.007（几乎完全独立），
  TreeVGR φ=0.014（VQA 99% 对但只有 26% 定对）。
- 极端案例最有说服力：**TreeVGR VQA 0.99 / grounding 0.26**、**visual-rft VQA 0.88 / grounding 0.05**、
  **Vision-R1 VQA 0.93 / grounding 0.30**——语言判别能力饱和，定位能力却塌陷。

> 这就是阶段一命题的直接证据：**VLM 的 VQA（判别）能力与 grounding（定位）能力是两条可分离的能力轴，
> 前者对后者几乎没有预测力。** 因此"在语言/判别层做幻觉缓解"无法迁移到 bbox 定位层。

---

## 2. 佐证：VQA 拒绝 ≠ 不定位（负样本）

对负样本（BOH/ROH 反事实，目标不该存在），看 VQA 正确拒绝（T1 判 no）后，T2 是否仍画框：
**P(box_drawn | vqa_rejected)**。若高，说明"判别层说没有"和"定位层照样画"并存——同一模型内部两条链路矛盾。

- Seg-zero 0.95、UniVG-R1 0.997、TreeVGR 0.61、VisionReasoner 0.68：**VQA 已正确说"不存在"，grounding 仍几乎必画框。**
- 另一端 Vision-R1 0.11、visual-rft 0.12、qwen2.5-vl 0.16：判别与定位较一致。
- **跨模型分布极广（0.11–0.997）**：再次说明两条链路的耦合方向/强度**不是模型的稳定属性**，是解耦的。

---

## 3. 三任务交叉印证 ROH>BOH（贡献一另一半）

同一评测里 ROH 系统性比 BOH 难，三任务独立复现：
- T1 判别 ROH-BOH 幻觉率 gap：11–31pp（如 Qwen3-VL 24.2pp、Seg-zero 30.8pp）。
- T2 定位前景误定位（FG@Neg）：ROH≫BOH（多数模型 gap 15–33pp）。
- T4 caption+grounding ROH-BOH gap：27–34pp。

与 verifier 级 Gate A 的 gap 0.11–0.12 形成**模型级 + verifier 级双层证据链**。

---

## 4. 可写进论文的主张（严格版）

1. **VQA↔grounding 解耦（强）**：P(ground|vqa_correct)=0.35、φ=0.12（跨 11 模型，5500 正样本）。
   VQA 能力对 grounding 能力几乎无预测力；判别层的正确不能保证定位层的正确。
2. **判别–定位内部矛盾（强）**：负样本上 VQA 正确拒绝后仍画框的概率跨模型 0.11–0.997，
   证明两条链路在同一模型内可系统性冲突。
3. **caption↔grounding 解耦（中，T4）**：caption 干净时误定位率均值仍 42%（详见 §5）。
4. **机制错位推论**：由 1–3，语言/判别/caption 层的幻觉缓解手段（对比解码、注意力干预等）
   不能迁移到 bbox grounding —— 支撑"train-free 语言层方法不适用"和"必须在候选/定位层做 verifier"。

## 5. caption↔grounding（T4，作为第二轴，诚实边界）

- caption 干净时误定位率：11 模型均值 0.42，范围 0.16–0.88（caption 正确 ≠ 定位正确）。
- coupling_gap 均值仅 +0.15，9 正 2 负（TreeVGR −0.60、UniVG-R1 −0.88），方向跨模型不一致。
- 小样本警告：TreeVGR(14)、UniVG-R1(9)、visual-rft(30) 的 caption 幻觉样本少，极端值不稳；
  主证据用大样本（LENS 1112、Seg-zero 291、VisionReasoner 251）。

## 6. 不能过度宣称

- VQA↔grounding 用 φ 而非"完全独立"：φ≈0.12 是**极弱关联**，措辞用"几乎无预测力/解耦"，
  个别模型 φ≈0 可称"近似独立"，但整体不宣称严格统计独立。
- T4 的极端负 coupling 受小样本影响，加置信区间或以大样本模型为准。

---

*生成 2026-09-05。核心数值来自 `vqa_grounding_decouple.json`（T1↔T2 join）与 `cross_model_summary.json`（T4）。
修正说明：早期版本误把 T1 当 caption 轴；现按评测脚本定义 T1=判别VQA、T2=VQA+grounding、T4=caption+grounding 重做。*
