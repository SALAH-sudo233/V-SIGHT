# V-SIGHT

**Visual Support and Instance Grounding with Hallucination-aware Triage**

V-SIGHT 研究一类被 positive-set IoU 掩盖的 grounding 失败：被查询的类别可能可见，但某个属性或关系
被绑定到了错误的同类实例上。给定图像、指代表达和一个上游框，V-SIGHT 先把 proposal 覆盖度与
target/reference/relation 绑定错误分离开，再评估一个选择性动作：

```text
    ACCEPT      保留有支持的上游框
    ABSTAIN     无支持或无安全替代时拒答，不返回框（代码仍以 REJECT 序列化）
    RELOCALIZE  对可恢复的错误实例，用验证过的 Top-K proposal 改定位
    DEFER       verifier 不确定 → 转人工复核优先级
```

## 核心命题（2026-09 重构）

**VLM 的 grounding 能力与 hallucination mitigation 能力不是同一个目标。** 对容易的目标级 grounding，
模型已接近饱和；对关系场景中的 target-reference binding，模型仍普遍出现关系型幻觉。区分两类幻觉：

- **BOH（basic-object hallucination）**：目标对象不存在/类别不成立。瓶颈在存在性与候选覆盖。
- **ROH（relation-object hallucination）**：目标与参考物可能都在，但属性/关系/角色绑定错误。

**权威叙事与最新结论见 [`docs/updates_2026_09/`](docs/updates_2026_09/)**：
- [`DIRECTION_UPDATE.md`](docs/updates_2026_09/DIRECTION_UPDATE.md) — 当前定位、三阶段主线、四贡献、动作映射。
- [`FINDINGS.md`](docs/updates_2026_09/FINDINGS.md) — 实验结论汇总。
- 可复现脚本见 [`experiments/roh_boh/`](experiments/roh_boh/)。

## 当前结论（要点）

- **ROH>BOH 与 VQA↔grounding 解耦有因果机制**：P(定位对|判别对)=0.35、φ=0.12；成因是上游 RL grounding 模型
  "正样本-only 训练 + 坐标 token 无独立梯度"两个范式选择的必然产物（11 篇上游模型论文原文核实，
  含 Visual-RFT 剂量-反应反例）。
- **即插即用 verifier 在真实上游模型上降幻觉**：以 11-model 报告自身 T2 指标衡量，正样本 mIoU 损失 ≤0.005
  约束下，高幻觉上游 FG@Neg 下降 22–30pp。
- **分层成本设计**：BOH 便宜检测器信号可判、ROH 内在需 VLM；级联在 ~75% VLM 调用率下 catch 超两个纯端点；
  3B 骨干保住 7B ~90% ROH 判别力。

## 决策历史（negatives，保留但非当前方法）

以下早期工作已被上面的重构取代，作为**可复现的负结果 / 决策历史**保留在本仓库：

- **E2 学习式联合验证器**：CLIP 与 Qwen-LoRA candidate verifier 均未通过 oracle-gap 与 repaired-500 迁移门
  → 见 `docs/E2_RESULTS.md`。**注**：2026-09 的弱标签 LoRA verifier（`experiments/roh_boh/b1b_*`）是不同设定，
  已在真实上游模型输出上取得正向增益。
- **TRACE（decoder-trajectory）**：trajectory-only AUROC≈0.5009（近随机），保留为诊断性负结果与匹配 ablation。
- **CCV / CABLE**：作为实现资产（typed claim parser、composite 单次检测器、固定 proposal 池、原子/边 ledger、
  校准与安全指标）保留；其 train-free 动作策略未安全解决关系绑定。

## Layout

```text
configs/                    frozen experiment specifications
data/                       manifests and append-only audit artifacts
docs/                       method, data, annotation, and evaluation protocols
docs/updates_2026_09/       CURRENT authoritative narrative + findings
experiments/roh_boh/        2026-09 reproducible pipeline scripts (env-configurable paths)
legacy/candidate_pool_v1/   validated pre-V-SIGHT candidate snapshot
scripts/                    isolation and audit-manifest checks
src/vsight/                 joint decision and data-integrity primitives
tests/                      dependency-free unit tests
```

## Verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
# provide the held-out path via env (do not hardcode server paths)
python3 scripts/check_data_isolation.py \
  --dev legacy/candidate_pool_v1/data/refcocog_500_dev.semantic_strict.json \
  --heldout "$VSIGHT_HELDOUT_1996"
python3 legacy/candidate_pool_v1/code/verify_effective_snapshot.py
```

Read `docs/EXPERIMENT_PLAN.md` before generating training data or opening the held-out split.
`experiments/roh_boh/README.md` documents the env vars (`VSIGHT_IMAGE_ROOT`, `VSIGHT_EVAL11_ROOT`,
`QWEN25VL_7B`, `GROUNDING_DINO`, …) for the 2026-09 pipeline.

## E1 data build

The canonical sources are RefCOCO UNC, RefCOCO+ UNC, and RefCOCOg UMD train splits. The builder removes
all 2,496 protected repaired-500/repaired-1996 image IDs before assigning train and calibration by COCO image ID.

```bash
python3 scripts/build_e1_source_manifest.py
python3 scripts/build_e1_candidate_supervision.py
```

The frozen source summary is `data/e1/source/e1_source.summary.json`; human-readable counts are in
`data/e1/source/E1_SOURCE_REPORT.md`.

## Human audit

Loopback-only review interfaces:

```bash
python3 scripts/review_zero_iou.py       # 127-group IoU=0 audit
python3 scripts/review_e3_binding.py     # E3 target/reference binding review
```

Reviews are appended to per-interface JSONL logs; T2/T4 labels and reviewer IDs are kept separate.
