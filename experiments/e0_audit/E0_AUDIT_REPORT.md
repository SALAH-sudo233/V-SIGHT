# E0 — 可靠性审计与切分冻结（报告）

**日期：2026-09-08　执行：本地 + vlm1（`~/SVD/V-SIGHT_AGENTIC_FLYWHEEL`）**
**对应路线图：0.1 P0、2.2、4.1(E0)、5.1。本报告为本轮实际执行结果，非历史报告转写。**

---

## 0. 结论速览

- **切分边界已冻结且全部隔离断言通过**：pilot(300 人工金集) / dev500(拆 fusion-fit 207 / calib 142 / dev 151) / bench1996(封存) 三集在 coco 图像 id 上两两不相交，dev500 三子集为精确不重叠覆盖。
- **§2.2 的 fit/calib 耦合已用解耦协议复核**：权重拟合、阈值选择、评估分别路由到三个图像不相交的角色。
- **判别信号是真的**：ROH AUROC 在解耦前后基本不变，收益不是拟合泄漏的产物。
- **诚实成本已量化**：解耦后 catch@FNR≤3pp 普遍下降（阈值不再偷看测试分布）；Qwen3-VL 在小校准集下阈值迁移不稳（realFNR 冲到 0.2），如实记录，不用耦合假象掩盖。

---

## 1. 数据角色与隔离（`manifests/binding_v2_splits.json`）

隔离单位 = coco 图像 id；一张图的所有表达/模型输出/反事实后缀/近重复归属唯一角色。

| 角色 | 状态 | n(图像) | 用途 | 禁止 |
|---|---|---:|---|---|
| pilot_human_dev | available | 300 | E2 人工证据 oracle；诊断；协议调优 | 不升训练/最终测试；不在此定最终阈值 |
| dev500_fusion_fit | available | 207 | 拟合融合/小 router 权重 | 不选阈值；不回流训练 |
| dev500_calibration | available | 142 | **仅**选决策阈值与动作停止参数（此后冻结） | 不拟合权重；不回流训练 |
| dev500_development | available | 151 | 方案与 checkpoint 选择 | 非封存测试 |
| final_test_bench1996 | sealed | 1996 | 冻结后一次性最终确认 | 不提前读标签/选阈值/挖例 |
| train_seed | planned | 0 | R0 关系监督（新标注） | 须与以上全不相交；不用 protected 图 |
| train_feedback_R1 / R2 | planned | 0 | 飞轮更新 | 不从评测错误挖训练数据 |

**已验证不相交对**（本轮实跑）：
- pilot ∩ bench1996 = 0
- pilot ∩ dev500(all) = 0
- dev500 ∩ bench1996 = 0
- fusion_fit ∩ calibration ∩ development = 两两 0，且精确覆盖 500

dev500 子集划分：`sha256("vsight_binding_v2_dev500:"+id) % 10`，bucket<4→fit / 4–6→calib / 7–9→dev。无种子、顺序无关、可复现。manifest sha256 记于文件内。

> 关键收获：goldenapple 的 pilot_300 人工金集与现有 dev500、封存 bench1996 **完全不重叠**，是独立于两者的第三批图——直接可作 E2 人工 dev，无需重标即可启动首轮诊断。

---

## 2. fit/calib 解耦复核（`b1_crossfit_eval.py` → `results/b1_crossfit_eval.json`）

**OLD（耦合，旧 B1a/B1c 口径）**：权重在全部 train-model 行上拟合；阈值 + 评估在 held-out model 的全部 500 图行上（阈值选择与评估共享图像）。
**NEW（解耦）**：权重在 train-model 的 `fusion_fit` 图行；阈值在 held-out model 的 `calibration` 图行冻结；AUROC/catch 在 held-out model 的 `development` 图行。仍为 leave-one-model-out（跨上游即插即用），且三操作图像不相交。

4 个上游（S5-plugin 集：LENS / Orsta-7B / Qwen3-VL-8B / Seg-zero），ROH 结果：

### b1c_ensemble（方法主结果）

| model | AUROC_ROH OLD→NEW | catchROH@3pp OLD→NEW | realFNR OLD→NEW |
|---|---|---|---|
| LENS | 0.779 → **0.797** | 0.327 → 0.262 | 0.028 → 0.014 |
| Orsta-7B | 0.643 → **0.703** | 0.104 → 0.219 | 0.027 → 0.000 |
| Qwen3-VL-8B | 0.817 → 0.749 | 0.681 → 0.618 | 0.000 → **0.200** |
| Seg-zero | 0.765 → **0.783** | 0.299 → 0.173 | 0.029 → 0.000 |

（b1a_only / b1b_only 全表见 JSON。）

### 解读

1. **AUROC 稳健** → 判别信号真实。ROH AUROC 解耦前后基本持平甚至升高（Orsta +0.06），说明 B1c 的分离能力不是「阈值偷看测试」造成的假象。这是可以写进论文的独立校准结论，比旧 P1「独立但耦合」更强。
2. **catch@3pp 下降是诚实成本**。阈值现在只见过独立 calib 图，评估在独立 dev 图，catchROH 普遍下降（LENS 0.327→0.262，Seg-zero 0.299→0.173）。旧口径高是因为阈值和评估同分布。**论文应报 NEW 数字。**
3. **Qwen3-VL 暴露阈值脆弱**。n_cal 仅 254（该模型有效行少），冻结阈值迁到 dev 时 realFNR=0.2——小校准集下阈值不稳。如实记录；缓解见下。

### 下一步（不在 E0 范围，供 E1/校准阶段）

- 小校准集问题用**交叉拟合/交叉校准**（K 折 calib，轮换选阈+评估）替代单次冻结，减少 Qwen3-VL 式方差；或按路线图 §9.2 报最差组风险而非只报宏平均。
- 校准阈值按训练模型宏平均 + 最差组风险双报；未见模型分布可能移，不把校准集风险当对所有未见模型的保证。

---

## 3. 产物清单

```
experiments_v2/e0_audit/
  build_splits_manifest.py        # 生成 manifest + 隔离断言（本地跑，全绿）
  manifests/binding_v2_splits.json
  b1_crossfit_eval.py             # 解耦复核（vlm1 跑）
  results/b1_crossfit_eval.json   # OLD vs NEW 全表
  pilot_image_ids.txt dev500_ids.txt bench1996_ids.txt
  golden_review/                  # goldenapple 人工金集 + 复核前端存档（上级目录）
```

## 4. 待办（E0 收尾 + 交接 E2）

- [ ] 行主键 `(model,task,sample_id)` 全库零覆盖/零重复断言脚本（`audit_cost_and_leakage.py` 雏形）。
- [ ] `upstream_training_audit.csv`：Seg-Zero/Visual-RFT 逐 checkpoint 核训练配方（不按输出反推）。
- [ ] 成本基线：统一硬件/精度/分辨率，分记 detector/VLM/控制器前向、视觉 token、p50/p95。
- [ ] **E2 可启动**：pilot_300 的 253 参照物 + 231 relation 人工真值 → 机审 oracle 版先跑（标注为机审），人工版直接用金集。
