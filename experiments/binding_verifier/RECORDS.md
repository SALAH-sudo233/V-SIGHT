# Binding Verifier / Two-Stage — Ongoing Experiment Records

> 状态：**进行中的探索路线（ongoing）**，不是已完成的有效贡献。
> 本目录只保存脚本、reward plugin、训练启动脚本、数据审计文档和小体量结果摘要；
> 大体量训练数据（`*_train2000.jsonl`、`two_stage_s*.jsonl`）、checkpoint（`saved_ckpts/`）、
> 逐样本评测 dump 和训练日志留在服务器 vlm1 `~/SVD/grpo_verifier/`，不进仓库。

## 目标

在 GRPO decision verifier（KEEP/REJECT）之上，尝试让 3B 显式输出 **binding**（MATCH/MISMATCH/NA）
判断，并探索若干训练配方：
- `binding_grpo_v1`：binding + decision 联合。
- `symmetric_v1`：对正/负查询对称加权的 reward。
- `group_v1`：按 group 组织 reward。
- `decision_only_pilot`：仅 decision 的对照。
- `two_stage_s1/s2`：两阶段（先 decision，后 binding/对齐）。

对应脚本：`run_binding_grpo_*.sh` / `run_two_stage_s*.sh` / `run_decision_only_pilot.sh`；
reward plugin：`vsight_reward_plugin_{binding,symmetric,group,two_stage,decision_only,structured}.py`；
数据构造：`binding_data.py` / `prepare_binding_remote.py` / `prepare_group_swift.py` / `prepare_two_stage.py`。

## 训练数据的诚实边界（关键，见 `BINDING_VERIFIER_DATA_AUDIT_v1.md`）

审计对象 `refcocog_train2000.corrected_v2.json`（8,000 pairs / 2,000 images）。审计结论：

1. **标签来源是 programmatic 启发式**（`challenge_confidence=low`），**不是独立人工语义金标**；
   同一模型的独立复核不等于跨模型共识。
2. **relation 类存在结构性缺陷**：1,428/2,000 是"给正例加一个指向新 reference 的关系子句"，
   但数据**无 reference_box**；1,995/2,000 负例相对正例引入了新名词 token 却没有任何 reference 位置证据。
   → 这类样本**无法有效训练 relation binding**。真正方向/角色变更仅 162 条（启发式上限，未人工确认）。
3. co_occurrence 里 27 条为同一对象内部反义词冲突（standing/sitting 类），属"文本捷径"风险，应作 ambiguous。
4. 因此 binding 训练目前**只能训 decision，不能宣称训成了 relation binding**。

## 评测状态（不作为有效结论）

- `results/sanity_eval.json`（340 条，来自 pilot binding_pilot_v1）：格式与 KEEP/REJECT sanity 检查，
  含 relation 上 gold=MISMATCH 而 pred=MATCH 的失败样例，用于确认 pipeline 可跑，不是性能结论。
- `results/shard_*.jsonl.summary.json`：upstream CCV 评测分片的 proposal-recall/mIoU 摘要（train-free CCV 侧）。
- **已知问题（必须保留）**：`ccv_eval/verifier_{zero,sft,ck400,ck800}.json` 四个 checkpoint 的整体
  decision accuracy **完全一致**（ALL 0.5644 / BOH 0.7642），无法体现 checkpoint 差异。
  原因未定位（疑似 CCV 评测走 train-free 提议路径、未真正加载对应 LoRA，或评测口径问题）。
  **在定位并修复前，这批 CCV verifier 数字不作为有效结果**；对应 2.1MB 逐样本 dump 不入库，留服务器。

## 复现位置

- 服务器：vlm1 `~/SVD/grpo_verifier/`，conda env `grpo_ayb`（`export LD_LIBRARY_PATH=$HOME/.miniconda3/envs/grpo_ayb/lib:$LD_LIBRARY_PATH`）。
- checkpoint 保护：`protect_binding_ckpts.py`；saved_ckpts 含 `binding_grpo_v1` / `binding_sft_ck250`。
- 权威 decision verifier 结果见 `../grpo_verifier/TRAINING.md`（v2 ck600 为当前最优开发集观察点）。
