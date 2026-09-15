# E2 补跑核验与缺失原因更正

- 已在 vlm1 空闲 GPU 5 重新运行 E6 flat：70/70 条成功，E2a 分数与历史结果完全一致。
- 此前所谓30条漏跑，实际均无 agent target：25 REJECT、5 ABSTAIN；不是推理失败。不能用人工框或另选检测框填充 E2a。
- 排除整个 prepared train 图像集合后，共100条；其中70条有框：32正确、37 wrong-instance、1 target-absent。AUROC比较只用32+37，属于有框条件下的诊断，不代表全系统表现。

| verifier | AUROC | wrong-instance flag<0.5 |
|---|---:|---:|
| E6 | 0.7010 | 40.5% |
| old | 0.7766 | 37.8% |
| zero | 0.6668 | 16.2% |

E6−旧LoRA 的配对 image-group bootstrap 95% CI：[-0.1723, 0.0143]（2000次）。这是探索性诊断，不能单凭点估计声称提升显著或达到能力上限。

relation_status 描述人工最终关系，不等于原框正确/错误或多合法目标。禁止把 unverified 当关系矛盾。E2c/d 更换为人工正确target后也不能继续套用原wrong-instance负标签。
