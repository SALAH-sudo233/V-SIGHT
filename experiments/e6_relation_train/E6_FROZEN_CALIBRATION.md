# E6 冻结校准复评

协议：只在142张 calibration 图像上选阈值，在151张 development 图像上评估；使用已有原始 P(yes)，未重新训练。

| 上游 | ROH AUROC 旧→E6 | ROH catch 旧→E6 | realFNR 旧→E6 |
|---|---|---|---|
| LENS | 0.720→0.793 | 9.5%→24.1% | 0.0%→2.7% |
| Orsta-7B | 0.651→0.648 | 13.3%→8.7% | 4.3%→0.0% |
| Qwen3-VL-8B | 0.575→0.500 | 30.4%→21.6% | 20.0%→20.0% |
| Seg-zero | 0.675→0.767 | 7.0%→18.9% | 1.1%→0.0% |

## 判断
- LENS/Seg-zero 改善在独立校准协议下仍存在；relation 与 attribute 分开后两者均改善。尚无置信区间，不宣称统计显著。
- Orsta 的 ROH AUROC 基本持平；不能沿用全500样本“三个上游都提升”的结论。
- Qwen3-VL 校准正确正例仅3条、development仅5条；新旧 realFNR 均20%，不满足3%预算。
- 新旧分数各7007条，key完全一致，无重复；htype/label_exists/IoU一致。分数文件不足以核验原始图像绘框、query与prompt一致性。
- prepared train split 374张图像，与500-dev交集为0，与E2的294张图像交集为204。实际平衡采样训练子集仍需核对；全E2结果不能称作独立泛化证据。
- 本次没有完成十一模型扩展、权重重新加载或report-native收益评测。

产物：e6_crossfit_eval.py；results/e6_frozen_calibration.json。
