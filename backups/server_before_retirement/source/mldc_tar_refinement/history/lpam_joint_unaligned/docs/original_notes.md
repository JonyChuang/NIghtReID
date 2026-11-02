# 当前实验：MLDC + 自由查询 LPAM v2

本目录直接替换原 `a3pfn_mldc_lpam` 的代码。旧日志和旧检查点保留在原输出目录，新训练输出在其 `free_queries_v2` 子目录。没有新建实验代码副本。未自动启动训练。

## 模块与监督

- 从已完成的 MLDC 第 120 轮初始化完整全局分支（包含原全局 BN 统计量）。拒绝用旧 CLS-LPAM 检查点初始化。
- ViT 第 6、9、12 层的全部空间 patch token 分别经 LayerNorm 和 768→256 投影。四个可学习查询共享跨深度身份，采用四头余弦交叉注意力，温度 0.1。不做水平条带划分，不使用 FFT。
- 每个区域根据当前图像的各深度区域摘要预测三层门控权重，再做残差 MLP。四个局部区域分别使用独立 BN；拼接后的整体局部特征使用一个 ID 分类头。不要求每个小区域独立预测完整身份。
- 局部 Triplet 使用 BN 前的四区域等权归一化描述符。正样本为批内同 ID 的最难非自身样本；负样本从全局特征最相似的 8 个异 ID 样本中，选局部距离最近者。全局负样本候选选择停止梯度；局部 Triplet margin=0.3。
- 联合阶段损失：`全局 ID + 全局 Triplet + 0.5*(局部整体 ID + 局部 Triplet) + 0.01*注意力图重叠惩罚`。全局 Triplet 保留原始 MLDC 特征和原 soft-margin 形式。冻结阶段只优化后两项。重叠惩罚为不同区域注意力图余弦相似度平方的均值。
- 检索使用全局 BN 后 768D 与四个局部 BN 后 256D 特征。各局部区域分别 L2 归一化后等权拼接；最终向量为 `[sqrt(0.8)*global_unit, sqrt(0.2)*local_unit]`，共 **1792D**。相似度严格等于 `0.8*global_cos + 0.2*mean(part_cos)`，避免局部分支数量自动放大其影响。权重预先固定为 0.2，不按测试结果搜索。

这是参考自由查询局部建模思想重新实现的 LPAM 替代方案，并非 A³PFN 作者原始 LPAM，也不是对作者源码的等价复现。保留 A³PFN MLDC；不修改独立 `a3pfn_mldc` 基线代码。方法参考：PAT (Diverse Part Discovery), AAformer (Auto-Aligned Transformer), BPBreID (Body Part-Based ReID) 的自由区域/局部监督思路；不依赖姿态或分割模型。

## 训练与恢复

在项目根目录执行：

```bash
cd /root/TransReID-SSL-main/transreid_pytorch
/root/miniconda3/envs/QREID/bin/python mldc_tar_refinement/train.py --config_file mldc_tar_refinement/variants/tar_cls_residual_three_heads/config.yml
```

前 30 轮冻结全局参数、全局 BN 统计和骨干随机层，仅训练局部模块；局部 LR=0.002，预热 3 轮后余弦下降。第 31～45 轮仅解冻最后四个 Transformer 块、MLDC 和全局 BN/分类头，骨干 LR=0.00002，局部与全局头 LR=0.0002，余弦下降至初始值的 1%。优化器为 SGD，momentum=0.9，weight decay=0.0001；保留原 PK=16×4 和训练增强。局部 SGD 动量跨阶段保留。

30 轮是保守的预设超参数，并非文献规定或已经验证的最优值。可在启动命令末尾加 `SOLVER.LPAM_FREEZE_EPOCHS 0`，从第 1 轮联合训练；此时三组参数使用联合阶段学习率，前 3 轮一起预热。也可以设置较短的冻结阶段（需大于 warmup 轮数）。

断线后恢复：

```bash
/root/miniconda3/envs/QREID/bin/python mldc_tar_refinement/train.py --config_file mldc_tar_refinement/variants/tar_cls_residual_three_heads/config.yml --resume /root/log/transreid/night600/vit_base_ics_a3pfn_mldc_lpam/free_queries_v2/transformer_latest.pth
```

每轮原子保存 latest（模型、优化器、AMP scaler、阶段、轮数、Python/NumPy/PyTorch/CUDA 随机状态）。恢复要求相同配置及原输出目录。另保留第 30 轮阶段检查点和第 45 轮最终检查点。主结果预先固定为第 45 轮；best 为按 Night600 测试 mAP 选择的诊断结果。该实验在 MLDC-120 上额外训练 45 轮，不能宣称为训练预算匹配的独立消融。

## 观察模块是否有效

第 0 轮及每 5 轮评估。`evaluation.csv` 保存主指标；`diagnostics.csv` 保存同检查点的全局、局部、各区域、BN 前 readout/末层 CLS/均匀 CLS；`evaluation_diagnostics.jsonl` 保存区域深度门控均值、注意力重叠、去除单个区域后的检索指标。

`query_diagnostics/epoch_XXX/` 保存各查询 AP、首个正确样本排名和 Top10，以及 `repair`（全局错→融合对）、`harm`（全局对→融合错）数量、AP 变化。判断互补性应看纠正是否超过损害、AP 是否提升、去掉区域是否改变检索，不以热力图差异单独下结论。

`visualizations/epoch_XXX/` 始终包含相同的 6 对跨摄像头正样本，以及当轮部分 repair/harm/both_wrong 案例：

- 查询与正样本的四个真实交叉注意力图，使用同一概率色标；每张图的区域深度门控。
- 全局、局部、融合对“正样本相似度减全局困难负样本相似度”的有符号 gradient×patch embedding 图；红色支持正样本、蓝色支持负样本。它与注意力概率含义不同。
- 精确相似度贡献分解，及全局/局部/融合 Top10 图。该得分分解是精确的，有符号梯度图是局部诊断，不是因果解释。
- 原始注意力、门控、归因数组 `.npz` 与样本/得分 `.json`，便于量化比较。

`compare.py` 对比初始 MLDC-120、相同检查点的全局分支和融合分支。旧 3072D CLS-LPAM 检查点无法由新模型加载；仍可使用保留的旧日志分析历史结果。

目标 mAP/Rank-1=12.5/25 是待验证的实验目标，代码修改不能保证达到。
