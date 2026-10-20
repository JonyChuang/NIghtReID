# Night600：A³PFN MLDC 独立实验副本

只迁移 MLDC，不包含 FFVT、LPAM、FFT 或额外融合分支。复用项目的 Night600 数据加载、SSL 预训练、ViT-Base/16 + stem、优化器、学习率调度和检索评估。

来源：[作者公开实现](https://github.com/zhguoqing/A3PFN/blob/main/model/backbones/vit_pytorch.py)，核对于 2026-10-09。公开实现中的层权重为所有图像共享的可学习参数。

## 直接迁移的计算

收集第 1–12 层 Transformer block 输出的 CLS（未经过骨干最后的 LayerNorm）：

```python
layer_weights = nn.Parameter(torch.randn(12))
weights = softmax(layer_weights / 0.9, dim=0)
feature = sum(cls_by_layer * weights[:, None, None], dim=0)
```

随后使用一个 768 维 BNNeck 和线性身份分类器。ID 损失使用 BN 后特征，Triplet 损失使用融合原始特征；检索默认使用 BN 后的 768 维特征，再由现有评估器 L2 归一化。MLDC 参数使用骨干学习率；新 BN/分类器沿用当前实验的 5 倍学习率。初始化、温度、原始 CLS 和加权规则按作者公开代码保留，没有增加投影、最终 LayerNorm 或输入自适应门控。

## 两个配置

- `vit_base_ics_mldc.yml`：MLDC；输出 `../../log/transreid/night600/vit_base_ics_a3pfn_mldc`。
- `vit_base_ics_last_cls.yml`：末层原始 CLS 对照；输出 `../../log/transreid/night600/vit_base_ics_a3pfn_last_cls`。

对照也不经过最终 LayerNorm，其余预训练、分类器、损失、数据增强、随机种子和调度相同。两种模式都创建同样的 MLDC 参数以保持相同的初始化随机数序列，对照中的层权重冻结且不参与前向。这个对照用于隔离“多层加权读出”的作用；它与标准最终 LayerNorm 后 CLS 基线不同，也不能直接用此前 FFVT 双分支训练的全局结果替代。

默认设置：输入 256×128；batch 64；SGD；基础学习率 0.0004；20 轮 warmup；120 轮；每 10 轮评估。数据和预训练文件沿用项目根目录，无需复制。

## 运行

将本目录放在 TransReID 项目根目录下，命名为 `a3pfn_mldc`，从项目根目录运行：

```bash
python mldc_cls_readout/train.py
python mldc_cls_readout/train.py --config_file mldc_cls_readout/variants/last_raw_cls_control/config.yml
python mldc_cls_readout/test.py TEST.WEIGHT ../../log/transreid/night600/vit_base_ics_a3pfn_mldc/transformer_best.pth
```

也可以一次顺序完成 MLDC、独立末层 CLS 对照和结果比较：

```bash
python mldc_cls_readout/run_pair.py
```

已经分别跑完两个配置时，可执行 `python mldc_cls_readout/compare.py` 生成同轮次对照曲线、最终共同轮次的指标差值和各自最佳 mAP 对应的整组指标。该脚本会核对种子和其余训练配置一致。训练入口要求可用 CUDA GPU；没有 GPU 时不会退回 CPU 启动完整训练。

## 输出

- `train_log.txt`：身份损失、Triplet 损失、准确率和每轮层权重。
- `layer_weights.csv`：第 0 轮初始化及每轮训练后的 12 个实际读出权重；对照记录末层 one-hot。
- `evaluation.csv`：每 10 轮 mAP、Rank-1/5/10（数值为 0–1）。
- `diagnostics.csv`：同一个检查点中，原始训练读出、末层原始 CLS、12 层均匀平均三种读出的检索结果，均在 BN 前做相同 L2 归一化。
- `evaluation_diagnostics.jsonl`：上述诊断、层权重、各层原始 CLS 平均范数、加权范数占比和读出/末层 CLS 的平均余弦相似度。加权范数占比仅表示尺度，不是因果贡献。
- `run_metadata.json`：种子、读出模式和完整运行配置，用于独立实验的匹配检查。
- `transformer_best.pth`、`best_metrics.json`：按 mAP 保存最佳轮次。
- `transformer_120.pth`：最终权重。

权重包含读出模式和温度。测试时会核对它们与配置一致。本实验使用自己的检查点，不能加载已有 FFVT 双分支检查点。

比较模块效果时，使用相同 seed 的 MLDC 与末层 CLS 对照，分别报告最终结果和最佳 mAP 对应的整组指标；层权重日志用来检查是否收敛为单层主导。多随机种子重复可以在后续训练阶段完成。

## 可视化

默认在初始化第 0 轮、每 10 轮以及最终轮保存 6 个不同身份的固定跨相机 query/positive 对。两个独立配置使用同一个种子、同样的选样规则，因而可以逐样本和逐轮对应观察。样本通过种子选择，不根据结果挑选。

- `visualization_pairs.json`：固定样本及其索引，所有轮次复用。
- `visualizations/epoch_XXX/query_XXXXX_pid_XXX.png`：六面板显示 query、跨相机正样本、末层原始 CLS 热图、训练读出的原始热图、实际检索空间的热图，以及空间归因占比之差。上排的两张原始特征热图共用色标；实际检索热图另用自己的色标并标注 BN 前/后。
- 同名 `.npz`：保存原始 16×8 归因网格、差值、层权重、原始层范数、样本信息、相似度和相关系数。
- `heatmap_statistics.csv`：各固定样本的原始热图 Pearson 相关系数、最强 16 个位置重合率、描述子余弦和跨相机正样本相似度。
- `layer_weights.png`：当前层权重柱状图和从初始化到当前轮的权重热图；每个可视化轮次也保存对应版本。
- `retrieval_curves.png`：主检索曲线及同一检查点下三种 BN 前读出的诊断曲线。

归因使用相同正样本的余弦相似度对输入 patch embedding 求 `|gradient × activation|`。原始特征的末层/MLDC 比较都在 BN 前，避免把只对 MLDC 训练的 BN 直接套到末层 CLS 上；实际检索空间另行展示。差值面板为每张原始归因网格归一到总和 1 后相减，使用固定 ±0.03 色标，表示相对空间分配变化，不表示检索增益。

同模型内的末层/均匀读出未独立训练，只能作为机制诊断。判断模块是否有效应主要比较两次独立训练的主检索结果，并结合固定样本观察和层权重变化。热图变得不同本身不是有效性的判据。

可通过 `TEST.MLDC_VISUALIZE False` 关闭热图，或调整 `TEST.MLDC_VIS_SAMPLES` 和 `TEST.MLDC_VIS_PERIOD`。检查点测试入口会把诊断与热图保存到输出目录的 `checkpoint_analysis/epoch_XXX`，不会覆盖训练时的图片。

本地或服务器 CPU 上的合成输入验证只用于确认计算、梯度和图片输出流程，不是 Night600 训练结果。
