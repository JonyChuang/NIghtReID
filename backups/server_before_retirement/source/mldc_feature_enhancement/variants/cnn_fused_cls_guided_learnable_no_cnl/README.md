# 可学习 MLDC 融合 CLS 通过交叉注意力读取 CNN，无 CNL

在已完成的 `cnn_fused_cls_guided_no_cnl_fixed_equal_120` 方案上，将 CLS9–12 的固定均值替换为原 MLDC 的可学习 Softmax 加权。复用 `mldc_deep4_readout.model.mldc.MLDC`，四个 logit 初始化为零，温度为 0.9，初始层权重均为 0.25。四个 logit 与骨干、CNN 一起优化，不是针对每张图像预测权重。

融合后的原始 CLS 同时用于全局描述子和 CNN Token Reader 的 Query。浅层损失可以经 Query 回传到四个融合权重。CNN/MRFP、全局及浅层的 ID＋Triplet 监督延续此前方案。读取模块将复制的最后一个 Transformer Block 改为交叉注意力：融合 CLS 独立提供 Q，CNN 多尺度 tokens 独立提供 K/V，CLS 不进入 K/V。沿用复制块的 SSL Q/K/V、输出投影、LayerNorm 和 FFN 参数，保留 Query 残差、FFN 残差及最终 LayerNorm；CNN tokens 不经过该块的自注意力或 FFN 更新。CNL 权重为 0，不建立记忆库。

最终仍然对全局和浅层分别 BN、L2 归一化，再按 `[g;s]/sqrt(2)` 拼接，输出 1536 维。可学习的是四层 CLS 融合权重；两分支拼接仍为等能量。

## 参数

- `GLOBAL_FUSION: learnable_softmax`，`MLDC_INIT: uniform`，`MLDC_TEMPERATURE: 0.9`。
- `MLDC_WEIGHT_LR_FACTOR: 1.0`：融合 logit 的基础学习率和权重衰减与原 MLDC 相同；沿用此前 CNN 实验的分组余弦调度。
- `CNN_SD_READER: cross_attention`，12 个头，维度 768。
- `CNN_SD_QUERY: fused_cls`，`SHALLOW_LOSS_WEIGHT: 0.001`，`CNL_WEIGHT: 0.0`。
- CNN 学习率倍数 10，复制的 Reader Block/LN 倍数 1，分类头倍数 5；SGD、120 轮、基础学习率 0.0004、种子 1234、批量 64 均延续原 CNN 配置。
- 配置从已完成实验记录恢复。`CPPI_ENABLED: false` 明确关闭注入；CNNShallowDeepReID 原本也不执行 CPPI 注入。

## 运行

在工程根目录、准备好数据和 SSL 权重后，手动执行：

```bash
python mldc_feature_enhancement/train_cnn.py --config_file mldc_feature_enhancement/variants/cnn_fused_cls_guided_learnable_no_cnl/config.yml
```

此前自注意力版本同步到原实验服务器，代码目录为 `/root/TransReID-SSL-main/transreid_pytorch`，解释器为 `/root/miniconda3/envs/QREID/bin/python`。默认输出为 `/root/log/transreid/night600/mldc_feature_enhancement/cnn_fused_cls_guided_learnable_cross_attention_no_cnl`。跨平台使用时调整 `OUTPUT_DIR`。

训练日志和 `training.csv` 记录四个层权重。评估同时保存全局、浅层和拼接指标，以拼接 mAP 选最优检查点。检查点保存四个 logit、配置和独立架构标识，可由 `test_cppi.py --weight ...` 严格恢复。原固定等权 CNN 和纯全局 CNL 的架构标识继续保留。

2026-10-21 的交叉注意力修改已在本地完成，上传时保留服务器浅层损失权重 0.001；未启动训练。随机 64×32 输入检查通过前向、浅层损失到 MLDC/CNN/注意力参数的梯度、显式注意力参考计算及严格权重恢复。旧配置缺少 CNN_SD_READER 时仍恢复为 self_attention。新架构标识为 cnn_fused_cls_guided_learnable_cross_attention_no_cnl。
