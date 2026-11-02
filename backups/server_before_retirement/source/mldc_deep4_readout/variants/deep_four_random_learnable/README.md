# 后四层随机起步（独立副本）

## 实验思路与模块

仅L9–12随机logits参与可学习融合。

配置：`mldc_deep4_readout/variants/deep_four_random_learnable/config.yml`。检索维度：768。

## 实验流程

从SSL初始化完整ViT，仅融合第9—12层原始CLS；随机或均匀初始化层权重，随后与骨干共同学习，保存层权重与检索指标。

1. 准备Night600数据与配置指定的初始化模型。
2. 使用本目录配置完成训练；记录损失、配置、随机种子及检查点。
3. 使用对应测试入口与同一检查点评估，按本配置记录的特征读出解释结果。

## 运行方式

在工程根目录执行：

```bash
/root/miniconda3/envs/QREID/bin/python mldc_deep4_readout/train.py --config_file mldc_deep4_readout/variants/deep_four_random_learnable/config.yml
```

训练设置：epochs=120，batch=64，optimizer=SGD，base_lr=0.0004，warmup=20。

## 输出与已有结果

语义输出路径：`/root/log/transreid/night600/mldc_deep4_readout/deep_four_random_learnable`。
已有结果原位置：`/root/log/transreid/night600/vit_base_ics_a3pfn_mldc/deep4_random_learnable_120`。已有结果存在时语义路径作为访问别名；不移动或改写旧检查点。

新的一次训练请另设有明确含义的 `OUTPUT_DIR`，如当前语义路径下的 `seed_1024_repeat_2`；已有结果目录保持为历史记录。

## 参考文献与本项目改动

- [A³PFN](../../../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [TransReID-SSL](../../../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

骨干来源为TransReID-SSL。其他文献对应迁移的模块；监督、层选择和ReID读出以本实验配置与代码为准。


旧随机配置仍可显式选择；当前不作为默认配置。

## 本配置的模块开关与关键设置

| 参数 | 值 |
|---|---|
| `MODEL.MLDC_READOUT` | `mldc` |
| `MODEL.MLDC_TEMPERATURE` | `0.9` |
| `MODEL.MLDC_LAYERS` | `[9, 10, 11, 12]` |
| `MODEL.MLDC_INIT` | `random` |
| `MODEL.MLDC_TRAIN_FEATURE` | `mldc` |
| `SOLVER.MLDC_HEAD_LR_FACTOR` | `5.0` |
| `SOLVER.MLDC_WEIGHT_LR_FACTOR` | `1.0` |
| `TEST.MLDC_RETRIEVAL` | `mldc` |
| `TEST.MLDC_VISUALIZE` | `True` |
| `TEST.MLDC_VIS_SAMPLES` | `6` |
| `TEST.MLDC_VIS_PERIOD` | `10` |
