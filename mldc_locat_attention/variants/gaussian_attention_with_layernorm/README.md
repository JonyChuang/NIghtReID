# GAug＋LN

## 实验思路与模块

开启GAug和读出LayerNorm，关闭PRR。

配置：`mldc_locat_attention/variants/gaussian_attention_with_layernorm/config.yml`。检索维度：768。

## 实验流程

按配置控制GAug、读出LayerNorm和PRR；融合完整多层token后读取CLS，使用相同ID/Triplet和评估设置比较各组件。

1. 准备Night600数据与配置指定的初始化模型。
2. 使用本目录配置完成训练；记录损失、配置、随机种子及检查点。
3. 使用对应测试入口与同一检查点评估，按本配置记录的特征读出解释结果。

## 运行方式

在工程根目录执行：

```bash
/root/miniconda3/envs/QREID/bin/python mldc_locat_attention/train.py --config_file mldc_locat_attention/variants/gaussian_attention_with_layernorm/config.yml
```

训练设置：epochs=120，batch=64，optimizer=SGD，base_lr=0.0004，warmup=20。

## 输出与已有结果

语义输出路径：`/root/log/transreid/night600/mldc_locat_attention/gaussian_attention_with_layernorm`。
已有结果原位置：`/root/log/transreid/night600/vit_base_ics_mldc_locat_gaug_norm`。已有结果存在时语义路径作为访问别名；不移动或改写旧检查点。

新的一次训练请另设有明确含义的 `OUTPUT_DIR`，如当前语义路径下的 `seed_1024_repeat_2`；已有结果目录保持为历史记录。

## 参考文献与本项目改动

- [LocAtViT](../../../REFERENCES.md#locat)：https://github.com/sinahmr/LocAtViT
- [A³PFN](../../../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [TransReID-SSL](../../../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

骨干来源为TransReID-SSL。其他文献对应迁移的模块；监督、层选择和ReID读出以本实验配置与代码为准。




## 本配置的模块开关与关键设置

| 参数 | 值 |
|---|---|
| `MODEL.MLDC_READOUT` | `mldc` |
| `MODEL.MLDC_TEMPERATURE` | `0.9` |
| `MODEL.LOCAT_GAUG` | `True` |
| `MODEL.LOCAT_PRR` | `False` |
| `MODEL.LOCAT_READOUT_NORM` | `True` |
| `SOLVER.MLDC_HEAD_LR_FACTOR` | `5.0` |
| `TEST.MLDC_VISUALIZE` | `True` |
| `TEST.MLDC_VIS_PERIOD` | `10` |
| `TEST.MLDC_VIS_SAMPLES` | `6` |
