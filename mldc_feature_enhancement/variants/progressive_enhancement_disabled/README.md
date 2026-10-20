# PFE 关闭模块对照

## 实验思路与模块

关闭PFE模块，保持匹配的固定等权全局训练。

配置：`mldc_feature_enhancement/variants/progressive_enhancement_disabled/config.yml`。检索维度：768。

## 实验流程

固定后四层CLS均值作为全局分支；逐层增强patch并形成补偿残差；按配置关闭残差、使用原增强或控制残差范数，再以相同训练和检索设置比较。

1. 准备Night600数据与配置指定的初始化模型。
2. 使用本目录配置完成训练；记录损失、配置、随机种子及检查点。
3. 使用对应测试入口与同一检查点评估，按本配置记录的特征读出解释结果。

## 运行方式

在工程根目录执行：

```bash
/root/miniconda3/envs/QREID/bin/python mldc_feature_enhancement/train_pfe.py --config_file mldc_feature_enhancement/variants/progressive_enhancement_disabled/config.yml
```

训练设置：epochs=120，batch=64，optimizer=SGD，base_lr=0.0004，warmup=20。

## 输出与已有结果

语义输出路径：`/root/log/transreid/night600/mldc_feature_enhancement/progressive_enhancement_disabled`。
已有结果原位置：`/root/log/transreid/night600/vit_base_ics_featenhancer_mldc_20261017/pfe_fixed_equal_baseline_120`。已有结果存在时语义路径作为访问别名；不移动或改写旧检查点。

新的一次训练请另设有明确含义的 `OUTPUT_DIR`，如当前语义路径下的 `seed_1024_repeat_2`；已有结果目录保持为历史记录。

## 参考文献与本项目改动

- [A³PFN](../../../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [PFE-Net](../../../REFERENCES.md#pfe)：https://doi.org/10.1007/s44443-026-00927-9
- [TransReID-SSL](../../../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

骨干来源为TransReID-SSL。其他文献对应迁移的模块；监督、层选择和ReID读出以本实验配置与代码为准。




## 本配置的模块开关与关键设置

| 参数 | 值 |
|---|---|
| `MODEL.GLOBAL_LAYERS` | `[9, 10, 11, 12]` |
| `MODEL.GLOBAL_FUSION` | `fixed_equal` |
| `MODEL.PFE_ENABLED` | `False` |
| `MODEL.PFE_LAYERS` | `[2, 4, 6, 8]` |
| `MODEL.PFE_DIM` | `192` |
| `MODEL.PFE_KERNEL` | `3` |
| `MODEL.PFE_EXPANSION` | `0.5` |
| `MODEL.PFE_GAIN` | `0.1` |
| `SOLVER.MLDC_HEAD_LR_FACTOR` | `5.0` |
| `TEST.PFE_VISUALIZE` | `False` |
| `TEST.PFE_VIS_INDICES` | `[810, 1544, 11, 2059, 160, 24]` |
