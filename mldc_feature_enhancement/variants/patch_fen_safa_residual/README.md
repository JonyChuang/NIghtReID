# 方案 B：Patch FEN/SAFA

## 实验思路与模块

L4/6/8 patch增强，固定deep4全局引导中心化局部读取，使用g+0.1r。

配置：`mldc_feature_enhancement/variants/patch_fen_safa_residual/config.yml`。检索维度：768。

## 实验流程

固定后四层CLS均值形成全局分支；对patch执行FeatEnHancer启发的层次增强与SAFA读出，形成补偿残差；使用关闭补偿的配置作为对照。

1. 准备Night600数据与配置指定的初始化模型。
2. 使用本目录配置完成训练；记录损失、配置、随机种子及检查点。
3. 使用对应测试入口与同一检查点评估，按本配置记录的特征读出解释结果。

## 运行方式

在工程根目录执行：

```bash
/root/miniconda3/envs/QREID/bin/python mldc_feature_enhancement/train.py --config_file mldc_feature_enhancement/variants/patch_fen_safa_residual/config.yml
```

训练设置：epochs=120，batch=64，optimizer=SGD，base_lr=0.0004，warmup=20。

## 输出与已有结果

语义输出路径：`/root/log/transreid/night600/mldc_feature_enhancement/patch_fen_safa_residual`。
已有结果原位置：`/root/log/transreid/night600/vit_base_ics_featenhancer_mldc_20261017/scheme_b_patch468_fixed_equal_120`。已有结果存在时语义路径作为访问别名；不移动或改写旧检查点。

新的一次训练请另设有明确含义的 `OUTPUT_DIR`，如当前语义路径下的 `seed_1024_repeat_2`；已有结果目录保持为历史记录。

## 参考文献与本项目改动

- [A³PFN](../../../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [FeatEnHancer](../../../REFERENCES.md#featenhancer)：https://github.com/khurramHashmi/FeatEnHancer
- [TransReID-SSL](../../../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

骨干来源为TransReID-SSL。其他文献对应迁移的模块；监督、层选择和ReID读出以本实验配置与代码为准。


与旧方案A的RGB输入增强不同；当前config.yml默认B。

## 本配置的模块开关与关键设置

| 参数 | 值 |
|---|---|
| `MODEL.GLOBAL_LAYERS` | `[9, 10, 11, 12]` |
| `MODEL.GLOBAL_FUSION` | `fixed_equal` |
| `MODEL.PATCH_LAYERS` | `[4, 6, 8]` |
| `MODEL.PATCH_DIM` | `192` |
| `MODEL.PATCH_HIDDEN` | `64` |
| `MODEL.PATCH_GAIN` | `0.1` |
| `SOLVER.MLDC_HEAD_LR_FACTOR` | `5.0` |
| `TEST.PATCH_VISUALIZE` | `True` |
| `TEST.PATCH_VIS_INDICES` | `[810, 1544, 11, 2059, 160, 24]` |
