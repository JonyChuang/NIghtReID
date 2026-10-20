# Patch直接监督＋GRM v2

## 实验思路与模块

冻结已训练deep4全局模型；前5轮局部预热，之后带局部直接监督的门控残差融合。

配置：`mldc_patch_grm/variants/frozen_global_patch_grm/config.yml`。检索维度：768。

## 实验流程

加载已训练deep4初始化，固定全局骨干、融合和原BN；前5轮直接训练局部读出，随后启用GRM残差及最终特征监督，评估冻结全局、局部和融合读出。

1. 准备Night600数据与配置指定的初始化模型。
2. 使用本目录配置完成训练；记录损失、配置、随机种子及检查点。
3. 使用对应测试入口与同一检查点评估，按本配置记录的特征读出解释结果。

## 运行方式

在工程根目录执行：

```bash
/root/miniconda3/envs/QREID/bin/python mldc_patch_grm/train.py --config_file mldc_patch_grm/variants/frozen_global_patch_grm/config.yml
```

训练设置：epochs=120，batch=64，optimizer=SGD，base_lr=0.0004，warmup=20。

## 输出与已有结果

语义输出路径：`/root/log/transreid/night600/mldc_patch_grm/frozen_global_patch_grm`。
已有结果原位置：`/root/log/transreid/night600/vit_base_ics_mldc_patch_compensation_20261016/stage2_patch_direct_grm`。已有结果存在时语义路径作为访问别名；不移动或改写旧检查点。

新的一次训练请另设有明确含义的 `OUTPUT_DIR`，如当前语义路径下的 `seed_1024_repeat_2`；已有结果目录保持为历史记录。 初始化检查点已随包迁移到 `mldc_patch_grm/initialization/`；实验参数和权重内容保持原值。

## 参考文献与本项目改动

- [A³PFN](../../../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [TAR](../../../REFERENCES.md#tar)：https://github.com/VectorYangYiStar/TAR
- [TransReID-SSL](../../../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

骨干来源为TransReID-SSL。其他文献对应迁移的模块；监督、层选择和ReID读出以本实验配置与代码为准。
该配置属于本项目组合或对照设计，没有独立同名论文。

局部分支仅用learnable query读取L1–12 patch；与从SSL全体联合训练的BoQ不同。

## 本配置的模块开关与关键设置

| 参数 | 值 |
|---|---|
| `MODEL.MLDC_READOUT` | `mldc` |
| `MODEL.MLDC_TEMPERATURE` | `0.9` |
| `MODEL.MLDC_LAYERS` | `[9, 10, 11, 12]` |
| `MODEL.MLDC_INIT` | `uniform` |
| `MODEL.MLDC_TRAIN_FEATURE` | `patch_direct_grm` |
| `MODEL.PATCH_BASELINE_CHECKPOINT` | `/root/TransReID-SSL-main/transreid_pytorch/mldc_patch_grm/initialization/deep4_uniform_epoch120.pth` |
| `MODEL.PATCH_LAYERS` | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]` |
| `MODEL.PATCH_HIDDEN_DIM` | `192` |
| `MODEL.PATCH_HEADS` | `4` |
| `MODEL.PATCH_USE_NULL` | `False` |
| `MODEL.PATCH_ZERO_OUTPUT` | `False` |
| `SOLVER.MLDC_HEAD_LR_FACTOR` | `5.0` |
| `SOLVER.MLDC_WEIGHT_LR_FACTOR` | `1.0` |
| `SOLVER.PATCH_WARMUP_EPOCHS` | `5` |
| `SOLVER.PATCH_LOCAL_LOSS_WEIGHT` | `0.1` |
| `TEST.MLDC_RETRIEVAL` | `mldc` |
| `TEST.MLDC_VISUALIZE` | `True` |
| `TEST.MLDC_VIS_SAMPLES` | `6` |
| `TEST.MLDC_VIS_PERIOD` | `10` |
