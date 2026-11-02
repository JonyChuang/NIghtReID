# MLDC＋BoQ 多查询联合 v2

## 实验思路与模块

可学习deep4全局；L4/6/8各4个BoQ query，跨层slot投影后作为局部残差；全体联合训练。

配置：`mldc_boq_local_readout/variants/middle_patch_multi_query_boq/config.yml`。检索维度：768。

## 实验流程

全局使用可学习后四层MLDC；从第4/6/8层patch提取BoQ多查询局部描述子，形成全局残差，并对局部读出提供辅助监督。所有活动模块联合训练。

1. 准备Night600数据与配置指定的初始化模型。
2. 使用本目录配置完成训练；记录损失、配置、随机种子及检查点。
3. 使用对应测试入口与同一检查点评估，按本配置记录的特征读出解释结果。

## 运行方式

在工程根目录执行：

```bash
/root/miniconda3/envs/QREID/bin/python mldc_boq_local_readout/train.py --config_file mldc_boq_local_readout/variants/middle_patch_multi_query_boq/config.yml
```

训练设置：epochs=120，batch=64，optimizer=SGD，base_lr=0.0004，warmup=20。

## 输出与已有结果

语义输出路径：`/root/log/transreid/night600/mldc_boq_local_readout/middle_patch_multi_query_boq`。
已有结果原位置：`/root/log/transreid/night600/vit_base_ics_mldc_boq_20261017/middle468_q4_120`。已有结果存在时语义路径作为访问别名；不移动或改写旧检查点。

新的一次训练请另设有明确含义的 `OUTPUT_DIR`，如当前语义路径下的 `seed_1024_repeat_2`；已有结果目录保持为历史记录。

## 参考文献与本项目改动

- [BoQ](../../../REFERENCES.md#boq)：https://github.com/amaralibey/Bag-of-Queries
- [A³PFN](../../../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [TransReID-SSL](../../../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

骨干来源为TransReID-SSL。其他文献对应迁移的模块；监督、层选择和ReID读出以本实验配置与代码为准。


joint_local目录已由早期单query读取更新为BoQ；旧实现见备份。

## 本配置的模块开关与关键设置

| 参数 | 值 |
|---|---|
| `MODEL.LOCAL_LAYERS` | `[4, 6, 8]` |
| `MODEL.LOCAL_DIM` | `192` |
| `MODEL.LOCAL_HEADS` | `4` |
| `MODEL.LOCAL_READER` | `boq` |
| `MODEL.LOCAL_STRENGTH` | `0.1` |
| `MODEL.MLDC_READOUT` | `mldc` |
| `MODEL.MLDC_TEMPERATURE` | `0.9` |
| `MODEL.MLDC_LAYERS` | `[9, 10, 11, 12]` |
| `MODEL.MLDC_INIT` | `uniform` |
| `MODEL.MLDC_TRAIN_FEATURE` | `mldc` |
| `SOLVER.LOCAL_LOSS_WEIGHT` | `0.1` |
| `SOLVER.MLDC_HEAD_LR_FACTOR` | `5.0` |
| `SOLVER.MLDC_WEIGHT_LR_FACTOR` | `1.0` |
| `TEST.MLDC_RETRIEVAL` | `mldc` |
| `TEST.MLDC_VISUALIZE` | `False` |
| `TEST.MLDC_VIS_SAMPLES` | `6` |
| `TEST.MLDC_VIS_PERIOD` | `10` |
