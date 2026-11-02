# 旧随机初始化配置（当前入口不支持）

## 实验思路与模块

后四层CLS以可学习Softmax权重融合，均匀初始化；在原ID与Triplet基础上比较开启或关闭单空间监督式跨相机CNL。随机配置是继承文件，当前入口只接受均匀初始化。

配置：`mldc_learnable_global_cnl/variants/legacy_random_start_config/config.yml`。检索维度：768。

## 实验流程

由相同SSL初始化ViT；第9—12层CLS以均匀起步的可学习Softmax权重融合，联合优化ID、Triplet和按配置开关的跨相机CNL。记忆库在成功更新优化器后更新；检索使用768维BN后全局描述子，不恢复优化器或采样器状态。

1. 准备Night600数据与配置指定的初始化模型。
2. 使用本目录配置完成训练；记录损失、配置、随机种子及检查点。
3. 使用对应测试入口与同一检查点评估，按本配置记录的特征读出解释结果。

## 运行方式

在工程根目录执行：

```bash
历史实现/配置：使用前按包内说明恢复对应入口；不提供指向当前其他模型的训练命令。
```

训练设置：epochs=120，batch=64，optimizer=SGD，base_lr=0.0004，warmup=20。

## 输出与已有结果

语义输出路径：`/root/log/transreid/night600/mldc_learnable_global_cnl/legacy_random_start_config`。
已有结果原位置：`/root/log/transreid/night600/vit_base_ics_mldc_learnable_cnl_20261020/deep4_random_learnable_120`。已有结果存在时语义路径作为访问别名；不移动或改写旧检查点。

新的一次训练请另设有明确含义的 `OUTPUT_DIR`，如当前语义路径下的 `seed_1024_repeat_2`；已有结果目录保持为历史记录。

## 参考文献与本项目改动

- [A³PFN](../../../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [SDCL](../../../REFERENCES.md#sdcl)：https://github.com/yangbincv/SDCL
- [TransReID-SSL](../../../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

骨干来源为TransReID-SSL。其他文献对应迁移的模块；监督、层选择和ReID读出以本实验配置与代码为准。


不恢复优化器和采样器状态；训练从相同SSL预训练重新开始。随机初始化配置不适用于当前训练入口。

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
