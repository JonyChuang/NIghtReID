# 纯 MLDC：CNL=0 对照

## 实验思路与模块

与当前CNL模型同结构，关闭CNL；无浅层分支。

配置：`mldc_feature_enhancement/variants/global_cls_without_cnl_alias/config.yml`。检索维度：768。

## 实验流程

固定后四层CLS均值形成全局描述子；使用ID与Triplet监督，并按配置开启或关闭跨相机近邻CNL损失和记忆更新；使用同一全局读出比较。

1. 准备Night600数据与配置指定的初始化模型。
2. 使用本目录配置完成训练；记录损失、配置、随机种子及检查点。
3. 使用对应测试入口与同一检查点评估，按本配置记录的特征读出解释结果。

## 运行方式

在工程根目录执行：

```bash
/root/miniconda3/envs/QREID/bin/python mldc_feature_enhancement/train_cppi.py --config_file mldc_feature_enhancement/variants/global_cls_without_cnl_alias/config.yml
```

训练设置：epochs=120，batch=64，optimizer=SGD，base_lr=0.0004，warmup=20。

## 输出与已有结果

语义输出路径：`/root/log/transreid/night600/mldc_feature_enhancement/global_cls_without_cnl`。
已有结果原位置：`/root/log/transreid/night600/vit_base_ics_featenhancer_mldc_20261017/mldc_global_no_cnl_fixed_equal_120`。已有结果存在时语义路径作为访问别名；不移动或改写旧检查点。

新的一次训练请另设有明确含义的 `OUTPUT_DIR`，如当前语义路径下的 `seed_1024_repeat_2`；已有结果目录保持为历史记录。

## 参考文献与本项目改动

- [A³PFN](../../../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [SDCL](../../../REFERENCES.md#sdcl)：https://github.com/yangbincv/SDCL
- [TransReID-SSL](../../../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

骨干来源为TransReID-SSL。其他文献对应迁移的模块；监督、层选择和ReID读出以本实验配置与代码为准。

CNL在当前版本中是有监督的单全局空间跨相机近邻损失，保留SDCL的近邻与记忆思路；没有原论文的双空间交集与完整无监督训练流程。
两份YAML为配置别名，输出目录相同，合并为一条，不重复计数。

## 本配置的模块开关与关键设置

| 参数 | 值 |
|---|---|
| `MODEL.GLOBAL_LAYERS` | `[9, 10, 11, 12]` |
| `MODEL.GLOBAL_FUSION` | `fixed_equal` |
| `MODEL.CNN_SD_ENABLED` | `False` |
| `MODEL.CPPI_ENABLED` | `False` |
| `SOLVER.MLDC_HEAD_LR_FACTOR` | `5.0` |
| `SOLVER.CNL_WEIGHT` | `0.0` |
