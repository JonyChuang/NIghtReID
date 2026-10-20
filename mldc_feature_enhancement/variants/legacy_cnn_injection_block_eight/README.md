# CPPI 仅L8注入

## 实验思路与模块

CNN金字塔来源，仅向第8层patch注入；初始修正预算5%。

配置：`mldc_feature_enhancement/variants/legacy_cnn_injection_block_eight/config.yml`。检索维度：768。

## 实验流程

历史CPPI方案把CNN多尺度特征或ViT金字塔特征注入指定ViT层。当前训练入口已改为纯全局CNL，需要恢复对应历史训练入口后再运行本配置。

1. 准备Night600数据与配置指定的初始化模型。
2. 使用本目录配置完成训练；记录损失、配置、随机种子及检查点。
3. 使用对应测试入口与同一检查点评估，按本配置记录的特征读出解释结果。

## 运行方式

在工程根目录执行：

```bash
历史CPPI配置：当前train_cppi.py仅训练纯MLDC+CNL；该模式需恢复历史训练入口。test_cppi.py可按旧检查点配置评估。
```

训练设置：epochs=120，batch=64，optimizer=SGD，base_lr=0.0004，warmup=20。

## 输出与已有结果

语义输出路径：`/root/log/transreid/night600/mldc_feature_enhancement/legacy_cnn_injection_block_eight`。
已有结果原位置：`/root/log/transreid/night600/vit_base_ics_featenhancer_mldc_20261017/cppi_lite_cnn8_learnable_init5pct_lr10_fixed_equal_120`。已有结果存在时语义路径作为访问别名；不移动或改写旧检查点。

新的一次训练请另设有明确含义的 `OUTPUT_DIR`，如当前语义路径下的 `seed_1024_repeat_2`；已有结果目录保持为历史记录。

## 参考文献与本项目改动

- [A³PFN](../../../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [ViT-CoMer](../../../REFERENCES.md#comer)：https://github.com/Traffic-X/ViT-CoMer
- [TransReID-SSL](../../../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

骨干来源为TransReID-SSL。其他文献对应迁移的模块；监督、层选择和ReID读出以本实验配置与代码为准。


旧配置及模型类存在；当前train_cppi.py会拒绝cppi_lite模式，恢复训练需恢复相应旧入口。

## 本配置的模块开关与关键设置

| 参数 | 值 |
|---|---|
| `MODEL.GLOBAL_LAYERS` | `[9, 10, 11, 12]` |
| `MODEL.GLOBAL_FUSION` | `fixed_equal` |
| `MODEL.CPPI_ENABLED` | `True` |
| `MODEL.CPPI_SOURCE` | `cnn` |
| `MODEL.CPPI_LAYERS` | `[8]` |
| `MODEL.CPPI_DIM` | `192` |
| `MODEL.CPPI_HEADS` | `4` |
| `MODEL.CPPI_INITIAL_RATIO` | `0.05` |
| `MODEL.CPPI_UP_INIT_STD` | `0.01` |
| `SOLVER.MLDC_HEAD_LR_FACTOR` | `5.0` |
| `SOLVER.CPPI_LR_FACTOR` | `10.0` |
| `TEST.CPPI_VISUALIZE` | `True` |
| `TEST.CPPI_VIS_INDICES` | `[810, 1544, 11, 2059, 160, 24]` |
