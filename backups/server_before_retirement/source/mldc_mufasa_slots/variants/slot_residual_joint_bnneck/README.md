# 联合训练 BNNeck 修正版 v3

## 实验思路与模块

L9–12 patch做Slot Attention和跨层匹配；MLDC引导残差、门控及重建/残差度量辅助监督。

配置：`mldc_mufasa_slots/variants/slot_residual_joint_bnneck/config.yml`。检索维度：768。

## 实验流程

从SSL开始，读取后四层patch执行Slot Attention、跨层匹配和融合；MLDC引导整体slot残差，门控融合，并加入残差度量及patch重建辅助目标。

1. 准备Night600数据与配置指定的初始化模型。
2. 使用本目录配置完成训练；记录损失、配置、随机种子及检查点。
3. 使用对应测试入口与同一检查点评估，按本配置记录的特征读出解释结果。

## 运行方式

在工程根目录执行：

```bash
/root/miniconda3/envs/QREID/bin/python mldc_mufasa_slots/train.py --config_file mldc_mufasa_slots/variants/slot_residual_joint_bnneck/config.yml
```

训练设置：epochs=120，batch=64，optimizer=SGD，base_lr=0.0004，warmup=20。

## 输出与已有结果

语义输出路径：`/root/log/transreid/night600/mldc_mufasa_slots/slot_residual_joint_bnneck`。
已有结果原位置：`/root/log/transreid/night600/vit_base_ics_a3pfn_mldc_mufasa/joint_ssl_bnneck_120`。已有结果存在时语义路径作为访问别名；不移动或改写旧检查点。

新的一次训练请另设有明确含义的 `OUTPUT_DIR`，如当前语义路径下的 `seed_1024_repeat_2`；已有结果目录保持为历史记录。

## 参考文献与本项目改动

- [MUFASA](../../../REFERENCES.md#mufasa)：https://github.com/visinf/mufasa
- [A³PFN](../../../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [TransReID-SSL](../../../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

骨干来源为TransReID-SSL。其他文献对应迁移的模块；监督、层选择和ReID读出以本实验配置与代码为准。


从SSL开始全体联合训练；区别于旧冻结增量版与无新增融合BN的联合版。

## 本配置的模块开关与关键设置

| 参数 | 值 |
|---|---|
| `MODEL.MLDC_READOUT` | `mldc` |
| `MODEL.MLDC_TEMPERATURE` | `0.9` |
| `SOLVER.MLDC_HEAD_LR_FACTOR` | `5.0` |
| `MUFASA.LAYERS` | `[9, 10, 11, 12]` |
| `MUFASA.NUM_SLOTS` | `4` |
| `MUFASA.SLOT_DIM` | `256` |
| `MUFASA.ITERATIONS` | `3` |
| `MUFASA.SA_HIDDEN` | `1024` |
| `MUFASA.MATCH` | `True` |
| `MUFASA.FUSION` | `mfusion` |
| `MUFASA.RESIDUAL_SCALE` | `0.1` |
| `MUFASA.ADAPTIVE_GATE` | `True` |
| `MUFASA.MAX_RESIDUAL_SCALE` | `0.2` |
| `MUFASA.RESIDUAL_METRIC_WEIGHT` | `0.1` |
| `MUFASA.RESIDUAL_MARGIN` | `0.1` |
| `MUFASA.HARD_NEGATIVE_CANDIDATES` | `8` |
| `MUFASA.RECON_WEIGHT` | `0.1` |
| `MUFASA.FINAL_LOSS_WEIGHT` | `1.0` |
| `MUFASA.DECODER_HIDDEN` | `2048` |
| `MUFASA.VISUALIZE` | `True` |
| `MUFASA.VIS_PERIOD` | `10` |
| `MUFASA.VIS_SAMPLES` | `6` |
| `MUFASA.CPU_THREADS` | `8` |
