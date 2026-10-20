# 固定deep4＋末层双监督 v2

## 实验思路与模块

raw CLS9–12均值和末层LN后CLS分开监督，权重0.7/0.3；各分支单位化后等能量拼接。

配置：`mldc_last_cls_dual_supervision/variants/fixed_global_last_cls_unit_concat/config.yml`。检索维度：1536。

## 实验流程

构造后四层固定均值和末层LayerNorm后CLS两个分支；以0.7/0.3监督；推理时两路分别归一化后等能量拼接，得到1536维描述子。

1. 准备Night600数据与配置指定的初始化模型。
2. 使用本目录配置完成训练；记录损失、配置、随机种子及检查点。
3. 使用对应测试入口与同一检查点评估，按本配置记录的特征读出解释结果。

## 运行方式

在工程根目录执行：

```bash
/root/miniconda3/envs/QREID/bin/python mldc_last_cls_dual_supervision/train.py --config_file mldc_last_cls_dual_supervision/variants/fixed_global_last_cls_unit_concat/config.yml
```

训练设置：epochs=120，batch=64，optimizer=SGD，base_lr=0.0004，warmup=20。

## 输出与已有结果

语义输出路径：`/root/log/transreid/night600/mldc_last_cls_dual_supervision/fixed_global_last_cls_unit_concat`。
已有结果原位置：`/root/log/transreid/night600/vit_base_ics_mldc_dual_20261019/dual_lastw03_unit_concat_after_v2_120`。已有结果存在时语义路径作为访问别名；不移动或改写旧检查点。

新的一次训练请另设有明确含义的 `OUTPUT_DIR`，如当前语义路径下的 `seed_1024_repeat_2`；已有结果目录保持为历史记录。

## 参考文献与本项目改动

- [A³PFN](../../../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [TransReID-SSL](../../../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

骨干来源为TransReID-SSL。其他文献对应迁移的模块；监督、层选择和ReID读出以本实验配置与代码为准。
该配置属于本项目组合或对照设计，没有独立同名论文。

当前已恢复为两路监督，检索拼接不使用第三个分类头；0.3是损失权重。

## 本配置的模块开关与关键设置

| 参数 | 值 |
|---|---|
| `MODEL.GLOBAL_LAYERS` | `[9, 10, 11, 12]` |
| `MODEL.GLOBAL_FUSION` | `fixed_equal` |
| `MODEL.LAST_LOSS_WEIGHT` | `0.3` |
| `SOLVER.MLDC_HEAD_LR_FACTOR` | `5.0` |
