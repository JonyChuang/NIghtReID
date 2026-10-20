# LPAM＋FFT v4

## 实验思路与模块

全局L9–12固定等权；LPAM读取L3/6/9/12，使用原版FFT路径，四路BN后直接拼接。

配置：`mldc_lpam_fft/variants/fixed_global_lpam_fft_concat/config.yml`。检索维度：3072。

## 实验流程

全局始终取第9—12层原始CLS均值；LPAM读取第3/6/9/12层并执行FFT路径；分别BN后直接拼接四路描述子进行检索。

1. 准备Night600数据与配置指定的初始化模型。
2. 使用本目录配置完成训练；记录损失、配置、随机种子及检查点。
3. 使用对应测试入口与同一检查点评估，按本配置记录的特征读出解释结果。

## 运行方式

在工程根目录执行：

```bash
/root/miniconda3/envs/QREID/bin/python mldc_lpam_fft/train.py --config_file mldc_lpam_fft/variants/fixed_global_lpam_fft_concat/config.yml
```

训练设置：epochs=120，batch=64，optimizer=SGD，base_lr=0.0004，warmup=20。

## 输出与已有结果

语义输出路径：`/root/log/transreid/night600/mldc_lpam_fft/fixed_global_lpam_fft_concat`。
已有结果原位置：`/root/log/transreid/night600/vit_base_ics_lpam_fft_v4_20261017/deep4_fixed_equal_120`。已有结果存在时语义路径作为访问别名；不移动或改写旧检查点。

新的一次训练请另设有明确含义的 `OUTPUT_DIR`，如当前语义路径下的 `seed_1024_repeat_2`；已有结果目录保持为历史记录。

## 参考文献与本项目改动

- [A³PFN](../../../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [TransReID-SSL](../../../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

骨干来源为TransReID-SSL。其他文献对应迁移的模块；监督、层选择和ReID读出以本实验配置与代码为准。


launch.py另支持关闭LPAM的固定等权基线；本表不枚举任意命令行参数组合。

## 本配置的模块开关与关键设置

| 参数 | 值 |
|---|---|
| `MODEL.GLOBAL_LAYERS` | `[9, 10, 11, 12]` |
| `MODEL.GLOBAL_FUSION` | `fixed_equal` |
| `MODEL.LPAM_ENABLED` | `True` |
| `MODEL.LPAM_LAYERS` | `[3, 6, 9, 12]` |
| `MODEL.LPAM_FFT` | `True` |
| `SOLVER.MLDC_HEAD_LR_FACTOR` | `5.0` |
