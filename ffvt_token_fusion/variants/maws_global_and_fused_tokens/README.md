# 跨层 Token 融合双分支

## 实验思路与模块

从L3/6/9/11按MAWS各选12个patch，独立末层融合；全局与融合分支双监督。

配置：`ffvt_token_fusion/variants/maws_global_and_fused_tokens/config.yml`。检索维度：1536。

## 实验流程

原始ViT在第3/6/9/11层按MAWS各选12个patch，组成融合序列；原全局末层和独立融合末层分别监督，推理拼接两路BN后特征。

1. 准备Night600数据与配置指定的初始化模型。
2. 使用本目录配置完成训练；记录损失、配置、随机种子及检查点。
3. 使用对应测试入口与同一检查点评估，按本配置记录的特征读出解释结果。

## 运行方式

在工程根目录执行：

```bash
/root/miniconda3/envs/QREID/bin/python ffvt_token_fusion/train.py --config_file ffvt_token_fusion/variants/maws_global_and_fused_tokens/config.yml
```

训练设置：epochs=120，batch=64，optimizer=SGD，base_lr=0.0004，warmup=20。

## 输出与已有结果

语义输出路径：`/root/log/transreid/night600/ffvt_token_fusion/maws_global_and_fused_tokens`。
已有结果原位置：`/root/log/transreid/night600/vit_base_ics_alf_end2end`。已有结果存在时语义路径作为访问别名；不移动或改写旧检查点。

新的一次训练请另设有明确含义的 `OUTPUT_DIR`，如当前语义路径下的 `seed_1024_repeat_2`；已有结果目录保持为历史记录。

## 参考文献与本项目改动

- [FFVT](../../../REFERENCES.md#ffvt)：https://github.com/Markin-Wang/FFVT
- [TransReID-SSL](../../../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

骨干来源为TransReID-SSL。其他文献对应迁移的模块；监督、层选择和ReID读出以本实验配置与代码为准。


目录仍叫 alf_end2end，当前实际为 FFVT；不能按名称当作旧 ALF。

## 本配置的模块开关与关键设置

| 参数 | 值 |
|---|---|
| `MODEL.FFVT_LAYERS` | `[3, 6, 9, 11]` |
| `MODEL.FFVT_TOKENS_PER_LAYER` | `12` |
| `SOLVER.FFVT_HEAD_LR_FACTOR` | `5.0` |
