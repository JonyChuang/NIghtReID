# QCAT Rank

## 实验思路与模块

保留QNorm排序先验模型、独立backbone和训练处理器。

配置：`experiment_variants/qnorm_context_rank/config.yml`。检索维度：768。

## 实验流程

从项目根目录显式指定本配置，加载SSL与数据集，执行该入口已有的训练和评估流程。旧QNorm/QCAT配置需要先恢复配套旧入口，不能直接套用当前多层融合train.py。

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

语义输出路径：`/root/log/transreid/night600/experiment_variants/qnorm_context_rank`。
已有结果原位置：`/root/log/transreid/night600/vit_base_ics_qcat_rank`。已有结果存在时语义路径作为访问别名；不移动或改写旧检查点。

新的一次训练请另设有明确含义的 `OUTPUT_DIR`，如当前语义路径下的 `seed_1024_repeat_2`；已有结果目录保持为历史记录。

## 参考文献与本项目改动

- [TransReID-SSL](../../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

骨干来源为TransReID-SSL。其他文献对应迁移的模块；监督、层选择和ReID读出以本实验配置与代码为准。
该配置属于本项目组合或对照设计，没有独立同名论文。

没有单独train_qcat_rank.py；当前train.py固定导入多层融合模型，旧入口/依赖需恢复。
