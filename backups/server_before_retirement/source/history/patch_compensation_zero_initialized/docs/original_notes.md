# MLDC跨层局部补偿实验（2026-10-16）

本副本基于后四层等权起始、已训练120轮的MLDC检查点，原指标为mAP 10.7185%、Rank-1 21.8349%。原检查点在服务器内复制到本目录的initialization文件夹，不修改原实验。

## 模型与监督

- 固定ViT骨干、9—12层CLS融合权重、BN参数和运行统计量，并持续保持它们为eval模式。
- 读取1—12层Transformer block输出、最终LayerNorm之前的patch。当前分辨率下，每层128个768维patch。
- 各层独立LayerNorm，共享K/V投影；全局CLS上下文加共享可学习查询，使用192维、4头注意力读取。
- 空键可学习，空值固定为零。空logit增加log(真实token数)并加可学习每头偏置，使等内容logit时空选项与全部真实patch各占一半总权重；避免单个空选项因1536个候选而被数量淹没。这是候选数量先验，不是置信度或正确性标签。
- 输出投影无偏置，权重零初始化。空值不会通过输出偏置变成额外改动；初始化时补偿严格为零。
- 最终f=g+delta，ID监督BN(f)，Triplet监督f。只训练patch读出和分类头，不增加其他损失。
- 默认检索L2(BN(f))，仍为768维，不额外加末层CLS，不做重排序。

## 入口

从项目根目录执行（当前未启动训练）：

```sh
python mldc_patch_grm/train.py
python mldc_patch_grm/test.py TEST.WEIGHT /path/to/this_experiment/transformer_best.pth
```

默认配置位于configs/Night600/vit_base_ics_mldc_patch_compensation.yml。配置中的OUTPUT_DIR为本实验独立日志目录；训练入口拒绝覆盖已有运行记录或模型的输出目录。

本实验从旧120轮模型进行权重初始化，新实验epoch从1开始；不是恢复旧优化器或旧学习率进度。训练仍需要CUDA GPU。

## 诊断与检查点

每次验证同时记录补偿后检索和冻结anchor_bn检索指标、各头空权重、各层patch注意力总量、补偿/全局范数比。等权CLS诊断只计算9—12层。可视化记录各层patch注意力分配；这些图不能单独证明身份信息的因果贡献。

新检查点保存实验名、冻结阶段、patch层/维度/头数/空选项等结构元数据，重新加载时核对。只有明确配置的旧基线检查点允许缺失新增patch参数；新检查点严格加载，避免漏载补偿参数而静默变回基线。

本副本有自己的model、backbone、loss、processor和配置。项目通用config、datasets、solver、utils与数据集仍复用项目根目录。服务器文件和诊断结果不下载到本地。

## 验证边界

验证脚本仅进行CPU前向、无参数更新的梯度检查、优化器参数分组检查和检查点往返加载。不会调用训练循环或optimizer.step，不执行完整数据集评估，也不产生新的性能结论。
