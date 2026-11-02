# ViTMatte细节捕获模块迁移：MLDC＋CNL

本包在调度已对齐的可学习MLDC＋CNL上，增加ViTMatte的独立细节读取路径。种子固定1234。代码接入和验证不代表检索提升，完整训练由独立入口显式启动。

## 结构

```text
归一化RGB ──→ 原SSL ViT ──→ CLS9–12可学习融合 ──→ 原BN → 全局单位向量g（768维）
    │              │                                  │
    │              └─→ 末层patch（16×8×768）            └─→ 局部读取的Query
    │                       ↓ LayerNorm
    └─→ ConvStream ──→ 四级上采样、拼接、卷积融合 ──→ 全局引导池化 → BN → r（256维）
                                                                  │
             检索向量 f = concat(sqrt(0.9)×g, sqrt(0.1)×r) ←─────────┘
```

ViTMatte上游代码固定于commit `8cd7ef068380977c3962c4cb733cb1fe7f2241a5`。

- 保留官方ConvStream，通道48、96、192；256×128输入对应128×64、64×32、32×16细节图。
- 保留全部四级Fusion_Block：双线性2倍上采样，与对应细节图拼接，再执行Conv3×3＋BN＋ReLU。输出通道256、128、64、32，最终细节图为256×128。
- 输入从原RGB＋trimap的4通道适配为RGB的3通道；没有引入trimap、人体分割标签、透明度预测头或抠图损失。
- 末层patch增加可学习LayerNorm以适配当前主干的原始post-block输出。MLDC仍使用原来的CLS9–12，主干只新增一个可选patch返回接口。
- 使用原始MLDC向量生成32维Query，读取融合后的32通道细节图，再投影为256维局部描述子；均值池化对照使用相同细节流及融合模块。
- 本次采用官方DCM的独立RGB卷积流，没有复用ViT内部的卷积stem，也没有改变ViT内部注意力或向中间层注入特征。
- 未加载ViTMatte抠图权重；仅复用当前实验原SSL初始化，新增模块由种子1234初始化。新增参数的创建置于独立RNG上下文，保持原全局参数及随后RNG状态与基线一致。

官方原始实现逐字保存在model/vitmatte_official.py；许可证为licenses/ViTMatte-MIT.txt。原始仓库及论文链接见provenance.json。

## 训练与检索

```text
L = ID(g_raw) + Triplet(g_raw)
  + 0.1 × CNL(g_unit)
  + DETAIL_LOSS_WEIGHT × [ID(f) + Triplet(f)]
```

其中分类使用对应BNNeck后的特征；融合路径的Triplet直接约束实际检索单位向量f。DETAIL_LOSS_WEIGHT初始设为1，是本次新实验配置，不是已有验证得到的最优值。

CNL仍只使用768维全局空间及一份训练图片记忆库。只在成功的AMP优化器更新后写入记忆，不改变原来的同身份、跨摄像头邻居规则。

默认融合alpha固定0.1。两个单位向量的平方根加权拼接是1024维单位向量，其余弦相似度精确等于0.9×全局余弦＋0.1×局部余弦。训练后的局部分支需要重新提取特征；先前冻结验证的缓存不能作为新模型特征使用。

主干、可学习MLDC、DCM、读出与分类头从第1轮共同训练。保留原batch64、120轮、20轮预热、SGD基础学习率0.0004、原数据增强和逐参数组缩放余弦调度。DCM与池化模块默认基础学习率1倍，原头部及新增BN／分类头5倍，融合logit1倍；偏置倍率沿用原优化器。

## 实验入口

在现有TransReID工程根目录和原QREID环境中执行：

```bash
python -m mldc_vitmatte_cnl.train
python -m mldc_vitmatte_cnl.train --config_file mldc_vitmatte_cnl/configs/baseline.yml
python -m mldc_vitmatte_cnl.train --config_file mldc_vitmatte_cnl/configs/dcm_mean.yml
```

三行分别是DCM全局引导读取、无DCM独立训练基线、DCM均值池化控制。各自使用不同输出目录，不接替或覆盖原实验。baseline.yml不计算融合辅助损失，恢复原全局ID＋Triplet＋CNL；默认与均值池化实验均计算融合辅助损失。以后比较模块收益时仍需考虑该额外监督，并可增加等容量重映射控制。

入口只接受种子1234，默认从相同SSL重新训练。没有实现训练状态的完整断点续训。评估入口：

```bash
python -m mldc_vitmatte_cnl.test --weight <本实验检查点>
```

默认代码验证只执行推理，不创建优化器、不更新参数：

```bash
python -m mldc_vitmatte_cnl.verify --report mldc_vitmatte_cnl/verification_cpu.json
```

verify的`--allow-update`属于显式选用的临时集成测试，会在临时模型上更新一次参数；默认关闭。此次代码交付不启动训练任务。

保存的检查点包含配置、精确架构、全部参数、BN缓冲区和CNL记忆；按架构严格恢复。load_global_checkpoint仅用于显式导入旧MLDC＋CNL全局路径作诊断，不会自动将未训练的DCM变成训练后分支。

## 日志与判断标准

- evaluation.csv：实际融合描述子的mAP、Rank-1/5/10，以及相对同检查点全局分支的纠正、破坏、净增查询数。
- branch_metrics.csv：融合、全局、局部分支分别评估。
- evaluation_diagnostics.jsonl：逐查询AP／Rank-1及对应分支信息。
- training.csv：两条监督路径的ID／Triplet、CNL、AMP成功更新数与邻居覆盖率。
- layer_weights.csv：CLS9–12的学习过程；run_metadata.json记录初始化、参数规模与完整配置。

同一检查点上的global和fused比较属于路径诊断，独立训练baseline.yml才是方法增益对照。标准测试使用完整Night600查询／图库；此前两轮抽样冻结结果用于设计动机，不是本模型的训练后成绩。
