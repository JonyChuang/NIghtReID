# 方案B：Patch特征增强＋固定等权后四层MLDC

在现有 `a3pfn_mldc_featenhancer_20261017` 包内原地修改。方案A的120轮结果与80轮最佳检查点保持原路径；其小型源码快照保存在原结果目录的 `scheme_a_source_before_b.zip`，没有新建项目副本。

## 结构

```text
原始ReID归一化图像 → SSL ViT＋stem
    ├─ CLS9–12 → 算术平均 g
    └─ Patch4/6/8 → 各层LayerNorm＋768→192投影
                     → 共享三尺度FEN/SAFA（16×8、8×4、4×2）
                     → g引导的余弦注意力读取空间中心化的V
                     → 三个深度读出等权平均 → 192→768补偿向量
                     → L2方向归一化并匹配g范数，得到r
最终描述符：BN(g＋0.1r)，评估器再做L2归一化，768维
```

这是对FeatEnHancer的FEN/SAFA进行特征空间重新设计，不是原论文RGB模块的直接用法。FEN保留7层卷积及拼接路径，三个分辨率共享参数；小网格用2倍/4倍平均下采样，SAFA使用8组尺度选择，再融合最粗尺度。使用无偏置卷积与复制填充，保持有符号特征残差。原版输入端增强器只用于兼容旧方案A检查点，B模式不执行它。

读取的Q来自当前全局g，K与V来自增强patch。Q/K分别L2归一化后做余弦softmax，固定logit scale=8；V减去空间均值，均匀注意力时补偿为0。没有人体区域划分、多个自由learnable query、预定义前景掩码或额外教师。这个读出约束避免仅用全图公共偏移作为局部补偿，不保证训练后一定学到身份纹理。

第9–12层CLS始终各25%。第4/6/8层patch为当前默认选择，可由 `MODEL.PATCH_LAYERS` 配置；并未用测试标签挑选层数。全部活动主干和补偿分支从第1轮共同训练，不detach、不冻结、不做局部分阶段预热。只优化最终融合特征的完整ID＋Triplet，系数各1，没有新增辅助损失。未使用的骨干norm/fc/gem与原BN固定偏置沿用原处理。

## 初始化与训练

从相同SSL文件 `pretrain_model/vit_base_ics_cfs_lup.pth` 起跑，不接着方案A检查点训练。seed1234，256×128，batch64，SGD4e-4，BN/分类头5倍学习率，120轮，学习率warmup20轮，每10轮验证。验证batch为64。

在服务器项目根目录执行：

```bash
/root/miniconda3/envs/QREID/bin/python -u mldc_feature_enhancement/train.py
```

结果目录：

`/root/log/transreid/night600/vit_base_ics_featenhancer_mldc_20261017/scheme_b_patch468_fixed_equal_120`

匹配控制组关闭patch补偿，保留相同固定等权与ID＋Triplet：

```bash
/root/miniconda3/envs/QREID/bin/python -u mldc_feature_enhancement/train.py --config_file mldc_feature_enhancement/variants/patch_residual_disabled/config.yml
```

控制组输出 `scheme_b_fixed_equal_baseline_120`。同seed主干与分类头初始化相同，不等于全部训练随机序列逐项相同。入口要求可用CUDA并拒绝覆盖已有非空结果目录。

## 诊断与复测

逐轮记录真实batch数、更新数、AMP跳步数与固定层权重；逐次验证记录补偿/全局范数比、FEN修正幅度与空间注意力熵。固定查询索引810、1544、11、2059、160、24会保存叠加原图的patch范数、增强修正与读取注意力；各面板独立色标，范数不能解释为身份贡献。

新方案与旧方案A均可通过当前test入口从保存配置恢复：

```bash
/root/miniconda3/envs/QREID/bin/python -u mldc_feature_enhancement/test.py --weight CHECKPOINT_PATH
```

`verify.py`在服务器用真实ViT和随机输入检查patch收集不改变CLS结果、固定均值、空间常量输入不产生补偿、全部补偿参数及主干的身份梯度、控制组初始化。它不执行优化器更新，不构成训练或性能验证。完整模型、数据和中间特征继续留在服务器。
