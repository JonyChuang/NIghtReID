# MLDC patch直接监督 + GRM（v2，2026-10-16）

在原独立副本上修改，代码包路径不变。v1代码有修改前备份，已完成v1训练日志与检查点不改动。本版本默认输出到独立的stage2_patch_direct_grm目录。

## 模型

- 从initialization/deep4_uniform_epoch120.pth初始化后四层CLS基线，不加载v1的patch补偿权重。
- 固定原ViT、9—12层CLS融合、原BN参数和统计量，以及原全局分类头；冻结模块持续保持eval模式。
- 第1—12层post-block、最终LayerNorm之前的patch参与读取。各层独立LayerNorm，共享K/V，192维4头soft attention。
- 查询仅为可学习query token；局部读取与局部身份监督均不接收CLS。
- 聚合向量经192→768投影、LayerNorm、768→256→768 MLP得到a。投影正常初始化，不再使用零输出或空token。
- 局部监督为ID(local_classifier(local_BN(a))) + Triplet(a)，局部BN与分类头只用于训练和独立诊断。
- GRM使用全局、局部内容预测gamma，f=g+gamma*stopgrad(norm(g))*a/(norm(a)+eps)。默认gamma_max=0.1，启用融合时初始gamma=0.01。
- 默认仍检索L2(原BN(f))，768维，无末层CLS额外相加，无重排序。

## 两阶段与损失

1. epoch 1—5：local_warmup，仅局部ID+Triplet；gamma强制为0，最终检索仍为原基线。局部出口非零，读取器可以正常获得梯度。
2. epoch 6—120：joint_fusion，最终ID+Triplet + 0.1*(局部ID+Triplet)；启用GRM。

PATCH_WARMUP_EPOCHS=5指局部启动阶段，独立于原SOLVER.WARMUP_EPOCHS=20的学习率warmup。

原全局分类头冻结但不会切断对f的梯度；局部监督不经过gamma，gamma为0时仍可学习。上限只约束BN前补偿范数，不保证检索排序不发生错误翻转。

## 启动和测试

从项目根目录执行，当前没有启动训练：

```sh
python mldc_patch_grm/train.py
python mldc_patch_grm/test.py TEST.WEIGHT /path/to/grm_v2/transformer_best.pth
```

默认配置路径保持configs/Night600/vit_base_ics_mldc_patch_compensation.yml。训练入口拒绝复用已有训练结果的输出目录。此入口从基线开始完整两阶段训练，不提供优化器断点恢复。

## 记录和兼容性

- training.csv记录两阶段、完整epoch损失、全局与局部分类准确率、平均gamma。
- 每次验证记录最终检索、冻结anchor_bn、独立local_bn指标；第5轮额外验证并保存局部启动边界检查点。
- 记录gamma均值/标准差/范围、patch层分配、注意力熵、补偿/全局范数比。热力图只表示注意力分配。
- v2检查点保存结构、局部启动轮数和实际阶段/epoch，测试恢复时会恢复同一融合行为。
- v1 patch检查点与v2结构不同，明确拒绝加载；原始deep4基线仅用于明确配置的初始化。
- 旧verification目录会改名为verification_v1，新验证记录位于verification_grm_v2。旧版本验证不能作为本版本验证。

## 验证范围

服务器当前内存较小。本轮只做静态检查和小型局部模块/替代骨干接口、无优化器更新的梯度与检查点测试，不加载完整ViT、不执行真实数据集评估、不训练。完整骨干与GPU运行验证需足够资源后再进行。
