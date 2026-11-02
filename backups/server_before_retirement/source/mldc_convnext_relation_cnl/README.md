# 冻结 ConvNeXt-T 的块间关系蒸馏

独立实验包：`mldc_convnext_relation_cnl`。用户已提供ImageNet预训练权重；独立学生实验于北京时间2026-10-27 20:03:51启动，教师全程冻结，不进行教师微调。

## 权重

使用 Facebook Research 官方 ConvNeXt-T **ImageNet-1K、224、EMA** 检查点：

[convnext_tiny_1k_224_ema.pth](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth)

服务器默认放置位置：

```text
/root/TransReID-SSL-main/transreid_pytorch/pretrain_model/convnext_tiny_1k_224_ema.pth
```

只需下载这个文件。当前加载器使用官方检查点的 `model` 字段并严格校验全部参数；Torchvision 的检查点键名不同，不能直接混用。训练入口不会自动下载；缺少文件时明确报错。

## 数据流

```text
同一张经过原训练增强的 NightReID 图像，256×128
  ├─ 学生原归一化 → 原 Stem + ViT，保留所有原预训练权重
  │                    ├─ 第12层128个 patch token → 通道L2归一化 → R_S
  │                    └─ CLS 9–12 → MLDC → BNneck → 768维检索特征
  │                                                       ↓
  │                                             原 ID + Triplet + CNL
  └─ 转换至 ImageNet 归一化 → 冻结、eval、no_grad 的官方 ConvNeXt-T
                               ↓
                         第三阶段 384×16×8
                               ↓
                        128个 patch → 通道L2归一化 → R_T

  L_relation = mean_{b,i≠j} (R_S[b,i,j] - R_T[b,i,j])²
  L_total = L_ID + L_Triplet + 0.1 L_CNL
            + 0.1 × min(1, epoch/20) × L_relation
```

`R=ZZᵀ`，其中每个 patch 的通道向量归一化。师生通道数允许不同；空间位置一一对应，排除 CLS 和关系矩阵对角线。每图有128个位置、16256个有序非对角关系项，按图像和关系数取平均。本版采用这一明确的 FSKD 启发式迁移目标，不声称完整复现 FSKD 的原始三项损失。

没有最终向量 InfoNCE、可学习师生投影头、额外纹理滤波或注意力蒸馏。教师使用完整标准 ConvNeXt-T 权重严格加载；在线前向只执行到第三阶段，不运行第四阶段和分类头。它提供通用空间特征，尚未验证在夜间域是否有身份收益。

教师与学生共享已完成裁剪、翻转、擦除后的图像。只转换教师输入的均值和标准差，不增加中心裁剪或缩放。教师所有参数冻结，并且学生调用 `train()` 后教师仍保持 `eval()`。原始 ViT 源码不插入新算子，通过一次前向期间的临时 hook 读取 block12 输出。

## 原学生协议

- seed=1234；SGD；batch64；120轮；基础学习率0.0004。
- 原20轮 warmup 和原余弦调度原样保留，包括既有的 warmup/余弦连接方式。
- 原MLDC head倍率5、MLDC权重倍率1、CNL权重0.1、gamma0.9、温度0.05、动量0.2。
- CLS9–12、uniform初始化、原Stem与SSL初始化、数据增强、采样和检索评估均保留。
- 独立输出目录，不覆盖既有实验。没有训练恢复支持。

配置：`configs/convnext_relation_nightreid.yml`。
同预算无KD对照：`configs/baseline_nightreid.yml`。

保存的学生检查点剔除冻结教师，保留学生原全部参数和训练结构描述。测试入口不构建教师，不需要教师权重文件。原MLDC和BNneck继续由原身份任务和CNL优化；块间蒸馏直接监督patch，其梯度经过ViT传播，不直接约束MLDC权重。

## 运行入口（当前学生实验已在后台运行）

在服务器项目根目录、QREID环境中，下载并放置权重后可先验证：

```bash
python -m mldc_convnext_relation_cnl.verify --repository-root . --device cuda --weights pretrain_model/convnext_tiny_1k_224_ema.pth
```

这是batch2的接线和前反向检查，优化器更新0次，不是batch64显存测量。实际完整batch验证入口：

```bash
python -m mldc_convnext_relation_cnl.verify_batch64
```

训练入口（本轮已启动，勿重复运行）：

```bash
python -m mldc_convnext_relation_cnl.train
```

测试入口：

```bash
python -m mldc_convnext_relation_cnl.test TEST.WEIGHT /absolute/path/to/transformer_best.pth
```

## 官方代码来源

`model/convnext_official/UPSTREAM.json` 记录固定版本：
`048efcea897d999aed302f2639b6270aedf8d4c8`。

保留官方完整源码副本和MIT许可证。可执行副本只将 `timm` 工具导入换成本地等价工具；网络运算、参数名称、层数及通道数不变，避免额外引入不兼容的timm版本。

本地验证使用随机初始化的标准网络实例，检查公式、通道维度独立性、教师停止梯度、原学生初始化及随机数状态、KD传播、冻结行为、教师不参与推理和检查点往返。

2026-10-27已追加实际权重验证：用户上传的ImageNet检查点182项状态全部严格匹配，28,589,128个参数；CPU上2张真实增强训练图像的教师输出为2×384×16×8，使用原SSL初始化的学生输出为2×768，KD梯度能到达原Stem和ViT首末层。教师保持冻结，权重文件未修改，优化器更新0次。服务器结果见 `verification_uploaded_weights.json`。

2026-10-27已在RTX4090上通过真实batch64的AMP联合反向检查，峰值分配4.787GiB、保留5.234GiB，教师冻结且不在优化器中。正式120轮学生训练已按用户指令启动，seed1234和原调度保持不变；每10轮评估。接线与显存检查不代表NightReID检索指标提升。
