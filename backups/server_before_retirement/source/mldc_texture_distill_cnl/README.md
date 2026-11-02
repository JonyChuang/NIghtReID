# NightReID：身份监督纹理蒸馏实验

本轮把纹理分支作为训练期教师，通过辅助损失引导 ViT 第 4 层的 Patch 表征。最终检索仍使用原有后四层 CLS 的可学习 MLDC 融合和 768 维 BNneck 特征。

代码已在独立目录实现并同步服务器；本轮仅进行代码与真实图像前向、反向验证，没有训练教师或学生，没有本轮 mAP / Rank-1 结果。随机种子固定为 **1234**。

## 为什么这样结合当前实验

之前的纹理叠加图显示了衣物纹理、人体边界及背景响应。它说明特征中存在局部变化信息，但还不能说明这些信息能区分身份。当前设计先用训练身份的 ID + Triplet 监督学习纹理编码器，再把它的空间关系蒸馏给 ViT。目标是使纹理成为主干学习的约束，同时保持原有预训练网络的前向计算。

这一目标仍需训练结果验证。特别是背景纹理也可能具有较强响应，因此保留相同容量的原始 CNN 特征教师作为对照，以判断增益是否来自纹理变换本身。

该实验借鉴 CNN → ViT 结构蒸馏的思路；固定 Gaussian / Sobel 纹理能量、教师结构和损失组合是本项目的实验设计，不作为 FSKD 或 SSTKD 的完整复现。

## 阶段 A：离线训练教师

输入为同一 NightReID 官方训练集的 256 × 128 行人图像。教师 Stem 从原始 `pretrain_model/vit_base_ics_cfs_lup.pth` 提取 CNN 权重，冻结参数和 BN 统计量。这里没有使用已经在 NightReID 上训练过的第 110 轮模型初始化教师。

```text
RGB 图像 [B, 3, 256, 128]
    ↓ 原始 SSL CNN Stem（冻结）
Stem 特征 [B, 64, 128, 64]
    ↓ 二选一，分别训练两个教师
    ├─ raw：直接自适应平均池化
    └─ texture：高频 + 带通 + 水平/垂直梯度的局部能量，再池化
输入特征 [B, 64, 16, 8]
    ↓ 两层 Conv3×3 + GroupNorm + ReLU（可学习）
空间特征 T [B, 128, 16, 8]
    ↓ GAP → Linear(128→256) → BNneck → ID 分类器
ID + Triplet 身份监督
    ↓ 固定训练 40 轮，保存第 40 轮教师
```

两种教师使用相同学习头、初始化、数据增强和训练预算。Stem 冻结，编码器和身份学习头参与训练；教师使用 SGD、初始基准学习率 0.01、5 轮 warmup 和 cosine 调度。教师成本作为额外的 40 轮准备阶段单独记录。最终教师只进行一次完整 Query/Gallery 评估用于报告，不根据测试指标选择教师轮次。

## 阶段 B：蒸馏到现有 MLDC + CNL 学生

原始 ViT 的 Attention、FFN、Block 和 CLS 路径保持完整，学生从原始 SSL 权重开始训练。接入辅助模块的随机初始化不改变原主干和分类头的随机数序列。

训练时，教师与学生接收**同一个增强后的图像张量**，包括相同的裁剪、翻转和随机擦除位置。取第 4 个 Block 输出的 128 个 Patch Token，去掉 CLS，经训练期的 `LayerNorm + Linear(768→128)` 映射成 `[B,128,16,8]`，与冻结教师的空间特征对齐。

主方案使用 Patch 关系 MSE 和空间能量分布 KL：

\[
L=L_{ID}+L_{Tri}+0.1L_{CNL}
 +0.1\min(1,e/20)(L_{rel}+0.1L_{att}).
\]

`L_rel` 比较每张图像内部 128 × 128 的 Patch 余弦相似关系，去掉对角项；`L_att` 比较 128 个空间位置的特征能量概率分布。这里的能量分布不是 ViT 的 CLS Attention。另有直接归一化特征 MSE 对照配置。

蒸馏梯度传回适配器及 Block 1–4；原有身份、Triplet、CNL 损失继续训练整个有效主干、MLDC 和分类头。教师在学生阶段始终 `eval + no_grad`。教师与适配器在推理时不执行，导出工具会移除其权重。

## 五组对照

| 配置 | 教师输入 | 学生辅助损失 | 用途 |
|---|---|---|---|
| `baseline_nightreid.yml` | 无 | 无 | 当前 MLDC + CNL 基线 |
| `raw_feature_nightreid.yml` | 原始 Stem 池化特征 | 归一化特征 MSE | 普通特征蒸馏对照 |
| `raw_structure_nightreid.yml` | 原始 Stem 池化特征 | 关系 MSE + 0.1 能量 KL | 相同容量 CNN 结构蒸馏对照 |
| `texture_feature_nightreid.yml` | 固定纹理能量 | 归一化特征 MSE | 纹理特征直接匹配对照 |
| `texture_structure_nightreid.yml` | 固定纹理能量 | 关系 MSE + 0.1 能量 KL | 本轮主方案 |

优先比较 `baseline / raw_structure / texture_structure`。如果只有两种教师都提升，说明 CNN 空间监督可能有效；如果纹理组进一步优于 raw 组，才支持纹理变换带来额外价值。feature 两组用于判断关系蒸馏是否优于直接特征匹配。

五组学生共享原有协议：NightReID，500 个训练身份、15,514 张训练图，1,394 Query、26,111 Gallery；seed=1234，120 轮，20 轮相同 warmup，SGD 基准学习率 0.0004，batch=64，SIE 关闭，CNL 权重 0.1、γ=0.9。分类头及辅助适配器学习率为基准的 5 倍，原有偏置倍率保留。比较固定第 120 轮结果，同时保留同一评估频率下的 best 结果。

## 代码入口

| 文件 | 作用 |
|---|---|
| `model/texture_teacher.py` | 两种输入 bank、身份监督教师、三个蒸馏项 |
| `model/make_model_mldc.py` | 保留原主干，读取 Block4 Patch，加入训练适配器 |
| `train_teacher.py` | 两种教师分别固定训练 40 轮 |
| `train.py` | 学生训练入口、协议与教师来源检查 |
| `processor/processor_mldc.py` | 原任务 + 蒸馏训练、损失日志、完整检索评估 |
| `test.py` | 加载本实验完整学生权重进行评估 |
| `export_inference.py` | 导出可由原 MLDC 基线加载的推理权重 |
| `verify.py` | 真实图像、零优化器更新的代码验证 |

## 将来开始训练时的命令

以下命令仅为入口说明，**本轮没有执行**。工作目录为 `/root/TransReID-SSL-main/transreid_pytorch`，使用原 QREID 环境和可见 GPU。两个教师各训练一次，随后对应的 feature / structure 学生共用同一教师。

```bash
python -m mldc_texture_distill_cnl.train_teacher --config_file mldc_texture_distill_cnl/configs/raw_structure_nightreid.yml
python -m mldc_texture_distill_cnl.train_teacher --config_file mldc_texture_distill_cnl/configs/texture_structure_nightreid.yml

python -m mldc_texture_distill_cnl.train --config_file mldc_texture_distill_cnl/configs/baseline_nightreid.yml
python -m mldc_texture_distill_cnl.train --config_file mldc_texture_distill_cnl/configs/raw_structure_nightreid.yml
python -m mldc_texture_distill_cnl.train --config_file mldc_texture_distill_cnl/configs/texture_structure_nightreid.yml
```

教师输出：`/root/log/transreid/nightreid/mldc_texture_distill_cnl/teachers/{raw,texture}/teacher_final.pth`。

学生输出：`/root/log/transreid/nightreid/mldc_texture_distill_cnl/students/{配置名}_seed1234_matched120/`。不覆盖既有实验，入口要求使用空输出目录；当前入口不支持断点续训。

```bash
python -m mldc_texture_distill_cnl.test --config_file mldc_texture_distill_cnl/configs/texture_structure_nightreid.yml TEST.WEIGHT /path/to/transformer_120.pth
python -m mldc_texture_distill_cnl.export_inference /path/to/transformer_120.pth /path/to/texture_distilled_inference.pth
```

## 本轮验证范围

使用 4 张真实 NightReID 训练图像、2 个身份及跨相机样本进行 CPU 验证，优化器更新数为 **0**。验证包括：原 ViT 类与基线一致、五组配置的原有协议相同、初始化输出完全一致、已训练基线输出完全一致、教师身份监督反向、蒸馏梯度到达 ViT、教师权重与 BN 统计量冻结、完整学生保存/加载、导出后原模型加载输出一致。

CPU 反向检查为了控制内存，仅临时打开 Block4 和学习头的梯度；正式训练代码默认训练整个有效主干。当前无可见 GPU，因此没有验证 CUDA 混合精度、正式 batch 的显存占用及长程训练表现。验证中未训练的教师仅作为临时测试对象，正式训练入口会拒绝将其作为已完成教师加载。

数据流及损失定义见 `docs/module_dataflow.md`，图形版本为同目录的 `.drawio / .png / .pdf`。
