# 当前副本：单分支MLDC＋CNL

默认入口仍为train_cppi.py，在原包内直接修改。当前模型仅保留ViT的Block 9–12融合CLS，已从活动模型移除CNN、MRFP、浅层Reader、浅层BN/分类头及两支拼接。旧类仅用于恢复历史检查点，不参与默认训练。

## 前向与损失

`z = (CLS9 + CLS10 + CLS11 + CLS12) / 4`

`g = L2(BN(z))`，唯一检索描述子为768维。融合方式继续固定等权，主干、BN与分类头从第1轮共同训练。CNL只参与训练，不改变推理路径。

`L = ID_global + Triplet_global + 0.1 * CNL_global`

各身份损失继续使用原配置ID_LOSS_WEIGHT、TRIPLET_LOSS_WEIGHT。保留基线身份监督，仅将附加方法限定为CNL；SHALLOW_LOSS_WEIGHT=0，不计算浅层损失。

## 单分支CNL的定义

原来的双路CNL根据深层、浅层两个空间的邻居交集筛选正样本。当前没有浅层分支，因此明确改为**融合特征单空间的监督跨摄像头邻居损失**，不声称与原论文双路交集CNL完全相同。

使用BN后L2融合特征建立一份逐训练图片记忆库。batch索引对应训练图片，验证/测试图片不进入记忆。每个query先排除自身，再在候选融合特征中筛选相似度大于gamma乘最高候选相似度的邻居，限制为真实同身份、不同摄像头；默认gamma=0.9。

对筛选到的跨摄像头正样本计算平均对数概率，分母包含被选正样本与其他身份负样本。未被选中的同身份样本及同摄像头同身份样本不作为负样本。没有有效正邻居或负样本时，该anchor的CNL为0，普通ID和Triplet继续优化。

默认temperature=0.05、记忆动量=0.2、外部CNL系数=0.1，从第1轮启用。候选中的当前batch目标替换过时记忆，并detach；重复抽到一张训练图片时，其多个增强特征共用一个槽，取平均方向。记忆仅在AMP确认优化器更新成功后写入，AMP跳步不更新；前向、反向和评估不会写记忆。

训练日志记录原始CNL、有效anchor比例、无正邻居比例、跨相机正样本筛选覆盖率、记忆图片数。单空间没有“深浅交集”，统计字段使用empty_positive_sets。

## 配置与启动

|配置|含义|
|---|---|
|cppi.yml|默认：融合CLS＋ID/Triplet＋0.1×单分支CNL，无CNN|
|cppi_baseline.yml|同一模型、初始化与训练设置，CNL=0|
|cppi_no_cnl.yml|cppi_baseline.yml的配置别名，输出到相同对照目录，不需要重复跑|

默认MODEL.ENHANCEMENT_MODE=mldc_cnl、CNN_SD_ENABLED=False、CPPI_ENABLED=False。默认不创建CNN或浅层Reader，即使历史兼容配置项仍在schema中也不会启用。

默认输出：

`/root/log/transreid/night600/vit_base_ics_featenhancer_mldc_20261017/mldc_global_cnl01_fixed_equal_120`

CNL=0对照输出：

`/root/log/transreid/night600/vit_base_ics_featenhancer_mldc_20261017/mldc_global_no_cnl_fixed_equal_120`

保留既有日志、检查点和分析。seed1234、原SSL、batch64、原增强、SGD、120轮、20轮预热、每10轮验证保持原样。主干参考LR为4e-4，BN/分类头为5倍，偏置另乘2，沿用余弦调度；没有CNN/Reader优化器参数组。

在原服务器工程根目录，后续显式启动：

```bash
/root/miniconda3/envs/QREID/bin/python -u mldc_feature_enhancement/train_cppi.py
```

对照添加 `--config_file mldc_feature_enhancement/variants/global_cls_without_cnl/config.yml`。安装和验证不启动训练。入口从原SSL开始新实验，拒绝非空输出目录；当前不提供优化器/采样状态的完整断点续训。

## 保存、评估与兼容

新架构标识mldc_global_cnl_20261020。检查点保存模型、配置、架构、单份CNL记忆、训练图片索引指纹及评估结果。保存记忆不等于支持完整断点续训。

仅评估global读出的mAP、Rank-1/5/10和逐查询AP/Rank-1，用global mAP选择最佳检查点。推理不创建或访问CNL记忆，也没有浅层描述子与拼接。

```bash
/root/miniconda3/envs/QREID/bin/python mldc_feature_enhancement/test_cppi.py --weight <checkpoint> --device cuda
```

test_cppi.py仍按旧检查点的配置和架构恢复历史CNN、CLS11/CNL及CTI模型；原融合CLS指导CNN的120轮权重继续可读。旧CNN配置中的CNN_SD_ENABLED及查询来源不会被新默认覆盖。

## 检查

verify_cppi.py调用verify_global_cnl.py：检查解析可计算的邻居损失、自身/身份/摄像头过滤、未选同身份不作负样本、空邻居/无负样本、重复图片记忆槽、FP32损失、单记忆库、梯度、无浅层参数、768维输出、CNL=0对照相同初始化，以及模型/记忆检查点恢复。

本地验证使用归档真实ViT实现、生成图和随机权重；服务器验证载入原SSL及真实训练图，使用CPU bfloat16前向/反向，均不执行优化器step或训练循环。当前报告为verification_mldc_cnl_local.json和verification_mldc_cnl_server.json，不是新方法的检索性能或GPU显存测试。

CNL思想来源及单空间适配说明见cppi_provenance.json。历史源码生成脚本不应重新生成覆盖当前实现。
