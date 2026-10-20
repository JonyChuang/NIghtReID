# MLDC固定后四层＋末层CLS＋NightReID式拼接监督

独立包 `a3pfn_mldc_dual_supervision_20261019`。原SSL初始化，共享ViT主干；没有patch增强、区域划分或新排序损失。安装和导入不会开始训练。

- 融合：`g = mean(raw_CLS9, raw_CLS10, raw_CLS11, raw_CLS12)`，权重固定各0.25。
- 末层：`c = base.norm(raw_CLS12)`，沿用SSL末端仿射LayerNorm；该LayerNorm参与训练。
- 拼接：`z = cat(g, c)`，直接拼接BN前特征，不先对每支L2。
- ID：g、c、z各有独立BNNeck和分类器，维度分别768、768、1536。
- Triplet：分别使用BN前的g、c、z，保持原soft-margin batch-hard实现。
- 默认总损失：`0.7*(ID_g + Tri_g) + 0.3*(ID_c + Tri_c) + (ID_z + Tri_z)`。`MODEL.LAST_LOSS_WEIGHT=0.3`表示两个来源分支监督内部的末层比例，融合比例自动为`1-w`；新增`MODEL.CONCAT_LOSS_WEIGHT=1.0`。三个系数直接相加，不再除以2或3。
- 默认检索：对齐官方Night600配置的`TEST.NECK_FEAT=before`，分别对完整g、c、z做L2；拼接是1536维向量整体归一化。支持after读出，使用三个各自BN后的完整描述符。两种方式都不强制两支各占50%。
- 活动主干与三套头从第1轮共同训练。只排除没有进入前向的fc/gem；BN偏置沿用基线常量0约定。

迁移依据：[官方模型](https://github.com/msm8976/NightReID/blob/main/model/make_model.py)、[三分支训练](https://github.com/msm8976/NightReID/blob/main/processor/processor.py)、[Night600配置](https://github.com/msm8976/NightReID/blob/main/configs/night600/EDA.yml)、[整体归一化评估](https://github.com/msm8976/NightReID/blob/main/utils/metrics.py)。原项目是增强/原图两个独立ViT；这里沿用共享ViT的MLDC/末层来源，不迁移IED或双主干。原项目三支系数均1；此处按用户要求采用0.7/0.3/1。BN1d不混合通道，不应宣称拼接BN本身建立跨分支交互；拼接分类器和联合距离提供直接监督。

BN前g和末层LN后c的尺度可能不同，原始拼接会保留这种差异，不能把0.3监督系数解释为检索占比。评估记录实际拼接描述符前768维的能量比例；未加入自动尺度平衡，以便检验直接迁移的行为。

默认Night600、seed1234、batch64、SGD4e-4、120轮、20轮warmup，三套新头的标称学习率倍率均5。使用原MLDC的cosine scheduler factory，保持其共享标量warmup和最低学习率行为；日志记录实际LR。每10轮在同一检查点上评估fusion、last、concat并保存一次权重，最佳文件通过硬链接指向对应epoch文件。增强和采样配置保留原MLDC实验。

在服务器工程根目录执行以下命令才会训练：

```text
cd /root/TransReID-SSL-main/transreid_pytorch
/root/miniconda3/envs/QREID/bin/python -u mldc_last_cls_dual_supervision/train.py
```

训练要求CUDA，拒绝CPU回退或覆盖非空输出目录。默认0.3版输出：`/root/log/transreid/night600/vit_base_ics_mldc_dual_20261019/nightreid_cat3_lastw03_catw10_before_120`。

后续比较0.5时可用相同代码和独立输出目录：

```text
/root/miniconda3/envs/QREID/bin/python -u mldc_last_cls_dual_supervision/train.py MODEL.LAST_LOSS_WEIGHT 0.5 OUTPUT_DIR /root/log/transreid/night600/vit_base_ics_mldc_dual_20261019/nightreid_cat3_lastw05_catw10_before_120
```

这里只更改两个来源分支的监督占比，拼接监督系数仍为1。当前未启动任何版本的训练。

同一检查点的三分支评估：

```text
/root/miniconda3/envs/QREID/bin/python -u mldc_last_cls_dual_supervision/test.py --checkpoint /absolute/path/transformer_120.pth
```

对相同检查点追加`--neck-feat after`可评估BN后读出；默认沿用检查点中的before配置。每次评估会记录读出位置和拼接能量占比。新变体标识和三套头结构与原双监督检查点不兼容，不做宽松加载。

主要记录：

- `training.csv`：三分支ID/Triplet、三项系数和加权损失贡献、分类准确率、真实更新与AMP跳步。
- `evaluation.csv`：每轮三行，branch明确区分fusion/last/concat。
- `evaluation_concat.csv`：拼接主结果。
- `evaluation_diagnostics.jsonl`：拼接相对每个单分支纠正/破坏/净变化、两分支成功并集及流失。
- `evaluation/epoch_XXX/`：三个完整逐查询表和同序特征缓存，仅保留服务器。距离计算按32个query分块。
- `joint_gradients.jsonl`：共享主干首尾块、末端LayerNorm、三套头的梯度范数与实际LR。
- `transformer_XX.pth`：每次评估的完整快照；`transformer_best_fusion/last/concat.pth`分别按各分支mAP选出，报告时必须注明epoch，不能跨epoch拼接指标。

CPU验证包含原SSL、真实训练图片、完整12层ViT前后向、三套独立头、拼接损失回传到两个来源特征、配置/检查点重载、before/after实际描述符距离与同身份同摄像头排除；使用范数不同的样本确保错误替代为平均距离会被检查发现。内存受限的CPU反向验证临时使用激活重计算；训练源码保持常规前向。验证优化器更新0次，不代表已获得检索性能结果。
