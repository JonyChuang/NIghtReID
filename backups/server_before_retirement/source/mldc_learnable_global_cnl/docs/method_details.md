# 可学习权重的MLDC + CNL独立实验

原实验包 `a3pfn_mldc_deep4_random` 的独立副本，基线为历史 `deep4_uniform_start_120`。主干、可学习MLDC、初始化顺序、ID+Triplet、SGD参数组及原学习率调度均保留，仅加入训练时的单空间跨摄像头CNL。无需修改原实验包或已完成的固定等权CNL包。

- 后四层权重由4个可学习logits经temperature=0.9的Softmax得到；初始各0.25，训练中自由学习。
- 从相同SSL预训练 `pretrain_model/vit_base_ics_cfs_lup.pth` 重新训练120轮，seed=1234；不接着历史120轮检查点微调。
- 训练目标：原ID + 原soft-margin Triplet + 0.1 × CNL。CNL输入为L2(BN(实际可学习融合CLS))。
- CNL与已完成的等权实验相同：gamma=0.9，temperature=0.05，memory momentum=0.2；只选同身份跨摄像头近邻为正例，异身份为负例。原图自身被排除，未选中同身份不作为负例。
- 记忆库按训练图像索引保存，仅在优化器成功更新后更新；推理仍输出一个768维描述子。
- 保留原batch64、PK4、训练增强、120轮、每10轮验证及原可视化设置；BN后检索、无重排序。

在TransReID项目根目录运行：

```bash
python mldc_learnable_global_cnl/train.py
```

等价的显式配置：

```bash
python mldc_learnable_global_cnl/train.py --config_file mldc_learnable_global_cnl/variants/deep_four_uniform_cross_camera_cnl/config.yml
```

独立结果目录：

`/root/log/transreid/night600/vit_base_ics_mldc_learnable_cnl_20261020/deep4_uniform_start_cnl01_120`

另提供同配置CNL=0文件 `vit_base_ics_mldc_deep4_uniform_no_cnl.yml`，输出到独立的 `deep4_uniform_start_no_cnl_120`；未自动启动这组额外训练。

记录训练ID/Triplet/CNL、有效anchor比例、正对覆盖率、AMP跳步、逐轮四层权重、完整检索指标与逐查询结果。检查点保留原模型加载格式并增加CNL记忆库。没有实现完整优化器/采样器恢复，入口为从预训练开始的新训练。

检查：`python -m mldc_learnable_global_cnl.verify`。该检查比较原基线初始化与学习率轨迹，并检验CNL到融合权重的梯度以及检查点往返；不执行优化器更新，不产生实验指标。
