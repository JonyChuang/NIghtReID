# NightReID 实验导航

实验包按方法和模块命名；`variants/` 按配置区别命名；历史源码统一放在 `history/`。日期不参与目录和配置命名。

## 实验包

| 目录 | 研究模块 | 配置数 |
|---|---|---:|
| [mldc_cls_readout](mldc_cls_readout/README.md) | 同一SSL初始化ViT，收集原始CLS，按配置执行多层加权或末层对照；使用ID和Triplet联合训练，再提取BN前/后描述子评估。 | 8 |
| [mldc_deep4_readout](mldc_deep4_readout/README.md) | 从SSL初始化完整ViT，仅融合第9—12层原始CLS；随机或均匀初始化层权重，随后与骨干共同学习，保存层权重与检索指标。 | 2 |
| [mldc_learnable_global_cnl](mldc_learnable_global_cnl/README.md) | 由相同SSL初始化ViT；第9—12层CLS以均匀起步的可学习Softmax权重融合，联合优化ID、Triplet和按配置开关的跨相机CNL。记忆库在成功更新优化器后更新；检索使用768维BN后全局描述子，不恢复优化器或采样器状态。 | 4 |
| [mldc_last_cls_dual_supervision](mldc_last_cls_dual_supervision/README.md) | 构造后四层固定均值和末层LayerNorm后CLS两个分支；以0.7/0.3监督；推理时两路分别归一化后等能量拼接，得到1536维描述子。 | 1 |
| [mldc_feature_enhancement](mldc_feature_enhancement/README.md) | 按独立入口选择Patch补偿、PFE、PFE范数控制或纯全局CNL；统一从SSL初始化并使用固定后四层均值。先比较对应关闭模块对照，再解释增强模块或损失的作用。 | 10 |
| [mldc_boq_local_readout](mldc_boq_local_readout/README.md) | 全局使用可学习后四层MLDC；从第4/6/8层patch提取BoQ多查询局部描述子，形成全局残差，并对局部读出提供辅助监督。所有活动模块联合训练。 | 1 |
| [mldc_locat_attention](mldc_locat_attention/README.md) | 按配置控制GAug、读出LayerNorm和PRR；融合完整多层token后读取CLS，使用相同ID/Triplet和评估设置比较各组件。 | 6 |
| [mldc_tar_refinement](mldc_tar_refinement/README.md) | MLDC融合原始CLS；TAR在末层patch候选中读取并细化末层CLS；最终使用g+c_refined，训练时保留全局、细化和残差三处监督。 | 1 |
| [mldc_lpam_fft](mldc_lpam_fft/README.md) | 全局始终取第9—12层原始CLS均值；LPAM读取第3/6/9/12层并执行FFT路径；分别BN后直接拼接四路描述子进行检索。 | 1 |
| [mldc_mufasa_slots](mldc_mufasa_slots/README.md) | 从SSL开始，读取后四层patch执行Slot Attention、跨层匹配和融合；MLDC引导整体slot残差，门控融合，并加入残差度量及patch重建辅助目标。 | 1 |
| [mldc_patch_grm](mldc_patch_grm/README.md) | 加载已训练deep4初始化，固定全局骨干、融合和原BN；前5轮直接训练局部读出，随后启用GRM残差及最终特征监督，评估冻结全局、局部和融合读出。 | 1 |
| [ffvt_token_fusion](ffvt_token_fusion/README.md) | 原始ViT在第3/6/9/11层按MAWS各选12个patch，组成融合序列；原全局末层和独立融合末层分别监督，推理拼接两路BN后特征。 | 1 |

## 根目录的共享模型实验

[根目录实验配置](experiment_variants/README.md) 包含多层Patch-CLS、GLTrans和旧QNorm配置。它们共用根目录model/processor，不复制公共代码。

## 使用方式

从工程根目录执行各配置README中的命令。需要CUDA的训练入口不会由整理脚本启动。

环境、数据准备与其他机器的路径调整见 [环境与运行入口](SETUP.md)。

## 历史与结果

- [原名与新名映射](PATH_MAPPING.md)
- [参考文献和模块来源](REFERENCES.md)
- [实验索引](experiment_index.csv)

数据、预训练和已有模型结果保留原位置。新配置的语义输出路径在已有结果存在时链接到原目录。历史检查点内部的架构标识保留，以支持严格加载。

## 旧实验的使用状态

`history/` 保存修改前快照，其中部分只含改动文件；配置README会说明需要恢复的入口。旧QNorm/QCAT/SFM方案共用的某些backbone依赖在整理前已缺失，不能仅凭存在配置认定可直接训练。当前CNL默认不包含CNN分支；历史CPPI配置不适用于当前CNL训练入口。

## 备份与恢复

服务器原始源码归档：`/root/experiment_organization_backup/original_source.tar.gz`；本地也保存同一份归档。`PATH_MAPPING.md` 可用于查找旧路径。备份包含源码、配置和说明，不包含数据集、预训练或大模型结果；这些文件保留在服务器原位置，包内初始化权重随代码包整体迁移。
