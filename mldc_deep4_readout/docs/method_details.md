# MLDC 后四层等权初始化独立实验

本副本默认已改为第9—12层等权初始化：四个可学习logits从0开始，Softmax后每层25%，随后正常学习。第1—8层CLS不参与融合，完整12层ViT保留。

目录沿用 `/root/TransReID-SSL-main/transreid_pytorch/a3pfn_mldc_deep4_random`；目录名保留历史名称，默认实验现为uniform。

训练、测试入口默认配置：`mldc_deep4_readout/variants/deep_four_uniform_learnable/config.yml`。

主干、损失、学习率、120轮设置及BN后检索保持原设置。初始化保留原随机数消耗后将logits清零，以保持主干和分类头随机状态一致。

结果目录：`/root/log/transreid/night600/vit_base_ics_a3pfn_mldc/deep4_uniform_start_120`。

旧随机配置保留，可显式指定；默认入口不会使用它。本次只修改代码和配置，没有启动训练。
