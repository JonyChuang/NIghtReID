# MLDC 后四层随机初始化独立实验

代码副本：`/root/TransReID-SSL-main/transreid_pytorch/a3pfn_mldc_deep4_random`

仅第9—12层 CLS 参与可学习加权，四个随机 logits 通过温度0.9的 Softmax 得到权重。前8层 CLS 不参与融合；完整12层 ViT 保留。初始化保留原模型的12次随机数采样，再取后四项，维持主干及分类头的随机数状态。

seed=1234 的四层初始权重依次为5.3617%、81.1506%、8.0791%、5.4087%。训练使用原 ID＋Triplet 监督，主检索采用 BN 后特征并统一 L2，输出768维；不叠加末层残差。

代码副本拥有自己的模型、骨干、损失、处理器和学习率调度封装。根项目中的数据加载、通用优化器、评估工具及原 SSL 权重继续共享。原 mldc_cls_readout 代码不改动。

120轮、seed=1234、batch=64、SGD基础学习率0.0004、融合学习率倍率1、分类头倍率5、20轮warmup，数据处理沿用原设置。

结果目录：`/root/log/transreid/night600/vit_base_ics_a3pfn_mldc/deep4_random_learnable_120`

用户自行启动训练：

```bash
cd /root/TransReID-SSL-main/transreid_pytorch
/root/miniconda3/envs/QREID/bin/python mldc_deep4_readout/train.py
```

显式指定配置时使用 `--config_file mldc_deep4_readout/variants/deep_four_random_learnable/config.yml`。

默认配置和模型注册默认选择均为第9—12层。训练保存12列权重，前8列为0。检查点记录选择的层；评估时指定该副本的 test.py 及 TEST.WEIGHT。

部署验证只使用CPU前向、反向与临时检查点，不执行优化器更新，不启动训练。
