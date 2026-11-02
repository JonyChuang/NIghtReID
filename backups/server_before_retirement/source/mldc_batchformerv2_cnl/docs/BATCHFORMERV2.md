# BatchFormerV2 到 MLDC+CNL 的迁移

参考论文：[BatchFormerV2: Exploring Sample Relationships for Dense Representation Learning](https://arxiv.org/abs/2204.01254)，2022年技术报告。
参考源码：[作者仓库 batchformer-v2/deit_share](https://github.com/zhihou7/BatchFormer/tree/main/batchformer-v2/deit_share)，其中 `bt.py` 定义逐token的batch注意力，`main.py` 支持按原ViT块索引插入。这里迁移V2的token交互与共享双路机制，插入位置按现有CLS9—12读出选择，属于ReID适配。

## 独立实验

包名 `mldc_batchformerv2_cnl`，原包 `mldc_convstream_texture_cnl` 及其他实验保持原样。包内引用均使用新命名空间，输出使用新目录。入口不提供断点续训；从相同SSL权重初始化，训练120轮、seed1234、PK64/K4，CNL的gamma0.9、权重0.1与 `all_nonself` 阈值参考集合不变。

## 插入位置与数据流

1. 原stem与ViT第1—8块执行一次，得到 `[B,129,768]`，B=64；129包含CLS和128个patch。
2. 一层 `TransformerEncoderLayer(768,12,768,0.5)` 读取该张量，使用ReLU、post-norm；不增加batch位置编码或身份/相机注意力掩码。**保持 `batch_first=False` 并直接输入 `[B,N,C]`**：B是注意力序列长度，N是129个独立token位置。每个位置在64张训练图之间交互，不把所有空间位置拼成一条注意力序列。
3. 拼接原token和增强token为 `[2B,129,768]`，输入同一组原第9—12块。分离两路各自的CLS9—12，再使用同一个可学习MLDC读出。早期CLS快照保持原值；MLDC仍只选9—12层。
4. 两路 `[B,768]` 拼接后共用一次BNNeck、Dropout和身份分类器。BN运行统计来自这2B行，与当前BatchFormer实验保持一致。主干和头部不存在独立教师副本，增强分支不detach。

模块只增加3,546,624个训练参数，从第1轮参与训练。`BATCHFORMERV2.INSERT_AFTER=8` 表示第8个块之后（1-based）；实现允许1—8之间调整以确保四个读出层均经过共享后续主干。该编号不直接等同于作者命令行的0-based索引。头数12与本文所用ViT-B主干一致；dropout0.5与作者默认一致。

## 目标函数与优化

沿用服务器最新BatchFormer实验恢复原路CE后的目标，避免再次削弱原单图监督：

`L = CE_original + 0.5*CE_augmented + Triplet_original + 0.1*CNL_original`

两路CE各自对原B个标签求平均，再按 `SINGLE_CE_WEIGHT=1` 和 `AUGMENTED_CE_WEIGHT=0.5` 相加。Triplet使用原路MLDC特征。CNL及其记忆更新只接收原路BN后L2描述子和原B个样本索引，AMP跳步时不更新记忆。增强路CE可经共享MLDC、ViT后四块、BatchFormerV2和前缀主干回传梯度。没有新增KD、IP、原型或多图一致性损失；PAD、PEM、NFC和NFC_CENTER均关闭。

`configs/batchformerv2_original_ce_nightreid.yml` 可选用0.5/0.5的平均CE对照，结构、优化器和数据协议相同，输出进入独立的 `mean_ce/from_ssl_seed1234_120` 子目录；默认训练命令使用1/0.5配置。

保留原SGD学习率、weight decay、bias倍率、头部倍率及20轮cosine学习率预热。模块不设关闭期，学习率预热不等于模块关闭。新层使用原主干参数组规则；没有照搬论文ImageNet/DeiT的AdamW与大weight decay配方。这是控制原实验变量的模块适配，不是分类训练配方的完整复现。

## 训练与评估

```bash
cd /root/TransReID-SSL-main/transreid_pytorch
/root/miniconda3/envs/QREID/bin/python -u -m mldc_batchformerv2_cnl.train_batchformerv2 --config_file mldc_batchformerv2_cnl/configs/batchformerv2_nightreid.yml
```

输出：`/root/log/transreid/nightreid/mldc_batchformerv2_cnl/from_ssl_seed1234_120`，要求空目录。每10轮独立单图评估，保存最佳及40/80/120轮检查点。

`training.csv` 保留原路与增强路CE、平均CE、实际加权CE、两路系数及增强路准确率等 `batchformerv2_*` 字段；`global_id` 与 `batchformerv2_weighted_ce` 是实际加权分类目标，`batchformerv2_mean_ce` 仅是算术平均诊断，`global_accuracy` 为原路准确率。`run_metadata.json` 记录结构、位置、参数量、初始化和目标函数。训练分类指标用于观察分支训练状态，检索增益以独立单图验证集mAP/Rank-1为准。

训练结束后可评估最佳检查点：

```bash
/root/miniconda3/envs/QREID/bin/python -u -m mldc_batchformerv2_cnl.test --config_file mldc_batchformerv2_cnl/configs/batchformerv2_nightreid.yml TEST.WEIGHT /root/log/transreid/nightreid/mldc_batchformerv2_cnl/from_ssl_seed1234_120/transformer_best.pth
```

评估保留配置开关以严格加载V2检查点，`eval()` 时不向主干传入BatchFormerV2，完整绕过跨图像注意力、增强路与训练记忆。输出仍为原768维单图特征，batch=1也走同一路径。检查点保留V2参数用于加载，推理不执行该层。

本次仅执行源码AST解析、配置读取、引用与差异静态核对、上传后的文件字节核对。没有导入实验模型、执行前向/反向或启动训练；运行行为与指标需正式训练验证。
