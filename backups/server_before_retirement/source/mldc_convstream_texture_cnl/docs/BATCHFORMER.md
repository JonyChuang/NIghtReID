# MLDC+CNL + BatchFormer（CVPR 2022）

本次在原实验包内移植 BatchFormerV1，并移除此前的多图分组、B8 token 拼接、共享后四层、综合 ID 和 IP 路径。没有建立新的模型副本。

原文：https://openaccess.thecvf.com/content/CVPR2022/html/Hou_BatchFormer_Learning_To_Explore_Sample_Relationships_for_Robust_Representation_Learning_CVPR_2022_paper.html

作者实现：https://github.com/zhihou7/BatchFormer （README 的 PyTorch Code）。

## 结构

单图经过原 ViT，取 CLS9–12 做可学习 MLDC，得到 `[N,768]`。训练时保留原特征，并将同一特征输入一层 `TransformerEncoderLayer(768,4,768,0.5)`：ReLU、post-norm、无位置编码，输入形状为 `[N,1,768]`。N 是注意力序列长度，整批样本参与关系建模；不使用身份或相机筛选注意力。

原特征和增强特征沿第 0 维拼接，共用一次原 BNNeck、Dropout 和同一个身份分类器。两路分类监督使用同一组标签。当前默认恢复原路CE系数为1，增强路CE系数保持0.5，分别由 `BATCHFORMER.SINGLE_CE_WEIGHT` 和 `AUGMENTED_CE_WEIGHT` 控制。

`L = CE_single + 0.5 * CE_batchformer + Triplet_single + 0.1 * CNL_single`

原版为 `0.5*(CE_single+CE_batchformer)+Triplet_single+0.1*CNL_single`；显式复现配置是 `configs/batchformer_original_ce_nightreid.yml`。代码默认权重仍回退到0.5/0.5，以保持缺少新增字段的旧配置和旧检查点兼容。恢复版仅改变原路CE系数；BN仍使用原两路拼接统计，不使用冻结验证中的重校准BN。

Triplet 仍使用原 MLDC 特征。CNL 使用原路 BN 后归一化特征，记忆库仅写入这 N 行原路描述子，AMP 跳步时不更新记忆库。增强分支不 detach，分类梯度通过 BatchFormer 回传至 MLDC 和 ViT。

BatchFormer 从第 1 轮启用；保留原优化器的 20 轮学习率预热。新增参数使用主干学习率及已有 bias 倍率；原 BN/分类器与 MLDC 学习率倍率不变。单层新增 3,546,624 个参数，只在训练时执行。

BNNeck 共享归一化与保留原单图 Triplet/CNL 是 ReID 适配，属于模块迁移，不是论文分类任务的完整复现。BN 训练统计取两路拼接特征，这与旧多图方案的单路统计不同。

## 基线和 BatchFormer

### Night600：原版CE的两份独立配置

两份配置均使用0.5原路CE+0.5增强路CE，Triplet和0.1 CNL保持原样：

- `configs/batchformer_original_ce_night600.yml`：后四层权重可学习；仅从各25%初始化。
- `configs/batchformer_fixed_equal_night600.yml`：后四层始终固定各25%，融合权重不参与优化。

其余参数分别与对应NightReID配置一致：原SSL初始化、120轮、seed1234、PK64/K4、学习率0.0004、20轮学习率预热、BatchFormer一层四头/dropout0.5、SIE关闭。只切换数据集及独立输出目录；可学习版显式写明 `MLDC_WEIGHT_MODE: learnable`。

Night600使用 `DATASETS.NAMES: night600`、`ROOT_DIR: data`，对应项目下 `data/night600`。训练入口按数据集分别校验划分，Night600为14462张训练图、300个训练身份、2180 Query、14351 Gallery；分类器根据数据加载器的300类创建，CNL记忆根据实际训练图片创建。

```bash
cd /root/TransReID-SSL-main/transreid_pytorch
# 可学习融合版
/root/miniconda3/envs/QREID/bin/python -u -m mldc_convstream_texture_cnl.train_batchformer --config_file mldc_convstream_texture_cnl/configs/batchformer_original_ce_night600.yml
# 固定各25%版
/root/miniconda3/envs/QREID/bin/python -u -m mldc_convstream_texture_cnl.train_batchformer --config_file mldc_convstream_texture_cnl/configs/batchformer_fixed_equal_night600.yml
```

结果分别位于 `/root/log/transreid/night600/mldc_convstream_texture_cnl/batchformer/from_ssl_seed1234_120` 和 `/root/log/transreid/night600/mldc_convstream_texture_cnl/batchformer_fixed_equal/from_ssl_seed1234_120`。NightReID配置和默认入口不变；本次仅配置，不启动任何一版训练。

### 原版CE与后四层固定等权对照

`configs/batchformer_fixed_equal_nightreid.yml` 在原版0.5原路CE+0.5增强路CE上，将CLS9—12的融合固定为算术平均，每层严格25%。`MODEL.MLDC_WEIGHT_MODE: fixed_uniform` 直接使用均值读出，融合logits不参与优化；并非只在初始化时置为25%。两路分类前的融合特征、Triplet、CNL及正式测试都使用这一个等权读出，其余BN和BatchFormer设置不变。

```bash
cd /root/TransReID-SSL-main/transreid_pytorch
/root/miniconda3/envs/QREID/bin/python -u -m mldc_convstream_texture_cnl.train_batchformer --config_file mldc_convstream_texture_cnl/configs/batchformer_fixed_equal_nightreid.yml
```

从原SSL权重训练120轮、seed1234、PK64/K4；结果目录为 `batchformer_fixed_equal/from_ssl_seed1234_120`。原可学习融合保持 `MLDC_WEIGHT_MODE: learnable`（旧配置缺省即为learnable），仍可通过原版CE配置复现。固定等权版检查点记录单独variant，测试应使用同一固定等权配置；不会把旧版已学到的融合权重直接覆盖成25%来冒充固定等权训练结果。本次仅准备代码和配置，不启动训练。

纯基线：

```bash
cd /root/TransReID-SSL-main/transreid_pytorch
/root/miniconda3/envs/QREID/bin/python -u -m mldc_convstream_texture_cnl.train
```

BatchFormer：

```bash
cd /root/TransReID-SSL-main/transreid_pytorch
/root/miniconda3/envs/QREID/bin/python -u -m mldc_convstream_texture_cnl.train_batchformer
```

默认配置分别为 `configs/mldc_cnl_baseline_nightreid.yml` 和 `configs/batchformer_nightreid.yml`。BatchFormer默认配置现在使用恢复原路CE的方案；两者均从原 SSL 权重初始化，120 轮、seed1234、PK64/K4、原 CNL gamma0.9、`all_nonself` 阈值参考集合。NFC、PAD、PEM 均关闭。

结果目录分别是原日志根目录下 `baseline_mldc_cnl/from_ssl_seed1234_120` 和 `batchformer_restore_single_ce/from_ssl_seed1234_120`。原版复现配置另用 `batchformer_original_ce/from_ssl_seed1234_120`，此前已完成的 `batchformer/from_ssl_seed1234_120` 保留。入口要求空目录；保存最佳及 40/80/120 轮检查点，不提供断点续训入口。

测试使用原 `test` 入口，指定 BatchFormer 配置及相应检查点。配置中的开关保留为 true 以加载训练参数，`eval()` 自动绕过 BatchFormer，不做测试集样本交互。检索仍是单图 768 维，支持 batch=1。检查点保留 BatchFormer 权重用于严格加载，但推理不执行该层。

加载此前52.57 mAP / 67.93 Rank-1的原版最佳检查点时，请使用 `configs/batchformer_original_ce_nightreid.yml`；新的恢复CE配置对应新训练方案的检查点，两者训练元数据按系数区分。

`training.csv` 记录两路CE、算术平均CE、实际加权CE、两路系数和增强路准确率。`batchformer_mean_ce` 仅是未加权诊断，`global_id`/`batchformer_weighted_ce` 才是实际加权分类目标；恢复版为原路CE加0.5倍增强路CE。`global_accuracy` 是原路准确率。正式评估始终使用独立单图特征。

本次仅修改代码与配置，不启动训练、不操作已有训练进程。恢复损失的方案需要以后从SSL重新训练，不能用原版110轮检查点或BN重校准结果替代验证。
