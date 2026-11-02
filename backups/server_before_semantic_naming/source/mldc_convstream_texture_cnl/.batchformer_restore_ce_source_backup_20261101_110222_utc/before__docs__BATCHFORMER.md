# MLDC+CNL + BatchFormer（CVPR 2022）

本次在原实验包内移植 BatchFormerV1，并移除此前的多图分组、B8 token 拼接、共享后四层、综合 ID 和 IP 路径。没有建立新的模型副本。

原文：https://openaccess.thecvf.com/content/CVPR2022/html/Hou_BatchFormer_Learning_To_Explore_Sample_Relationships_for_Robust_Representation_Learning_CVPR_2022_paper.html

作者实现：https://github.com/zhihou7/BatchFormer （README 的 PyTorch Code）。

## 结构

单图经过原 ViT，取 CLS9–12 做可学习 MLDC，得到 `[N,768]`。训练时保留原特征，并将同一特征输入一层 `TransformerEncoderLayer(768,4,768,0.5)`：ReLU、post-norm、无位置编码，输入形状为 `[N,1,768]`。N 是注意力序列长度，整批样本参与关系建模；不使用身份或相机筛选注意力。

原特征和增强特征沿第 0 维拼接，共用一次原 BNNeck、Dropout 和同一个身份分类器。两路分类监督使用同一组标签，等价于标签复制一次后对 2N 行交叉熵取平均。

`L = 0.5 * (CE_single + CE_batchformer) + Triplet_single + 0.1 * CNL_single`

Triplet 仍使用原 MLDC 特征。CNL 使用原路 BN 后归一化特征，记忆库仅写入这 N 行原路描述子，AMP 跳步时不更新记忆库。增强分支不 detach，分类梯度通过 BatchFormer 回传至 MLDC 和 ViT。

BatchFormer 从第 1 轮启用；保留原优化器的 20 轮学习率预热。新增参数使用主干学习率及已有 bias 倍率；原 BN/分类器与 MLDC 学习率倍率不变。单层新增 3,546,624 个参数，只在训练时执行。

BNNeck 共享归一化与保留原单图 Triplet/CNL 是 ReID 适配，属于模块迁移，不是论文分类任务的完整复现。BN 训练统计取两路拼接特征，这与旧多图方案的单路统计不同。

## 基线和 BatchFormer

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

默认配置分别为 `configs/mldc_cnl_baseline_nightreid.yml` 和 `configs/batchformer_nightreid.yml`，仅 BatchFormer 开关和结果目录不同。两者均从原 SSL 权重初始化，120 轮、seed1234、PK64/K4、原 CNL gamma0.9、`all_nonself` 阈值参考集合。NFC、PAD、PEM 均关闭。

结果目录分别是原日志根目录下 `baseline_mldc_cnl/from_ssl_seed1234_120` 和 `batchformer/from_ssl_seed1234_120`。入口要求空目录；保存最佳及 40/80/120 轮检查点，不提供断点续训入口。

测试使用原 `test` 入口，指定 BatchFormer 配置及相应检查点。配置中的开关保留为 true 以加载训练参数，`eval()` 自动绕过 BatchFormer，不做测试集样本交互。检索仍是单图 768 维，支持 batch=1。检查点保留 BatchFormer 权重用于严格加载，但推理不执行该层。

`training.csv` 记录两路 CE、平均 CE 和增强路分类准确率；`global_id` 是两路平均 CE，`global_accuracy` 是原路准确率。正式评估始终使用独立单图特征。

本次迁移不启动训练，不操作已有训练进程；检索收益待正式训练验证。
