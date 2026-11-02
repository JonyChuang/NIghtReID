# NightReID：MLDC+CNL 与 BatchFormer

默认 train 入口恢复原 MLDC+CNL 基线。BatchFormer 使用 train_batchformer 独立入口及配置，在同一实验包内运行。此前多图融合代码、入口和配置已移出活动源码；历史结果保留。

两者均使用原 CLS9–12 可学习 MLDC、BNNeck、ID、Triplet 与 0.1 CNL（gamma0.9，all_nonself），从原 SSL 训练 120 轮。NFC、PAD、PEM 均关闭。

BatchFormer 仅在训练中沿 batch 维学习样本关系，共享分类器；测试自动绕过该模块，保留独立单图 768 维检索。

当前 `train_batchformer` 默认配置已改为恢复原路CE的方案：`CE_single + 0.5*CE_batchformer + Triplet + 0.1*CNL`。两路BN处理保持原版；新结果目录为 `batchformer_restore_single_ce/from_ssl_seed1234_120`。原版0.5/0.5系数可用 `configs/batchformer_original_ce_nightreid.yml` 显式复现。

完整结构、损失、配置和运行命令见 [docs/BATCHFORMER.md](docs/BATCHFORMER.md)。

本次仅修改源码、配置并核对损失计算，没有启动训练或操作已有训练进程。
