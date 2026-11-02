# TransReID-SSL + CNL（原版配置）

独立实验包：`/root/TransReID-SSL-main/transreid_pytorch/transreid_ssl_cnl`。
仅准备、同步代码并进行 CPU 集成验证；未启动训练、未设置排队任务。

## 运行训练

在原实验服务器执行：

```bash
cd /root/TransReID-SSL-main/transreid_pytorch
/root/miniconda3/envs/QREID/bin/python -u transreid_ssl_cnl/train.py --config_file transreid_ssl_cnl/configs/night600.yml
```

默认使用 GPU 0。输出独立保存在：

```text
/root/log/transreid/night600/transreid_ssl_cnl/global_cnl01_original_config_120
```

## 保留的原版设置

| 项目 | 配置 |
| --- | --- |
| 骨干 | 原版 ViT-Base TransReID-SSL，STEM_CONV=True，stride=16 |
| 预训练 | pretrain_model/vit_base_ics_cfs_lup.pth，HW_RATIO=2 |
| 全局特征 | 最后一个 Transformer block → 原版最终 LayerNorm → CLS，768 维 |
| ID / Triplet | BNNeck 后分类、BNNeck 前 Triplet；soft Triplet；均为权重 1 |
| 检索 | NECK_FEAT=before，FEAT_NORM=yes，不使用重排序 |
| 数据 | Night600；256×128；原版翻转、padding、随机擦除、归一化 |
| 采样 | softmax_triplet，每个身份 4 张；8 个数据加载进程 |
| 训练 | 120 轮，batch=64，seed=1234 |
| 优化器 | 原版 SGD，基础学习率 0.0004，momentum=0.9，bias LR factor=2 |
| 权重衰减 | weight / bias 都是 0.0001 |
| 调度 | 原版 solver.scheduler_factory.create_scheduler；warmup=20；每轮末 step(epoch) |
| 分类头学习率 | 沿用原版；没有 MLDC 的额外 5 倍倍率 |
| 评估 | 每 10 轮评估，评估 batch=256；第 120 轮保存最终模型 |

骨干取自工作区内加入多层融合之前的原版快照。已与服务器当前模型关闭多层融合后的全局分支逐项核对：同种子初始化参数完全一致，同输入检索输出最大绝对差为 0。

## 新增的 CNL

总损失：`ID + Triplet + 0.1 × CNL`。

沿用当前 GlobalCNL 算法：同身份、不同摄像头，且相似度超过 `0.9 × 当前最大候选相似度` 的图像作为正邻居；不同身份图像为负样本；温度 0.05；记忆动量 0.2。

为了遵循原版的 BN 前检索配置，CNL 查询和记忆库特征使用 `L2(原版全局特征)`。这是末层最终 LayerNorm 后、BNNeck 前的特征。它与可学习 MLDC+CNL 版本使用的 BN 后融合特征不同。

记忆库使用训练集 14462 张图像的稳定索引，只记录训练图像特征。当前 batch 的特征替换相同索引的旧候选；仅在 AMP 优化器更新成功后写回记忆库。验证集和测试集不会写入。

## 输出与复测

保存 `training.csv`、`evaluation.csv`、每个 query 的评估记录、配置与数据记录、最佳 mAP 模型 `transformer_best.pth` 和最终模型 `transformer_120.pth`。检查点包含 CNL 记忆库；此入口从 SSL 预训练重新训练，不提供断点续训。

复测最终模型：

```bash
cd /root/TransReID-SSL-main/transreid_pytorch
/root/miniconda3/envs/QREID/bin/python -u transreid_ssl_cnl/test.py TEST.WEIGHT /root/log/transreid/night600/transreid_ssl_cnl/global_cnl01_original_config_120/transformer_120.pth
```

如需额外做完全同配置的无 CNL 对照，可手动执行以下命令，结果写入另一个目录：

```bash
/root/miniconda3/envs/QREID/bin/python -u transreid_ssl_cnl/train.py SOLVER.CNL_WEIGHT 0.0 OUTPUT_DIR /root/log/transreid/night600/transreid_ssl_cnl/global_no_cnl_original_config_120
```

## 已完成的验证

只在 CPU 上验证，无优化器参数更新：原版参数与特征一致、ID+Triplet 公式一致、CNL 有效邻居及骨干梯度、记忆更新、检查点读写、真实数据加载和评估入口、原版优化器参数组与 121 个调度阶段。详见 `verification_cpu.json`。尚未进行本方案的 GPU 训练或指标评测。
