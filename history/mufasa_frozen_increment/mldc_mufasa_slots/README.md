# Frozen MLDC + MUFASA increment

本实验从独立 MLDC 的第 120 轮检查点开始。只训练新增模块，不启动其他版本的实验。

## 开始训练

在 GPU 可用的服务器终端中运行：

```bash
cd /root/TransReID-SSL-main/transreid_pytorch
/root/miniconda3/envs/QREID/bin/python -u mldc_mufasa_slots/train.py --config_file mldc_mufasa_slots/variants/slot_residual_joint_bnneck/config.yml
```

如果需要断开终端后继续运行，可使用下面这条替代上面的启动命令；不要同时运行两条：

```bash
nohup /root/miniconda3/envs/QREID/bin/python -u mldc_mufasa_slots/train.py --config_file mldc_mufasa_slots/variants/slot_residual_joint_bnneck/config.yml > /root/log/transreid/night600/mufasa_frozen_increment_120.console.log 2>&1 &
```

查看日志：

```bash
tail -f /root/log/transreid/night600/mufasa_frozen_increment_120.console.log
```

程序首先完整评估原 MLDC，在其复现 mAP=10.2140669%、Rank-1=21.1467892%、Rank-5=37.7981663%、Rank-10=47.1100926% 后才开始参数更新。初始化模块的 epoch-0 检索指标仅用于诊断，不代表训练结果。若原指标不符，程序停止，不自行改动基线或评估协议。

默认输出目录：

```text
/root/log/transreid/night600/vit_base_ics_a3pfn_mldc_mufasa/frozen_increment_120
```

## 固定的首轮设置

- 原 ViT、MLDC 层权重、原 BN 和原分类器全部冻结，保持 eval 模式；每次保存检查点时核对全部原参数及 buffer。
- 第 9、10、11、12 层原始 post-block patch，每层独立 SA；4 slots、256 维、3 次迭代、SA MLP 隐层 1024。
- 确定性的 embedding 初始化，采用参考代码默认的 bi-level 梯度截断。
- 以二值空间掩码 IoU 做最优一一匹配；K=4 时枚举 24 个置换，解与 Hungarian 目标相同，不依赖 SciPy，支持 16×8 网格。
- 对齐后的相邻层 slots 取平均、拼接，经 M-Fusion 投影为 4×256；用 MLDC 引导的集合读出得到 768 维补充特征。
- 最终特征：`f = L2(L2(original_BN(g)) + 0.1 * L2(r))`，无新增最终 BN、无独立局部分类头。
- 最终 f 的 ID + 原 soft-margin batch-hard triplet + 0.1×重建 MSE。重建目标是冻结第 12 层的原始 patch，停止梯度。
- 训练期使用位置广播 MLP 解码器，4 个 2048 维隐藏层，预测每-slot patch 特征及混合掩码；推理不执行解码器。它是本项目采用的解码器配置，不等同于论文主实验的自回归 Transformer。
- Adam，学习率 0.0004，权重衰减 0，10 轮预热，随后余弦下降，120 轮。
- batch 64=16 个身份×4 张图，原增强流程，seed 1234。
- FP32 前向与训练，不使用混合精度；原 MLDC 使用原后端设置。
- 每 10 轮评估，每 20 轮保存和可视化；最后一轮始终评估与保存。主结果固定使用第 120 轮，不保存测试集最优检查点。

来源：[MUFASA 官方代码固定版本](https://github.com/visinf/mufasa/tree/cb4236e7fb37ebd157b7d6d73447565dd2e6f8f0)。独立 SA、空间匹配和 M-Fusion 来自该方法；MLDC 条件读出、单一归一化残差描述符与 ReID 监督是本项目迁移设计。

官方默认路径中额外的 SA 外层输入 MLP 不会被调用；本版同样只保留 SA 自身的输入 LayerNorm，没有额外加入那个 MLP。可视化使用确定性的均匀掩码融合，不引入无监督信号的可学习 mask-fusion 权重。不同图像的 slot 颜色不保证对应同一人体部位。

## 保存结果

| 文件 | 含义 |
|---|---|
| `train_log.txt`、`training.csv` | 总损失、ID、triplet、重建损失、准确率和学习率 |
| `evaluation.csv`、`retrieval_curves.png` | 同次评估的冻结基线、最终特征、补充特征单独检索指标 |
| `diagnostics.csv` | slot 占用、空 slot、相似度、匹配 IoU、读出权重、残差方向统计 |
| `evaluations/epoch_XXX/queries_*.csv` | 各描述符逐 query 的 AP、Rank-1、首个正样本排名 |
| `evaluations/epoch_XXX/query_changes.csv` | 相对冻结 MLDC 的 AP 改变、修正与退化样本 |
| `visualizations/epoch_XXX/` | 与原 MLDC 相同的固定跨镜头样本；PNG 图及 NPZ 原始分配 |
| `increment_20.pth` … `increment_120.pth` | 新模块参数、优化器和随机状态；复用原冻结检查点 |
| `late_summary.json` | 最后三次评估的均值，作为后期稳定性诊断 |

保存的增量检查点依赖原文件 `/root/log/transreid/night600/vit_base_ics_a3pfn_mldc/transformer_120.pth`，请保留该文件。程序不会覆盖它。

## 中断恢复与最终评估

例如从最新的第 40 轮恢复，在项目根目录运行：

```bash
/root/miniconda3/envs/QREID/bin/python -u mldc_mufasa_slots/train.py --resume /root/log/transreid/night600/vit_base_ics_a3pfn_mldc_mufasa/frozen_increment_120/increment_40.pth
```

恢复沿用检查点保存的完整配置、优化器和随机状态。在原输出目录恢复；检查点之后的未保存 CSV 记录会先备份，再从曲线中移除，以便重做这些轮次。完整检查点每 20 轮一次，故中断最多需要重做 19 轮。

训练完后只评估最终检查点：

```bash
/root/miniconda3/envs/QREID/bin/python -u mldc_mufasa_slots/train.py --evaluate /root/log/transreid/night600/vit_base_ics_a3pfn_mldc_mufasa/frozen_increment_120/increment_120.pth
```

评估结果写入 `checkpoint_analysis/epoch_120/`。

## 后续对照入口

仅当首轮获得稳定收益后，再分别运行这些对照，每组使用独立输出目录。一次只运行一个实验：

```bash
# 四个模块都读取第 10 层；第 12 层重建目标保持不变。
/root/miniconda3/envs/QREID/bin/python -u mldc_mufasa_slots/train.py MUFASA.LAYERS '[10,10,10,10]' OUTPUT_DIR /root/log/transreid/night600/vit_base_ics_a3pfn_mldc_mufasa/p10_four_branches_120

# 移除跨层匹配。
/root/miniconda3/envs/QREID/bin/python -u mldc_mufasa_slots/train.py MUFASA.MATCH False OUTPUT_DIR /root/log/transreid/night600/vit_base_ics_a3pfn_mldc_mufasa/no_matching_120

# 对齐后四层平均，重复到相同投影输入维度，保持投影参数量一致。
/root/miniconda3/envs/QREID/bin/python -u mldc_mufasa_slots/train.py MUFASA.FUSION mean OUTPUT_DIR /root/log/transreid/night600/vit_base_ics_a3pfn_mldc_mufasa/mean_fusion_120
```

首轮固定最终检查点与超参数。对照也保持同样的重建目标、读出、监督和训练长度。单次早期峰值、重建损失下降或匹配 IoU 上升，均不能代替最终检索收益。

## 实现检查

```bash
/root/miniconda3/envs/QREID/bin/python mldc_mufasa_slots/verify.py --device cpu
```

检查包含真实 PK batch 的一次临时更新、原 MLDC 参数及 buffer 不变、λ=0 输出恢复、确定性读出、置换不变性、跨层最优分配、检查点往返、评估 CSV 和固定样本可视化。临时更新不保留实验权重，不运行训练轮次。
