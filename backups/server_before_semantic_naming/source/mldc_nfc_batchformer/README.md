# MLDC + CNL + 缓存互近邻 BatchFormer

2026-11-02 的独立实验副本。运行入口只使用本文的 neighbor 配置。
原 `mldc_convstream_texture_cnl` 包、配置、进程和结果目录均不修改。

## 本次设置

- 继承当前实际运行的 Night600 对照：SSL 初始化、可学习 CLS9—12 融合、seed1234、120轮、PK64/K4、原学习率与20轮预热。
- 损失：`2*CE_original + 1*CE_neighbor_augmented + Triplet_original + 0.1*CNL_original`。
- 前20轮保留原路2倍CE、Triplet、CNL并积累缓存，不执行邻域增强；第21轮开始启用邻域增强。这一点与原版第1轮启用 BatchFormer 不同。
- 单独缓存每张训练图片最近一次有效优化步骤的 BN 前 MLDC 原始特征。缓存停止梯度，重复图片索引的多个增强先取均值。未成功执行优化的 AMP 步不更新缓存。
- 每轮开始，用缓存特征的 L2 副本，分块计算全训练有效候选池的 reciprocal top-32。按相似度保留最多8个互近邻；当轮固定邻居索引，缓存向量继续按训练步骤刷新。图关系和向量都存在训练时间上的滞后。
- 排除自身及同图副本；建图不使用身份或相机。标签仅用于原监督及事后的邻居诊断。
- 每组输入一个当前图片特征加最多8个缓存邻居。组间不做注意力交互；共享一层4头、FFN768的编码器；取当前图片位置输出作为增强特征。
- 邻居不足8个时使用 padding mask；没有有效邻居时只做原路监督。增强CE在当前有邻居的锚点上取平均，使用锚点本人的标签。
- 保留共享分类器与一次共享BN：BN输入为N行原路加M行有效增强锚点。缓存邻居行不直接进入BN。无邻居时不额外复制原路行。
- Triplet、CNL和CNL记忆库仍只使用原路。邻域缓存与CNL库相互独立，避免混合BN前、BN后的特征。
- 测试使用单图主干+同一MLDC+原BNNeck+L2，不访问缓存，不执行BatchFormer。训练不额外编码邻居图片。

全训练候选池在预热期间逐步填充，没有额外全数据集初始化前向。14,462张图片的FP32原始特征约44.4MB；图索引和刷新时的归一化副本另占空间，分块搜索不保存完整方阵。

## 运行

在现有工程根目录：

```bash
/root/miniconda3/envs/QREID/bin/python -u -m mldc_nfc_batchformer.train_batchformer --config_file mldc_nfc_batchformer/configs/neighbor_night600.yml
```

结果目录：
`/root/log/transreid/night600/mldc_nfc_batchformer/cached_reciprocal_k32_n8/from_ssl_seed1234_120`。

另外提供 `neighbor_nightreid.yml`，本次排队只运行Night600。

## 顺序执行

`queue_after_current.py`由部署脚本启动一次。它记录当前主训练PID及进程开始时间，只等待、不停止任何进程。当前实验退出后，核对120轮训练/评估/最终检查点，并等显卡空闲，然后依次运行：

1. 仅合成图像的GPU集成检查。
2. 新副本从SSL开始完整120轮训练，每10轮做独立单图检索评估。
3. 将最佳mAP对应指标及第120轮指标与当前对照比较，写入新结果目录的 `comparison_to_predecessor.json`。

队列状态见本包 `queue_state.json`；合成检查见 `gpu_check.log`；正式训练输出见 `training_console.log`。前序实验未完整结束或检查失败时不启动正式训练。

## 诊断

`training.csv`保留原损失和检索指标相关日志，并增加邻居有效率、平均数量、满8邻居比例、同身份率、跨相机率、缓存年龄、候选覆盖和图刷新时间。关系纯度只用于分析，不参与筛选。

该方案尚无新检索结果。它是对BatchFormer的邻域输入适配，不是NNCLR或原BatchFormer论文的完整复现。
