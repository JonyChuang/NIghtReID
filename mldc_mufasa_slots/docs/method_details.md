# MLDC + MUFASA 联合训练：BNNeck 尺度修正版 v3

从 `pretrain_model/vit_base_ics_cfs_lup.pth` 的 TransReID-SSL 权重开始，全部有效 ViT block、MLDC 层权重、全局 BN/分类器和 MUFASA 模块从第 1 轮一起训练。不加载独立 MLDC 第 120 轮权重。原模型未参与前向的末端 norm/fc/gem 沿用独立 MLDC 的禁用设置。

## 结构与改进

- 12 层原始 CLS 经 MLDC 得到 BN 前特征 `g_raw`，原 BN 后得到 `g`。
- 第 9/10/11/12 层的 128 个空间 patch 保留梯度，分别进入 4-slot、256 维、3 次迭代的 Slot Attention；掩码匹配及 M-Fusion 保持原设计。
- MLDC 查询融合 slot 集合得到整体残差 `r`，不对每个 slot 单独做身份分类，不拼接多个 slot 为检索输出。
- 图像门控以单位化 `g/r` 为输入，通过 1536→128→1 MLP 预测 `alpha=0.2*sigmoid(logit)`；初始每张图 alpha=0.1。
- 训练使用未单位化融合特征 `h=g+alpha*||g||*L2(r)`；新增独立 `fusion_bottleneck`（BNNeck，weight=1，bias=0且不训练bias），分类路径为 `classifier(dropout(BN(h)))`，Triplet 使用 BN 前的 `h`。
- 检索仍使用原公式 `f=L2(L2(g)+alpha*L2(r))`，与 `L2(h)` 方向一致，仍为单一 768 维；新增 BN 仅供分类训练，不用于检索。原 MLDC 分支的 BN/分类/Triplet 保持原实现。
- 对整体 `r` 增加 cosine-margin 难例监督：每个 anchor 先根据 detached 的当前 MLDC 相似度选最多 8 个异身份候选，再以 `r` 选最难负样本；正样本是批内同身份最难非自身样本。初始 margin=0.1、权重=0.1，均是待验证的实验设置。
- 训练期解码器重建当前第 12 层 patch；只有重建目标 detach，权重保持 0.1。该目标不参与推理。

`L = ReID(global) + ReID(final) + 0.1*residual_metric + 0.1*reconstruction`

全局 ReID 与独立 MLDC 一致：BN 后分类，BN 前原始特征做 PK batch-hard soft-margin Triplet；ID/Triplet 系数各为 1。融合分支采用同样的 BN 后分类、BN 前 Triplet 训练形式；融合使用 BN 后的全局 g，并不等同于原 MLDC 的 BN 前原始特征。整体残差的附加度量监督不修改全局难例选择的梯度。

## 独立 MLDC 训练配置

直接复用项目 `solver.make_optimizer` 和 `solver.scheduler_factory.create_scheduler`，保留 epoch 末调用调度器的时机。SGD、momentum=0.9、基础 LR=0.0004、bias 倍率=2、weight decay=0.0001、全局 BN/classifier、融合 BN/classifier 倍率=5；新 slot/readout/gate/decoder 使用基础组规则。原调度器预热期的实际 LR 行为也保留，不另写近似调度。

120 轮、20 轮预热、batch=64（16 ID×4）、seed=1234、256×128 输入及原数据增强。ViT 沿用原 AMP，slot/门控/重建和所有损失用 FP32。每 10 轮评估及保存固定 6 对可视化。

评估同时输出当前联合模型的 `mldc/final/residual_only`。`mldc` 不是历史独立基线；修复/破坏 query 也是相对当前模型的 MLDC 分支。门控均值、残差难例损失、两路 ID/Triplet/准确率和 MLDC 层权重均保留记录。训练新增全局/融合/融合 BN 特征范数及最终 logits 标准差；文本日志额外显示最终分类头实际学习率，用于判断监督尺度。

## 手动运行

在服务器 TransReID 项目根目录执行：

```bash
python mldc_mufasa_slots/train.py
```

默认输出为 `/root/log/transreid/night600/vit_base_ics_a3pfn_mldc_mufasa/joint_ssl_bnneck_120`。原 `joint_ssl_120` 目录及正在运行的旧进程保留；修改磁盘代码不会将已运行的进程切换为新版本。代码修改和验证不会启动这条训练命令。

按最终特征 mAP 保存 `transformer_best.pth`；每次评估保存可恢复的 `transformer_latest.pth`，第 120 轮保存 `transformer_120.pth`。检查点类型为 `joint_mldc_mufasa_bnneck_v3`，包含完整骨干、新 BN、新模块、优化器、原调度器、AMP scaler 和随机状态。旧冻结增量及 v2 联合训练检查点不兼容新训练协议；修正版从原 SSL 预训练重新训练，不能直接 `--resume` 当前旧实验。

```bash
python mldc_mufasa_slots/train.py --resume /root/log/transreid/night600/vit_base_ics_a3pfn_mldc_mufasa/joint_ssl_bnneck_120/transformer_latest.pth
python mldc_mufasa_slots/train.py --evaluate /root/log/transreid/night600/vit_base_ics_a3pfn_mldc_mufasa/joint_ssl_bnneck_120/transformer_120.pth
```

机制对照可以设置 `MUFASA.ADAPTIVE_GATE False`（固定 0.1）或 `MUFASA.RESIDUAL_METRIC_WEIGHT 0.0`（关闭残差附加监督），并使用不同 OUTPUT_DIR。它们不是新模型已经取得的实验结果。

## 代码验证

`verify.py` 检查配置/调度对齐、初始化、一个真实 PK batch 的前向与梯度、BN 分类/未单位化 Triplet、检索方向一致性，以及完整检查点格式；不调用 optimizer.step、不执行 epoch 训练，也不生成正式实验检查点。MUFASA 来源仍为 https://github.com/visinf/mufasa ，当前门控、ReID 监督和联合训练为本项目适配。本次只修正已定位的训练尺度，不新增 slot 正则或残差身份分类，slot 同质化是否改善需要新实验确认。
