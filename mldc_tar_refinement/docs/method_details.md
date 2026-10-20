# Night600：MLDC + TAR 末层 CLS 残差检索

本目录沿用原实验入口和目录名称 `a3pfn_mldc_lpam`，当前生效模型为
`mldc_tar_residual_three_heads_v1`。不实例化 LPAM、跨层局部读取、局部拼接或自适应融合门控。
历史 `model/lpam.py` 和旧可视化函数仅作为未调用的兼容源码保留。

## 计算

1. 原 ViT 输出第 1–12 个 block 的原始 CLS。MLDC 按 `softmax(layer_weights / 0.9)` 融合为 g。
2. TAR 读取第 12 个 block 的原始 CLS 与完整 patch 序列，均在骨干最终 LayerNorm 之前。
3. TRM 用 patch 范数的 z-score 得到 `sigmoid(-2*zscore)` 权重；保留权重最大的 50%（16×8 网格中为 64 个候选）。这不是前景分割。
4. CATS 以末层 CLS 为 query，对重加权候选做投影点积、sqrt(768) 缩放和温度 0.8 的 masked softmax。
5. GRM 将聚合结果经 LN、256D 瓶颈残差 MLP 和可学习门控注入末层 CLS：`c_refined = c12 + gamma*a`。
6. 最终原始特征为 `f = g + c_refined`。不先归一化、不替换 MLDC 内部原始 c12、不仅相加 TAR 增量。
7. 最终检索使用独立 `residual_neck(f)`，由原评估器做 L2 归一化，维数为 768。

MLDC 已包含 c12；外部残差再次加入 c12 是本版本的明确设计，需通过普通残差对照区分 TAR 的效果。

## 三处独立监督

MLDC g、TAR c_refined、最终残差 f 分别具有独立 BN/classifier。
ID 使用各自 BN 后特征，Triplet 使用各自 BN 前原始特征和原 PK batch-hard soft-margin。

`L = ReID(g) + 0.5*ReID(c_refined) + ReID(f) + 0.001*(0.05*KL(pi || uniform_candidates))`

没有停止梯度。全部有效模块从第 1 轮联合训练；辅助头只用于监督与独立诊断，最终只有一个检索描述子。

## 原训练协议

通用 SSL 初始化，120 轮，SGD，基础 LR 0.0004，20 轮 warmup，原调度器，PK=16×4，seed=1234。
骨干、MLDC、TAR 使用基础学习率，三个 BN/classifier 各使用原 5 倍头部倍率及原 bias 设置。
TAR 的范数统计、概率和细化采用 FP32；推理不加入随机扰动。
没有 epoch 0 评估；每 10 轮仅评估最终检索描述子一次，固定六对图像可视化另行保存。

新输出目录：`/root/log/transreid/night600/vit_base_ics_a3pfn_mldc_lpam/mldc_tar_residual_three_heads_120`。
旧实验结果不覆盖。旧 guided-readout / V3 检查点不能作为本架构的 resume。

从项目根目录手动启动：

```bash
python mldc_tar_refinement/train.py
```

断点恢复仅使用本架构、同配置、同输出目录的 latest 检查点：

```bash
python mldc_tar_refinement/train.py --resume /root/log/transreid/night600/vit_base_ics_a3pfn_mldc_lpam/mldc_tar_residual_three_heads_120/transformer_latest.pth
```

## 独立诊断

```bash
python mldc_tar_refinement/test.py --diagnostics TEST.WEIGHT /root/log/transreid/night600/vit_base_ics_a3pfn_mldc_lpam/mldc_tar_residual_three_heads_120/transformer_120.pth
```

一次提取后分别比较最终残差、MLDC 和 TAR 的 BN 前/后特征，以及原始末层 CLS、均匀 CLS 和普通 `g+c12`。
普通残差读出只属于同检查点诊断，不能替代独立训练的普通残差对照。

`training.csv` 记录六个 ReID 损失分量、KL、三头准确率、TAR gate、增量/普通残差范数比及候选数。
固定样本保存原始 patch 范数、TRM 权重、候选 mask、CATS 概率及 NPZ。上述图不等于检索因果贡献。

## 来源与迁移边界

TAR 官方代码：https://github.com/VectorYangYiStar/TAR ，核对 revision `731e4b33741b66b40a1b5f01a341648adc834311`。
模块依据 `model.py` 的 TRM/CATS/GRM；50% 比例参考 `scripts/run_cub.sh`；KL 外部权重与内部 beta 参考 `s2_GET_TAR.py` 和 DINOHead 默认参数。
作者 GET 脚本虽计算扰动一致性损失，但未纳入总损失；本版亦不加入。
公开 TRM 使用 `mask.scatter(...)` 后未接收返回值，导致原 mask 全零；本版正确构造布尔候选 mask。
数值核验以仅修正该 scatter 问题的官方模块为参考，并单独验证原代码空 mask 的现象。
迁移的是视觉 token refinement，不引入 GCD 的文本、聚类或未知类别训练任务。
与 MLDC 的残差、三头 ReID 监督均为本实验适配，不声称是原 TAR 全套训练的复现。
