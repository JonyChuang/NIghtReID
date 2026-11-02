# MLDC + LocAt：Night600 独立实验副本

本副本保留当前 TransReID-SSL 的 stem、12 层 ViT、MLDC 的随机层权重初始化与温度 0.9，以及单分支 768 维 BN-neck。新增 LocAtViT 的 Gaussian-Augmented attention（GAug）和 Patch Representation Refinement（PRR）。无新增外部模型、数据、人体解析、姿态、分割掩码或监督信号。

论文：[Locality-Attending Vision Transformer](https://arxiv.org/abs/2603.04892)。
官方实现：<https://github.com/sinahmr/LocAtViT>，核对版本 `f99aac4c3d0f417f205faf35742ca25ebcc30ea4`。其 MIT 许可证见 `LICENSE.locat`。

## 实现与迁移位置

1. **GAug**：在 12 层自注意力的 `QKᵀ / √d` 上添加由 query 预测的二维高斯补充项。每个 head 的预测器共享于空间位置：`Linear(64,2)` 预测方差，`Linear(64,1)` 预测 softplus 系数 alpha。16×8 网格使用真实高宽，不对 token 数开平方。CLS 行和 CLS 列的补充项均为零，保留全局注意力路径。
2. **完整 token 的 MLDC 融合**：原方法融合各层 CLS。本副本用同一组 MLDC 权重融合完整 token 序列 `T = Σ_l w_l X_l`；其中 `T[:,0]` 与原 CLS 加权和在浮点容差内一致。另行计算原来的 CLS 融合供诊断；关闭新增模块时，检索输出与原基线完全一致。
3. **LayerNorm → PRR**：将融合序列经过原 SSL backbone 的最终 LayerNorm，再以 `softmax(T_h T_hᵀ / √d) T_h` 聚合，取 CLS 输出。PRR 无可学习参数、无额外 QKV 投影、无残差。本实现只计算 CLS 对应的一行，其输出与官方完整 PRR 后取 CLS 等价。
4. **检索**：PRR 输出 → 原 BN-neck → 训练时 ID 分类器；训练继续使用 ID CE + batch-hard soft-margin triplet。默认检索仍取 BN 后特征，再做 L2 归一化。

**与原论文的区别**：官方在最后一层归一化后的 token 上应用 PRR；本副本在 MLDC 融合后的 token 上应用 PRR，以保留多层融合。它是针对现有实验的迁移，不能称为原论文的完整复现。该模块改善局部信息的传播与读出，没有像素纹理重建损失，也不能恢复 patch 化阶段已经丢失的像素信息。

**实现细节以代码为准**：官方代码的方差上界是 `max(H,W)^2`，与正文对缩放范围的文字表述存在差别。本实验对齐官方代码：上界 256，零 logit 对应方差 1，并沿用官方 EPS。alpha 为可学习 softplus，不是固定权重，也不是 Q/K 范数校准。

GAug 新增参数共 **2340**（12 × (64×2+2+64×1+1)），PRR 新增参数为 **0**。启用读出 LayerNorm 时，原来未使用且冻结的 1536 个 LayerNorm 参数参与训练。GAug 使用原 backbone 学习率与原优化器的 bias LR 规则；BN-neck 和分类器沿用 5 倍学习率。

## 配置与对照

| 配置 | GAug | 读出 LayerNorm | PRR | 用途 |
|---|---:|---:|---:|---|
| `legacy.yml` | 否 | 否 | 否 | 原始 MLDC 路径 |
| `norm_control.yml` | 否 | 是 | 否 | 分离新增归一化的影响 |
| `gaug_raw.yml` | 是 | 否 | 否 | 仅修改原始 MLDC 的内部注意力 |
| `gaug_norm.yml` | 是 | 是 | 否 | 归一化读出下的 GAug 对照 |
| `prr_only.yml` | 否 | 是 | 是 | 归一化读出下的 PRR 对照 |
| `locat.yml` | 是 | 是 | 是 | GAug + PRR 主实验 |

`norm_control / gaug_norm / prr_only / locat` 构成读出归一化一致的 2×2 对照。`legacy / gaug_raw` 补充回答 GAug 能否直接改善原 MLDC。仅比较 `locat` 与历史基线，无法把差异全部归因于 GAug/PRR。

训练协议仍为 seed=1234、120 epoch、SGD、base LR=4e-4、20 epoch warmup、batch=64、P×K=16×4。验证推理 batch 调整为 128；评估同一检查点的多种特征时依次计算距离矩阵，以降低内存占用。未改变距离、过滤协议或重排序设置。所有配置使用同一份 SSL 初始化 `pretrain_model/vit_base_ics_cfs_lup.pth`，不从已有 MLDC 的第 120 轮检查点续训。

## 后续运行

从服务器 TransReID 工程根目录运行；**代码修改阶段没有启动训练**：

```bash
/root/miniconda3/envs/QREID/bin/python mldc_locat_attention/train.py \
  --config_file mldc_locat_attention/variants/gaussian_attention_prr/config.yml
```

评估本副本生成的检查点：

```bash
/root/miniconda3/envs/QREID/bin/python mldc_locat_attention/test.py \
  --config_file mldc_locat_attention/variants/gaussian_attention_prr/config.yml \
  TEST.WEIGHT /absolute/path/to/transformer_120.pth
```

若以后需要真实样本的权重加载/梯度检查，可单独运行 `verify_server.py --device cpu`；它不训练 epoch，不更新优化器，也不产生完整检索指标。

## 日志与可视化

保留每 10 轮评估、固定跨相机同身份 query/reference 对、原热力图和层权重曲线；新增：

- `*_locality.png/json`：逐层八邻域 patch cosine、全局 patch cosine、patch/CLS cosine、局部与全局相似度差、高斯方差与 alpha，以及同一 Q/K 下添加高斯项前后的注意力。
- `*_locat_attribution.png`：同一检查点的原始 MLDC、LocAt BN 前、BN 后检索特征对应的相似度归因。三张热力图共用色标；PRR 注意力另用色标。
- `*_locat.npz`：注意力、方差、alpha、归因图的原始数值。注意力变化只能说明计算机制发生变化，识别收益需要独立训练对照和检索指标支持。
- `locality_statistics.csv`：固定 query 的逐层诊断，不能解释为全数据集的平均统计。
- `evaluation_diagnostics.jsonl`：`raw_readout` 是最终 BN 前特征；`raw_mldc` 是同一骨干下、LN/PRR 前的原 CLS 融合；`raw_last_cls` 始终是原始末层 CLS；`raw_uniform_cls` 是原 CLS 均值；`prr_input_cls` 是进入 PRR 前的 CLS。

GAug 不改变 CLS 行，故仅以 CLS 融合训练时，第 12 层 GAug 的梯度为零，这是结构结果。PRR 使融合 token 中的 patch 输出进入损失，从而为末层 GAug 建立梯度。首个训练 batch 会记录各层 GAug 梯度；启用 PRR 后若梯度不有限或为零，会报告问题。

## 已完成的验证范围

本地验证使用随机权重和合成图片，覆盖官方 GAug/PRR 数值一致性、关闭模块后恢复基线、完整 12 层前向/反向、末层梯度路径、固定样本可视化、检索评估接口和检查点恢复。合成数据的内部检索数值不作为 Night600 结果。真实 SSL/Night600 的检查、完整训练和指标不包含在这些本地测试结论中。
