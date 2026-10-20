# MLDC 引导的跨层空间读取

在现有 `a3pfn_mldc_lpam` 副本中替换局部模块。启动入口仍为项目根目录下的
`python mldc_tar_refinement/train.py`。代码修改不启动训练或评估。

MLDC 仍将全部 12 层原始 CLS 加权为 BN 前 768D 特征 g。局部分支分别读取第
3、6、9 个 block 的原始 patch 输出，每层 128 个 768D token。以 LN(g) 生成
各层的单 token Q，以该层 LN(patches) 生成 K/V，使用四头 scaled dot-product
交叉注意力，softmax 沿 128 个 patch。各层拥有独立的 Q/K/V/output 投影。

每层的 AV 读出经过输出投影、LN、残差 MLP 和 LN，得到一个 256D 描述子。
不将 g 加到读取输出，不跨深度预先融合 patch，不使用自由查询、水平条带或 FFT。
三个描述子表示全局引导的空间信息，不预先对应三个身体部位。

## 检索

g 和三个读取输出各有独立 BN。各自 L2 归一化后，由
LN -> Linear(1536,128) -> GELU -> Linear(128,4) -> softmax 预测当前图像的
[w_global,w_layer3,w_layer6,w_layer9]。初始为 [0.8,0.2/3,0.2/3,0.2/3]，
这是可学习门控的初始化，测试时没有固定融合比例。

最终特征为 concat(sqrt(w_global)*g_unit, sqrt(w_layer3)*r3_unit,
sqrt(w_layer6)*r6_unit, sqrt(w_layer9)*r9_unit)，共 1536D。
最后一层 CLS 已在 MLDC 输入中，不再单独拼入检索特征。

## 监督与梯度

loss = global_ID + global_triplet + 0.5*(read_ID+read_triplet) + 0.5*fusion_triplet

全局 ID 作用于 BN 后全局分类器，全局 Triplet 作用于 BN 前 g，原全局权重与
损失形式保留。读取 ID 作用于三个 BN 后描述子拼接的一个分类器；读取 Triplet
作用于三个 BN 前描述子的直接拼接。融合 Triplet 作用于实际用于检索的最终特征。
三项 Triplet 都沿用基线 PK batch-hard mining；NO_MARGIN=true 时使用 soft-margin。
融合描述子单位化后的距离尺度与原始特征不同，其损失值不宜直接等同为梯度贡献。

不设逐层身份分类/逐层 Triplet、注意力重叠惩罚、全局候选负样本限制或跨摄像头
特殊挖掘。不对 g、门控输入或 patch 停止梯度，不缩放局部到骨干的梯度。
所有有效模块从第 1 轮联合训练；读取监督通过 Q 回传 MLDC，通过 K/V 回传骨干。
保持 MLDC 计算与原监督并不代表其训练后特征数值不变。

## 配置与输出

保留 120 轮、SGD、基础 LR=0.0004、20 轮预热、原调度器、batch=64、每 ID 4 张、
seed=1234、原输入与增强。全局及读取 BN/classifier 用 5 倍学习率，读取核心、
MLDC 与融合门控使用基础学习率。仍沿用通用 SSL 骨干初始化，不加载旧实验权重。
旧检查点不可续训为新架构；--resume 只恢复本架构、本配置的检查点。

不做 epoch 0 评估；每 10 轮只用原评估器评估最终融合特征一次，无额外分支验证。
固定 6 对图像的可视化保留，并改为显示各层实际交叉注意力的头平均、层融合权重。
NPZ 同时保存逐头权重，热图不是因果贡献或身体部位标签。

代码目录保持不变，新训练输出到原日志根目录下的 `mldc_guided_readout_120`，
避免与已完成的 v3 日志/权重混用。没有新建实验代码副本。

## 独立测试的多特征诊断

训练期间仍每 10 轮只验证最终融合描述子，不运行额外分支诊断。训练结束后可手动执行：

```bash
python mldc_tar_refinement/test.py --diagnostics TEST.WEIGHT /root/log/transreid/night600/vit_base_ics_a3pfn_mldc_lpam/mldc_guided_readout_120/transformer_120.pth
```

一次遍历验证集提取所有特征，然后用同一原始评估器依次计算：最终融合（1536D）、
MLDC BN 前/后（各 768D）、原始末层 CLS（768D）、12 层均匀平均 CLS（768D）、
三个读取描述子拼接的 BN 前/后特征（各 768D）。每种情况使用相同的 L2 归一化、
相机过滤和检索协议。末层/均匀 CLS 不套用只为 MLDC 训练的 BN。

加 `--layer_diagnostics` 可再比较第 3/6/9 层读取描述子各自的 BN 前/后特征
（各 256D），总计 13 种情况。此参数须放在 TEST.WEIGHT 等配置覆盖项之前。
多种距离矩阵依次计算并释放，但总计算量仍随诊断数量增加；不会重复执行完整数据集的骨干提取。

同时输出 Raw readout/last-CLS mean cosine，保存结果到
`checkpoint_analysis/epoch_120/retrieval_diagnostics.csv` 和 JSON，并保留固定对热图。
raw_last_cls 等是同一个联合模型检查点的读出诊断，不是独立训练的 TransReID-SSL
或独立 MLDC 基线结果；归因时应区分它们。
