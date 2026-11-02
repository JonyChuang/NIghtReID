# 多层发现邻居、最终检索特征对比 CNL

2026-10-21 修正。在当前 `mldc_information_preservation` 实验副本原地修改，保留后四层均匀起步的可学习融合、原 ID／Triplet、BN、优化器与调度。仅替换辅助损失；CNL 不读取真实 PID，整体仍保留基线的身份监督。

## 运行与结果目录

从原服务器工程根目录运行：

```bash
python mldc_information_preservation/train.py
```

本地独立副本根目录的包装入口仍为 `python train.py`。
默认配置仍是 `configs/night600_shared_cnl.yml`；新输出为：

```text
./output/mldc_information_preservation/deep4_uniform_shared_cnl_retrieval09_120
```

该目录与上一版四层分别计算 CNL 的 `deep4_uniform_shared_cnl09_120` 分开。历史 `night600_bn_ratio.yml` 与 `night600.yml` 继续表示信息保持实验，已有结果不覆盖。代码更新不启动或停止训练；已运行进程继续采用其已加载的旧实现，下次启动采用修正版本。

## 分工与共享记忆

四层负责发现互补邻居，最终融合检索特征负责对比学习。一个按训练原图索引管理的逻辑记忆库同时保存：

- `features[N,4,768]`：CLS9–12 的四个同层挖掘键。
- `fused_features[N,768]`：最终融合、BN 后并 L2 归一化的检索键。
- `cameras[N]`、`valid[N]`：摄像头与有效槽标记，不保存 PID。

四层参考复用本批融合分支采用的 BN 均值、方差和仿射映射，再分别归一化；参考、统计和参考侧仿射参数全部停止梯度。不额外调用 BN forward，每批仍只更新一次运行统计。当前融合查询直接采用主分支实际输出的 BN 后归一化描述子，不另建描述子头。

库的索引覆盖全部训练图像，随成功优化步骤在线填充；只有已有效的历史槽与本批图片参与。没有在第一批前提取完整训练集特征，也不从验证／测试图库初始化。历史记忆有动量更新固有的时滞；训练采用 batch BN，评估采用 running BN，与原基线一致。

## 四层筛选伪正邻居

对查询图片 i，候选集合 E[i] 是所有有效历史图片与当前批图片，排除自身及不合规摄像头；默认要求摄像头不同。每层仅进行同层余弦比较：

```text
s[i,l,j] = cosine(z[i,l], M[j,l])
threshold[i,l] = 0.9 * max_{j in E[i]} s[i,l,j]
N[i,l] = {j in E[i]: s[i,l,j] >= threshold[i,l]}
P[i] = union_{l=9..12} N[i,l]
```

每层保留所有达标图片，取并集后按原图索引去重，不使用全层共用最大值、跨层直接余弦、Top-K 截断或邻居交集。记录支持每个候选的层数，但不据此加权第一版损失。若某层最大相似度为负，原0.9相对门槛可能使该层没有邻居，不另加回退规则。

当前图片的四层与融合键共同替换其旧槽；重复增强先分别合并归一化方向，再成为一个候选。自身所有增强均排除，不会重复计为多个正邻居。

## 最终融合检索特征上的 CNL

四层的相似度只确定 P[i]，不用于 CNL 的概率。设 g[i] 为当前最终检索特征，G[j] 为记忆中的最终检索特征：

```text
p[i,j] = exp(cosine(g[i], G[j]) / 0.05)
         / sum_{k in E[i]} exp(cosine(g[i], G[k]) / 0.05)
L[i] = -mean_{j in P[i]} log p[i,j]
L_CNL = mean_{active i} L[i]
L = original_ID + original_Triplet + 0.1 * L_CNL
```

所有合规候选均参与分母，包括批次外正邻居与负候选。对每个正邻居分别取负对数再平均，避免只匹配一个容易的邻居即可满足目标。不计算四个单层 CNL，也不平均四个单层的概率。

未入选候选是伪负样本，无法在不读取 PID 的情况下保证其真实身份不同。没有正邻居或没有负候选时，该查询不参与辅助损失；整批无有效查询时返回可反传零值，原 ID／Triplet 继续工作。`PRESERVE_WEIGHT=0`，不叠加旧信息保持损失。

## 梯度与记忆更新

伪邻居选择、四层参考与所有候选键均停止梯度。CNL 只经当前最终融合查询回传，直接更新骨干、可学习融合 logit 与 BN 仿射参数。分类器仅由原 ID 损失更新。最终检索描述子维度仍为768。

每次 AMP 确认优化器成功更新后，四层键与融合键按同一图片索引同步写回本次前向特征：

```text
M[i,l] = normalize(0.2 * old[i,l] + 0.8 * current[i,l])
G[i]   = normalize(0.2 * old_fused[i] + 0.8 * current_fused[i])
```

首次直接写入；重复原图先合并方向。前向、反向和评估不写记忆；AMP 跳步不更新任何一类键。融合键来自实际融合描述子，不能通过平均四个记忆层键代替。

## 日志、检查点与验证

`training.csv` 保留 CNL、有效查询比例、正邻居数、各层发现数和支持层数分布，新增：

- `cnl_mean_out_of_batch_candidates`：每批历史候选图片数均值，摄像头过滤前。
- `cnl_out_of_batch_positive_fraction`：所选正邻居中批次外图片所占比例。
- `cnl_memory_coverage`：成功更新后全训练集记忆槽覆盖率。

`layer_weights.csv` 保留可学习融合权重轨迹。检查点标识为 `mldc_multilayer_mined_retrieval_cnl_20261021`，明确记录最终融合对比空间、v2记忆结构及全部键。上一版只有四层键的记忆不能直接恢复到新结构。入口从原SSL初始化，仍不支持完整断点续训。

```bash
python -m mldc_information_preservation.verify_shared_cnl --full-model --report verification_shared_cnl_retrieval.json
```

验证包括：融合空间解析损失、批次外正负候选影响、逐层门槛与并集、同图去重／自身与摄像头过滤、无效槽排除、五类键同步动量更新、无PID接口、BN映射与单次统计更新，以及真实ViT的256×128随机输入前后向。

完整模型检查单独验证 CNL 对融合logit、BN及Block9–12的非零有限梯度，候选键与四层挖掘参考无梯度，检查点记忆可恢复，评估输出768维。本次检查不加载SSL权重、不执行优化器更新、不运行训练循环，也不产生新检索指标。
