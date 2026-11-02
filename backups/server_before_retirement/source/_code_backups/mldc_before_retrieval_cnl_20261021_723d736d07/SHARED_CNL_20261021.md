# 共享四层记忆库 CNL

基于 `deep4_uniform_bn_ratio08_preserve01_120` 副本原地修改。按用户确认保留后四层均匀起步的可学习融合、原 ID／Triplet、BN、优化器和调度；将监督式信息保持损失替换为不读取真实 PID 的共享邻居损失。

## 默认运行

```bash
python train.py
```

或从工程根目录显式指定：

```bash
python mldc_information_preservation/train.py --config_file mldc_information_preservation/configs/night600_shared_cnl.yml
```

新输出：`./output/mldc_information_preservation/deep4_uniform_shared_cnl09_120`。
原 `night600_bn_ratio.yml` 和 `night600.yml` 继续表示历史保持损失实验，已有结果不覆盖。新版本从原 SSL 初始化，不恢复旧保持记忆或优化器状态；本次修改未启动训练。

## 特征与记忆

一个 `[N,4,768]` FP32 张量，按训练原图索引管理，四个槽分别对应 CLS9–12。另存摄像头编号及有效槽标记，不存真实 PID。

当前四层 CLS 复用本批融合分支实际采用的 BN 均值、方差及仿射映射，再分别 L2 归一化。统计量停止梯度，当前 CLS 和 BN 仿射参数可反传。该映射不另调用 BN forward，运行统计每批仍只更新一次。旧保持参考函数继续停止梯度，不改变历史配置行为。

当前批和记忆更新使用同一组映射后的特征；历史槽仍存在动量记忆固有的时滞。没有对同一图片的四个槽求均值，也没有跨层直接算余弦。

## 正邻居与损失

对当前图片 i、记忆图片 j、第 l 层：

```text
s[i,l,j] = cosine(z[i,l], M[j,l])
threshold[i,l] = 0.9 * max_j s[i,l,j]
neighbors[i,l] = eligible[i,j] AND s[i,l,j] >= threshold[i,l]
positive[i] = union_l neighbors[i,l]
```

求最大值前即排除自身及不合规摄像头。默认只允许其他摄像头；未写入的历史槽不参与。本批图片以停止梯度的当前特征替换旧槽，重复增强逐层合并成一个图片候选，因此自身任意增强均不成为邻居，同图也不会重复计算正样本数。

保留每层全部达标候选，不加入 Top-K 截断、四层交集或互为近邻硬门槛。按图片索引去重后得到共享伪正集合，并统计支持该候选的层数；支持数仅记录，不改变第一版损失权重。

每层用自己的同层分数，在所有合规图片候选上做温度 0.05 的 Softmax。对共享集合中的每个正邻居取负对数概率，先平均正邻居，再等权平均四层和有效查询。未入选的合规图像是候选负样本，不能保证真实身份不同；不再用 PID 排除可能的同身份负样本。没有正邻居或没有负候选的查询返回零辅助损失，ID／Triplet 继续工作。

```text
L = original_ID + original_Triplet + 0.1 * shared_CNL
```

新配置显式设置 `PRESERVE_WEIGHT=0`，不会同时叠加旧保持损失。CNL 从第一轮开始，沿用原副本的训练时间安排，没有另加预热或改变原学习率预热。

## 梯度与更新

伪邻居选择及所有候选键停止梯度；CNL 通过四个当前 CLS 查询和共享 BN 仿射参数回传。融合 logit 继续由原 ID／Triplet 学习，CNL 不直接监督这些 logit。最终仍输出原可学习融合的 BN 后 768 维描述子。

每次 AMP 确认优化器成功更新后，四层同步按下式写回本次前向特征：

```text
M[i,l] = normalize(0.2 * old[i,l] + 0.8 * current[i,l])
```

首次直接写入；重复原图先对归一化特征求均值方向。前向、反向、评估均不修改记忆，AMP 跳步也不写记忆。建库从有效本批候选开始，随成功更新逐步填充，不从测试集／图库初始化。

## 日志与检查点

`training.csv` 记录 CNL、有效查询比例、无正／无负比例、平均正邻居数、候选正例占比、支持层数分布、每层发现的邻居数，以及有效记忆图片数。`layer_weights.csv` 保留原融合权重轨迹。

新检查点架构标识为 `mldc_shared_layer_cnl_20261021`，保存四层共享记忆、相机与有效槽、CNL 定义参数及完整配置；旧 `preservation_memory` 不会误解释为新记忆。保存记忆不表示支持完整断点续训。

```bash
python -m mldc_information_preservation.verify_shared_cnl --full-model --report verification_shared_cnl.json
```

检查包括解析损失数值、各层独立门槛、正邻居并集、图片去重及自身过滤、相机过滤、四层动量更新、无 PID 接口、BN 映射一致性与单次统计更新、四层梯度、真实 ViT 256×128 前向／反向、768 维输出和检查点记忆恢复。使用随机输入，跳过 SSL 加载，不执行优化器更新或训练循环。
