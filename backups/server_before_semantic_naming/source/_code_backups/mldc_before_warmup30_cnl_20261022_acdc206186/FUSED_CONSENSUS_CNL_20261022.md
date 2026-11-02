# 融合提名＋至少三层确认的 CNL

2026-10-22，在当前实验副本原地修改。默认采用已讨论的保守规则：融合空间0.9相对门槛与至少三层支持共同决定正例；0.80相对门槛建立不确定池，不确定图片不进入本次CNL分母。

## 入口与配置

原服务器工程根目录：

```bash
python mldc_information_preservation/train.py
```

本地独立副本根目录包装入口：`python train.py`。默认配置为 `configs/night600_shared_cnl_consensus.yml`。新结果目录：

```text
./output/mldc_information_preservation/deep4_uniform_cnl_fused09_support3_uncertain08_120
```

本次只修改、验证并同步代码，不启动训练。原有结果不覆盖。历史 `night600_shared_cnl.yml` 继续使用原四层并集；`night600_bn_ratio.yml` 和 `night600.yml` 继续表示原信息保持实验。

## 正例、不确定图片和负候选

E_i包含有效训练记忆与当前批去重图片，先排除自身，默认仅使用其他摄像头。当前图片的四层及融合键共同替换其旧槽。同图多个增强合并为一个候选图片。

```text
s_l(i,j) = cosine(z_l(i), M_l(j))
s_f(i,j) = cosine(g(i), G(j))
m_l(i) = max_{j in E_i} s_l(i,j)
m_f(i) = max_{j in E_i} s_f(i,j)
A_l(i,j) = [s_l(i,j) >= 0.9 * m_l(i)] AND [m_l(i) > 0]
F(i,j)   = [s_f(i,j) >= 0.9 * m_f(i)] AND [m_f(i) > 0]
P_i = {j in E_i: F(i,j) AND sum_l A_l(i,j) >= 3}
B_i = 任一层或融合空间达到各自正最大相似度0.80倍的合规候选
U_i = B_i minus P_i
N_i = E_i minus (P_i union U_i)
```

等于门槛也保留。最大值计算不包含自身及不合规摄像头；最大值非正、无有效候选时，不强行提名正例。四层只做同层匹配，没有跨层直接余弦；不设置Top-K、互近邻或真实身份过滤。

P_i参与正向匹配，U_i既不是正目标也不进入CNL分母，N_i作为候选负例参与分母竞争。未确认的互补图片保留在记忆中，随后续特征更新重新判断。N_i仍可能含少量同身份图片，不宣称其真实身份可靠性可由无PID规则完全保证。

## 损失与梯度

```text
p_i(j) = exp(s_f(i,j)/0.05)
         / sum_{k in P_i union N_i} exp(s_f(i,k)/0.05)
L_i = -mean_{j in P_i} log p_i(j)
L_CNL = mean_{active i} L_i
L_total = original_ID + original_Triplet + 0.1 * L_CNL
```

没有正例或负候选时跳过该查询的CNL；整批无有效查询时返回可反传零值。原ID与Triplet继续工作。

四层挖掘参考、邻居选择、候选融合键及所有掩码停止梯度。CNL通过当前最终融合、BN后归一化描述子，直接更新骨干、可学习融合logit及BN。分类器仍由原ID更新。没有额外逐层CNL，也没有用新模块替代最终768维描述子。

## 保留的设置与记忆更新

- CLS9–12均匀起步的可学习融合，融合温度0.9、融合权重学习率因子1、BN／分类头因子5。
- 原SSL初始化、ID／Triplet、数据增强、PK采样、120轮及优化器调度。
- 原0.1 CNL权重、0.05概率温度、跨摄像头限制及单次BN统计更新。
- 一个全训练图像索引的逻辑记忆，同时保存[N,4,768]参考键及[N,768]融合键，不存PID。

记忆在线填充，未初始化槽不参与。每次确认优化器成功更新后，同一图片的五个键同步执行 `normalize(0.2*old + 0.8*current)`；首次直接写入，重复索引先合并方向。前向、反向、评估及AMP跳步不更新记忆。

从原SSL重新开始，不把旧已训练模型或旧记忆自动加载为新实验起点。入口仍不提供完整优化器断点续训。

## 配置和日志

```yaml
SOLVER:
  AUXILIARY_LOSS: shared_cnl
  PRESERVE_WEIGHT: 0.0
  CNL_WEIGHT: 0.1
  CNL_SELECTION: fused_consensus
  CNL_MIN_SUPPORT: 3
  CNL_GAMMA: 0.9
  CNL_UNCERTAIN_GAMMA: 0.8
```

`training.csv`新增不确定候选、负候选均值，以及正例／不确定／负候选占合规候选的比例、实际进入分母的比例。`cnl_support_*_fraction`只统计最终正例的支持层数；`cnl_candidate_support_*_fraction`统计融合确认前四层0.9并集候选的支持层数，不能将二者混用。

保存检查点时标识 `mldc_fused_consensus_cnl_20261022`，记录三层支持、0.80不确定池及分母排除规则。特征记忆结构仍为v2；相同结构不表示不同筛选策略可直接作为相同实验续训。

## 验证

```bash
python -m mldc_information_preservation.verify_consensus_cnl --full-model --report verification_shared_cnl_consensus.json
```

解析测试覆盖：融合与三层条件同时成立、0.9与0.8等号边界、层空间／融合空间两类不确定候选均从分母排除、负候选仍影响损失、非正最大值不提名、空正例／空负候选、同图去重、相机与自身过滤、无PID接口、前向不写记忆及FP32概率计算。

完整ViT使用随机256×128输入，只做前后向及检查点内存往返，核查CNL对融合权重、BN和Block9–12的非零有限梯度，以及768维输出。另运行历史四层并集与信息保持检查。上述检查均不启动训练、不执行优化器更新，不产生新的检索指标。
