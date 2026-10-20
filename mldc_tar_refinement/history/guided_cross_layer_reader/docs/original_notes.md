# MLDC + LPAM v3：浅层金字塔、竞争查询与可学习融合

正式入口仍为 `python mldc_tar_refinement/train.py`，在项目根目录执行。默认输出为原日志目录下的 `free_queries_v3_pyramid_adaptive_120`。此前的权重和实验记录保留。本次没有启动训练。

## 数据流

MLDC 保留对 12 层 CLS 的聚合与原全局 BN/classifier。LPAM 读取从 1 开始计数的第 3、6、9 个 block 的空间 patch。各层经独立 LN 和 768→256 投影，按 9→6→3 将相同位置的深层 patch 加入浅层，再做 LN。

四个查询使用四头余弦注意力，温度仍为 0.1。每个 patch 先沿查询维 softmax，形成查询间竞争，再沿空间位置归一化，得到各查询的加权平均。各查询继续学习三个金字塔阶段的门控；stage 3 已包含 stage 6/9 的输入，不应把其权重解释为纯第 3 层的独立贡献。

作者公开代码接入 3/6/9/12 层 CLS 并进行金字塔聚合。本版本读取 3/6/9 层空间 patch，是针对 Night600 的改造，不是原 LPAM 的等价复现。保持无 FFT、无水平条带。

## 可学习检索融合

将全局 BN 后的 768D 特征和四个局部 BN 后的 256D 特征分别 L2 归一化。门控网络接收这些特征的拼接，结构为 LN → Linear(1792,128) → GELU → Linear(128,5) → softmax，输出当前图片的 `[w_global,w_P1,w_P2,w_P3,w_P4]`。

初始为 `[0.8,0.05,0.05,0.05,0.05]`；这只通过门控末层的初始化实现，此后所有权重由训练更新。配置中的 `MODEL.LPAM_FUSION_INIT_LOCAL_WEIGHT=0.2` 是初始化值，不是测试融合系数。已经删除固定的 `TEST.LPAM_WEIGHT`。

最终 1792D 检索描述子：

`f(x) = concat(sqrt(w_global(x))*g_unit(x), sqrt(w_P1(x))*p1_unit(x), ..., sqrt(w_P4(x))*p4_unit(x))`

图片 q 与 t 的相似度为 `sum_i sqrt(w_i(q)*w_i(t))*cos(f_i(q),f_i(t))`。权重随图像变化，不能再简单称作固定 80/20 分数平均。预测权重尚不是经过校准的可见性标签。

## 训练目标

保留全局 ID 与原全局 soft-margin Triplet，权重均为 1。局部增加四个独立的辅助身份分类器；原局部整体分类器仍保留。

`local_ID = 0.5*whole_local_ID + 0.5*mean(part_ID)`

`local_triplet = 0.5*whole_local_triplet + 0.5*mean(part_triplet)`

局部度量使用 BN 前、分别归一化的特征，margin 仍为 0.3。批次中有跨摄像头同身份样本时优先从这些样本选最难正样本，否则回退至其他同身份样本。负样本沿用全局相似度前 8 个候选，再按相应描述子的距离选择。

新增 `fusion_triplet`：直接作用于实际部署的 BN 后自适应融合描述子，margin=0.1。该目标对全局特征及其门控输入停止梯度，对局部特征、局部 BN 和融合门控正常求导。于是融合权重由与检索一致的距离目标学习，而全局头仍接受原有监督。停止该附加损失的全局梯度并不冻结全局分支。

`loss = global_ID + global_triplet + 0.5*(local_ID+local_triplet) + 0.5*fusion_triplet + 0.01*attention_overlap`

局部路径进入共享骨干的梯度乘以 0.25；前向数值不变，LPAM、局部头和融合门控自身梯度不缩放。全部模块与骨干从第 1 轮起联合训练，无分阶段冻结，不加载已完成的 MLDC/LPAM 检查点；仍沿用原通用 SSL 骨干初始化。

## 保持的实验设置

120 轮、SGD、momentum=0.9、基础 LR=0.0004、20 轮预热、原调度器及轮末 step、bias factor=2、batch=64、每 ID 4 张、seed=1234、原输入尺寸与增强。全局 BN/classifier 及所有局部 BN/classifier 的学习率保持 5 倍；LPAM 核心与融合门控使用基础学习率。主检索仍为 BN 后、L2 归一化、原欧氏距离与同身份同摄像头过滤、无重排序。

每 10 轮只评估最终融合特征，不增加分支/分区域验证，不做第 0 轮评估。固定 6 对已有可视化沿用一次前向；每个局部标题显示对应融合权重及三阶段深度权重，原图标题显示全局权重。NPZ 增加 fusion_weights、attention_by_depth、patch_layers。training.csv 与训练日志增加平均融合权重、平均局部查询分类准确率和各新增损失分量。

第 120 轮为预定终点，best 仅为已记录测试 mAP 诊断。v3 检查点带独立 variant 与 architecture 标识，不能将 v2 权重直接续训为 v3。断线恢复只用本次 v3 的 latest，并保持原配置。此前输出目录非空保护保留。

## 证据与边界

已有 v2 联合模型 BN 后全局为 10.1032/20.4128，局部为 9.8953/19.4495，固定融合为 10.1708/20.6881；独立 MLDC 为 10.2141/21.1468。v3 针对全局略弱、局部净补充较小及查询重复进行改造，尚无新训练结果，不承诺 12.5/25。

局部空间/层级改造、梯度缩放和新增监督同时变化，这次主实验只能检验整体方案，不能单独归因各项收益。最小后续消融应在相同 120 轮预算下分别关闭竞争、逐查询监督、融合距离监督与梯度缩放；不自动启动这些实验。

参考公开机制：

- A3PFN 作者代码： https://github.com/zhguoqing/A3PFN/blob/main/model/backbones/vit_pytorch.py
- Slot Attention 的竞争分配思想： https://arxiv.org/abs/2006.15055 。本版本只借用跨查询竞争与归一化，不实现完整的迭代 Slot Attention。
- PAT 的可学习局部查询与局部判别表示： https://arxiv.org/abs/2106.04095 。本版本不复现 PAT 全套网络。

这是实验改造，不据此主张新的方法首创性。
