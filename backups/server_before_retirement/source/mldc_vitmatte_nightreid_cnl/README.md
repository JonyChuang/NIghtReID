# NightReID：ViTMatte DCM受控迁移

目的：验证从RGB原图读取的细节与ViT语义特征逐级融合，是否能改善完整图库中的排序。此前冻结patch读出没有稳定净收益，本实验改为从原SSL初始化开始联合训练独立RGB卷积分支。这是待验证的假设。

原文与官方代码：
- https://arxiv.org/abs/2305.15272
- https://github.com/hustvl/ViTMatte/blob/8cd7ef068380977c3962c4cb733cb1fe7f2241a5/modeling/decoder/detail_capture.py

迁移官方ConvStream（48/96/192通道）与四级逐步上采样、拼接、卷积融合（256/128/64/32通道）。将输入改成3通道RGB，不使用trimap、透明度预测头或抠图损失。固定末层patch网格为16×8；末层patch与RGB细节融合至256×128，再作均值池化并投影为256维单位向量r。保留原MLDC全局768维单位向量g；最终检索向量为concat(sqrt(0.9)g,sqrt(0.1)r)，共1024维。其余弦相似度等于0.9全局相似度+0.1细节相似度。0.1为预先固定的起始设置，未经测试集调参。

保持原全局ID+Triplet+0.1*CNL，额外对实际融合描述子训练ID+Triplet（系数1）。CNL及其记忆库仍在原全局768维空间；本轮有意固定这部分，避免同时改变CNL。DCM和融合监督一起构成本次待验证方案，只有与无RGB对照比较后，才能归因RGB信息的贡献。当前主干已有卷积stem，因此不能仅凭ViTMatte论文认定当前模型缺失细节。

固定：NightReID训练500IDs/15514图，查询1394图、图库26111图（含干扰身份）；SIE camera/view均关闭；相同SSL权重、CLS9–12可学习融合、120轮、20轮预热、batch64、SGD基础学习率0.0004、头部5倍、已对齐的分组余弦调度、CNL权重0.1、γ0.85、seed1234、原增强、无重排序。

三个完整配置：
- `configs/dcm_rgb.yml`：本次首个实验，只加DCM，无卷积Neck、无窗口注意力、无全局引导池化。
- `configs/dcm_no_rgb.yml`：保持DCM结构、参数、读出与额外监督，将独立RGB分支和末级RGB跳接输入置零。ViT仍看到正常原图。用于区分高分辨率RGB信息与额外变换/监督。
- `configs/baseline.yml`：关闭DCM与融合监督，恢复原全局路径。已有同协议原基线best为mAP51.8313%、Rank-1 67.5036%，epoch110。必要时可独立重跑。

训练入口（原工程根目录、QREID环境）：
```bash
/root/miniconda3/envs/QREID/bin/python -u -m mldc_vitmatte_nightreid_cnl.train --config_file mldc_vitmatte_nightreid_cnl/configs/dcm_rgb.yml
```
其他对照只需替换配置文件名。各配置有独立输出目录。不会自动从旧收敛模型继续训练。

每10轮用完整查询/图库分别评估融合、全局与细节分支，保留逐查询AP/Rank-1。`evaluation.csv`记录相对同检查点全局路径的纠正/破坏数、AP增益/损失；这些是路径诊断，不是独立基线增益。训练后：
```bash
/root/miniconda3/envs/QREID/bin/python -m mldc_vitmatte_nightreid_cnl.compare_baseline --output /root/log/transreid/nightreid/mldc_vitmatte_nightreid_cnl/dcm_rgb_gamma085_alpha01_seed1234_schedule120
```
对照原独立训练的best与最后120轮，输出完整图库AP增益/损失及净正确查询数。目标53.2/68.1仅作目标，不作已有成绩。单种子小提升不能证明稳定性。

本轮不迁移混合窗口注意力，因为它主要针对高分辨率抠图计算量，当前输入仅256×128且CLS全局读取必须保留。残差卷积Neck可另做单模块对照，暂不与DCM叠加。现有Night600的DCM+12层Neck是另一项组合实验，不能作为本实验NightReID结果。
