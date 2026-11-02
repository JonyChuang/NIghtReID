# 输入端FeatEnHancer方案A＋可学习MLDC＋CNL

2026-10-22建立的独立实验包，源自已对齐等权学习率调度的`mldc_learnable_global_cnl`。原包和已完成结果不修改。

图像经原ReID预处理及一次归一化后，先进入原版FeatEnHancer方案A，再进入SSL ViT-Base；第9–12层CLS以均匀起步的可学习Softmax权重融合，BN后L2特征用于CNL及检索。最终维度768。

训练目标为ID＋soft-margin Triplet＋0.1×有监督单空间跨摄像头CNL。主干从`pretrain_model/vit_base_ics_cfs_lup.pth`重新初始化；增强器卷积权重按方案A的normal(0,0.02)初始化、偏置使用原默认值。不从旧方案A或已训练CNL检查点继续训练。增强器、主干、融合权重和BN/分类头从第1轮共同学习。

配置`configs/night600.yml`：Night600，seed1234，256×128，batch64，120轮，SGD基础学习率0.0004，预热20轮；头部倍率5、融合logit倍率1。学习率采用与固定等权CNL相同的逐参数组缩放，保留预热和最低学习率处的倍率。验证batch64，每10轮验证，按mAP保存最佳模型。

在原服务器工程根目录运行：

```bash
/root/miniconda3/envs/QREID/bin/python -u mldc_featenhancer_learnable_cnl/train.py
```

输出：`/root/log/transreid/night600/mldc_featenhancer_learnable_cnl/input_a_learnable_cnl01_matched_schedule_120`。

检查点保存增强器、ViT、可学习融合、BN、分类头、CNL记忆及完整配置，标识`featenhancer_input_learnable_cnl_20261022`。测试入口：

```bash
/root/miniconda3/envs/QREID/bin/python mldc_featenhancer_learnable_cnl/test.py TEST.WEIGHT /absolute/path/transformer_best.pth
```

原有固定正对归因与逐层融合权重可视化随模型完整前向，包含输入增强器。图中响应是patch embedding处的匹配分数归因，不能视为图像增强质量指标。

历史方案A的最佳mAP10.8028%来自固定等权＋ID/Triplet，历史可学习CNL的11.7090%来自旧调度；都作为历史参考，不能直接当成本组合的单因素消融。当前对照配置为已对齐调度的可学习CNL；若其结果尚未完成，需后续补齐后再判断输入增强器的独立增益。

FeatEnHancer来源、原提交和初始化见`provenance.json`，许可证见`FEATENHANCER_LICENSE.txt`。
