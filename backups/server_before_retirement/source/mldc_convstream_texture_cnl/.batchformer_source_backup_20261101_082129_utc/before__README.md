# MLDC+CNL：阈值参考集合实验

当前代码仍在 mldc_convstream_texture_cnl。新默认配置为 configs/cnl_reference_nightreid.yml。

## 本轮方案

检索使用原始CLS9–12的MLDC可学习融合、原BN768与L2；原ID、Triplet和0.1 CNL保留。PAD与PEM关闭，不增加参数、身份原型、注意力蒸馏或额外分支。

CNL先构造跨相机同身份正样本、其他身份负样本。新规则只在这两类候选中计算最大相似度，再按原gamma=0.9筛选正邻居；同身份同相机图片不再抬高筛选阈值。负样本集合、损失归一化和记忆更新保持原样。

原规则仍可用 SOLVER.CNL_THRESHOLD_REFERENCE all_nonself 显式复现；其为旧配置的默认值。新配置使用 loss_candidates。

## 训练

~~~bash
cd /root/TransReID-SSL-main/transreid_pytorch
/root/miniconda3/envs/QREID/bin/python -u -m mldc_convstream_texture_cnl.train --config_file mldc_convstream_texture_cnl/configs/cnl_reference_nightreid.yml
~~~

从原SSL预训练权重重新训练120轮，seed1234、学习率0.0004、20轮预热、batch64、每身份4张图、SIE关闭。每10轮评估，保存最佳及第40/80/120轮检查点。

结果目录：/root/log/transreid/nightreid/mldc_convstream_texture_cnl/cnl_reference/loss_candidates_seed1234_120。入口要求输出目录为空，不覆盖历史结果。创建新结果目录不等于建立新代码副本。

## 诊断

training.csv同时记录实际、旧参考及候选参考的空邻居比例，以及新旧正样本覆盖率、参考最大值变化比例、恢复正邻居的样本比例。新旧规则的影子统计都基于当前同一批特征，不是另一个模型的训练结果。

run_metadata.json和检查点保存实际threshold_reference。原PAD与PEM配置仍可显式选择，其默认CNL参考保持all_nonself；历史方法说明见docs/PAD_METHOD.md。

本次部署不会自动启动训练。
