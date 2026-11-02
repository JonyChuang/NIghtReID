# MLDC+CNL：原型引导的第9–12层CLS精炼

当前目录仍为 `mldc_convstream_texture_cnl`，版本为 `mldc_pad_multilevel_cls_20261029`。按用户最新要求在原目录修改，没有建立新的实验副本。

## 当前结构

- 一次ViT前向分别取第9、10、11、12块输出的CLS与对应patch，四组patch各128×768。
- 每个层分支只有一次直接多头交叉注意力。学生Q为本层CLS，教师Q为同一个身份原型；K、V都来自本层patch。
- 四层参数独立；同一层的师生共享Q/K/V/输出投影。注意力内部宽度256、8头，没有三层解码器，没有新增FFN。
- 学生局部聚合更新本层CLS：cS_l=c_l+W_O_l Attention(c_l,P_l)。四个更新后的CLS再进入原MLDC可学习加权聚合。
- 教师也分别读取这四组patch，以相同方式得到四个教师精炼CLS，再用同一套MLDC权重聚合为一个教师描述子。
- 只有一张500×768身份原型表。教师最终聚合描述子经共享BNNeck和L2后执行原文AMU；不建立四张原型表，也不把四层当成四个部位原型。
- 最终只有一个768维检索身份空间。ID、Triplet、CNL保留原训练位置，原型CE也在该最终空间计算。

每层的注意力图形状为B×8×128，合并为B×4×8×128。逐位置计算KL(AT.detach() || AS)，再平均批次、四层和头。蒸馏系数5。日志分别记录L9、L10、L11、L12的KL与熵。

## 保留设置

单身份原型，论文AMU、gamma=0.05、成功优化器更新后写回、5轮原型构建，第6轮起有效教师参与。原SSL初始化、NightReID、seed1234、120轮、关闭SIE、0.1 CNL、原20轮学习率预热继续保留。

原型只提供Q；教师的V来自当前图像，没有将原型查询直接加进视觉聚合。新增注意力不复用ViT原生注意力投影。ViT内部原有FFN保持原样。

各层输出投影零初始化，初始模型与原MLDC+CNL逐值一致。推理保留全部四个CLS精炼分支、原MLDC和BNNeck，不访问身份原型。

## 配置和运行

沿用历史配置文件名 `configs/pad_p12_nightreid.yml`，其内容已改为第9–12层。该文件名不再表示只使用P12。

~~~bash
cd /root/TransReID-SSL-main/transreid_pytorch
/root/miniconda3/envs/QREID/bin/python -u -m mldc_convstream_texture_cnl.train --config_file mldc_convstream_texture_cnl/configs/pad_p12_nightreid.yml
~~~

新结果目录：`/root/log/transreid/nightreid/mldc_convstream_texture_cnl/pad/pad_multilevel_cls_singleproto_seed1234_120`。这是结果目录，用于保护已运行日志，不是新代码副本。

修改源码不会改变已经启动的Python训练进程使用的旧结构。本次未停止、重启或切换旧训练；新结构只在下次启动时加载。旧检查点不兼容新结构，应按入口从原SSL初始化新训练。

验证入口仍为 `python -m mldc_convstream_texture_cnl.verify`。验证覆盖真实层输出、直接注意力公式、分支隔离、四层CLS梯度、单原型AMU和推理导出。详见[方法说明](docs/PAD_METHOD.md)。
