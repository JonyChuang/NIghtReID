# NightReID 服务器实验导航

当前活动源码按实验包归类。配置存在不代表已完成训练；历史权重须与训练时的源码和协议对应。

## 当前保留的代码包

| 目录 | 研究内容 | 活动配置数 |
|---|---|---:|
| [mldc_batchformerv2_cnl](mldc_batchformerv2_cnl/README.md) | 第8块后逐token的BatchFormerV2 | 13 |
| [mldc_cls_readout](mldc_cls_readout/README.md) | 多层CLS融合、末层对照及残差监督 | 8 |
| [mldc_cnl_isolation](mldc_cnl_isolation/README.md) | 末层CLS有无CNL隔离对照 | 2 |
| [mldc_convstream_texture_cnl](mldc_convstream_texture_cnl/README.md) | MLDC/CNL、BatchFormer V1、PAD、NFC、PEM；旧纹理蒸馏已归档 | 17 |
| [mldc_deep4_readout](mldc_deep4_readout/README.md) | 后四层CLS融合，随机或均匀初始化 | 2 |
| [mldc_information_preservation](mldc_information_preservation/README.md) | 信息保持、BN比例及共享CNL | 5 |
| [mldc_learnable_cnl_nightreid](mldc_learnable_cnl_nightreid/README.md) | NightReID可学习MLDC＋CNL及参数对照 | 4 |
| [mldc_learnable_global_cnl](mldc_learnable_global_cnl/README.md) | Night600可学习MLDC＋跨相机CNL | 4 |
| [transreid_ssl_cnl](transreid_ssl_cnl/README.md) | 原版TransReID-SSL＋CNL | 1 |

## 共享基础代码

根目录旧Patch→CLS、GLTrans及QNorm/QCAT/DCE独立入口和配置已移除；model和processor中仍被当前验证代码引用的基础实现保留。`experiment_variants`仅剩历史数据集配置，不是独立模型方案。

## 已归档方案

用户选定的05—16系列已从活动源码移除：Patch/GRM、BoQ、TAR、MUFASA、LocAt、Patch/CNN增强、FeatEnHancer、LocalViT、Convpass、ViTMatte、纹理教师/蒸馏、ConvNeXt关系蒸馏。

完整源码与配置备份：[5fa1dd7](https://github.com/JonyChuang/NIghtReID/tree/5fa1dd7d72a61ce7a2dfce7ef40d321fadf01929/backups/server_before_retirement)。备份不含模型权重、数据集或图片特征；已删除的训练权重不能仅从此源码快照恢复。

[归档说明](RETIRED_EXPERIMENTS.md)记录删除范围与保留边界。共用包名称暂保留，以兼容保留实验的入口和检查点。

## 环境与索引

- [环境与运行入口](SETUP.md)
- [当前配置索引](experiment_index.csv)
- [参考文献与模块来源](REFERENCES.md)
- [历史路径映射](PATH_MAPPING.md)（历史记录，不表示路径仍存在）

数据集、SSL初始化、当前研究所用模型和seed1234读出/旧身份名单继续保留。

旧Patch→CLS、GLTrans、QNorm/QCAT/DCE和LPAM历史结果已进一步清理，源码及结果摘要见[GitHub归档](https://github.com/JonyChuang/NIghtReID/tree/84ffbc142604cb91facf2e36c29c032e400b5be5/backups/server_before_retirement)。

MVI Cross已按用户要求归档并删除，当前保留9个独立实验包。源码与旧指标摘要见[GitHub备份](https://github.com/JonyChuang/NIghtReID/tree/094c274d8e4191a0216c7f0e3ca0d811bbe8c03b/backups/server_before_retirement)。
