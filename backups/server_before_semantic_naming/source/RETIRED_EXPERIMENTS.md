# 已归档实验

源码备份：[5fa1dd7](https://github.com/JonyChuang/NIghtReID/tree/5fa1dd7d72a61ce7a2dfce7ef40d321fadf01929/backups/server_before_retirement)。

本轮按用户选定的05—16系列删除14个独立代码包、相关历史源码、专属结果和权重，并清理三个共用包中继承的纹理蒸馏文件。

删除的独立包：

- `mldc_patch_grm`
- `mldc_boq_local_readout`
- `mldc_tar_refinement`
- `mldc_mufasa_slots`
- `mldc_locat_attention`
- `mldc_feature_enhancement`
- `mldc_featenhancer_learnable_cnl`
- `mldc_localvit_cnl`
- `mldc_convpass_cnl`
- `mldc_vitmatte_cnl`
- `mldc_vitmatte_neck_cnl`
- `mldc_vitmatte_nightreid_cnl`
- `mldc_texture_distill_cnl`
- `mldc_convnext_relation_cnl`

保留边界：

- 保留MLDC/CNL、信息保持、PAD、NFC、PEM、BatchFormer/V2、MVI及未选中的旧实验。
- 原FeatEnHancer父目录中的mldc_global_cnl01_fixed_equal_120为纯MLDC/CNL基线，继续保留。
- Patch/GRM初始化与MLDC基线共享实体，删除初始化路径不删除基线。
- seed1234读出权重、旧身份和验证名单继续保留。
- 正在运行的BatchFormer任务未停止。
- 通用SSL/ConvNeXt预训练资产暂保留；历史混合源码压缩包不当作活动入口。

24份旧指标和元数据摘要已随源码备份保存在GitHub。训练权重不在GitHub备份中。

## 进一步清理的历史方案

早期Patch→CLS、GLTrans、QNorm/QCAT/DCE独立代码和配置、旧Teacher/Student权重、专属可视化及LPAM旧结果摘要已移除。源码与小型结果摘要已在[GitHub归档](https://github.com/JonyChuang/NIghtReID/tree/84ffbc142604cb91facf2e36c29c032e400b5be5/backups/server_before_retirement)。权重不包含在归档中。当前TransReID-SSL+CNL验证仍引用的共享基础模型文件保留，不作为旧实验训练入口。

## MVI Cross清理

MVI Cross独立包、配置、旧结果及专属源码副本已删除。旧训练权重此前已移除，本轮没有剩余模型权重。源码和指标摘要保存于[GitHub备份](https://github.com/JonyChuang/NIghtReID/tree/094c274d8e4191a0216c7f0e3ca0d811bbe8c03b/backups/server_before_retirement)。当前其余9个独立实验包继续保留。
