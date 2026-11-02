# mldc_feature_enhancement

## 实验思路

按独立入口选择Patch补偿、PFE、PFE范数控制、纯全局CNL或融合CLS引导CNN。从SSL初始化；新增CNN版本使用可学习后四层MLDC，其余配置沿用固定后四层均值。

## 模块与配置

| 目录 | 实验 | 配置性质 |
|---|---|---|
| [cnn_fused_cls_guided_learnable_no_cnl](variants/cnn_fused_cls_guided_learnable_no_cnl/README.md) | 可学习MLDC融合CLS引导CNN，无CNL | 当前配置 |
| [patch_residual_disabled](variants/patch_residual_disabled/README.md) | 方案 B 关闭增强对照 | 当前配置 |
| [patch_fen_safa_residual](variants/patch_fen_safa_residual/README.md) | 方案 B：Patch FEN/SAFA | 当前配置 |
| [global_cls_cross_camera_cnl](variants/global_cls_cross_camera_cnl/README.md) | 纯 MLDC＋监督式跨相机 CNL | 当前配置 |
| [global_cls_without_cnl](variants/global_cls_without_cnl/README.md) | 纯 MLDC：CNL=0 对照 | 当前配置 |
| [legacy_cnn_injection_block_eight](variants/legacy_cnn_injection_block_eight/README.md) | CPPI 仅L8注入 | 旧配置保留 |
| [global_cls_without_cnl_alias](variants/global_cls_without_cnl_alias/README.md) | 纯 MLDC：CNL=0 对照 | 当前配置 |
| [legacy_vit_pyramid_injection](variants/legacy_vit_pyramid_injection/README.md) | CPPI ViT金字塔来源 | 旧配置保留 |
| [progressive_patch_enhancement](variants/progressive_patch_enhancement/README.md) | PFE 原版 | 当前配置 |
| [progressive_enhancement_disabled](variants/progressive_enhancement_disabled/README.md) | PFE 关闭模块对照 | 当前配置 |
| [progressive_enhancement_norm_five_percent](variants/progressive_enhancement_norm_five_percent/README.md) | PFE v2：5%实际增量 | 当前配置 |

每个配置目录包含独立的 `config.yml` 与 README；不同配置的输出按语义名称隔离。

## 实验流程

CNN版本由 `train_cnn.py` 启动：可学习融合CLS9–12 → 引导独立CNN读出 → 双分支ID/Triplet监督 → 两路BN及L2后拼接。CNL版本仍由 `train_cppi.py` 启动，仅训练全局分支。

准备数据和初始化 → 选择配置 → 使用对应训练入口 → 保存配置、指标与检查点 → 使用同一读出协议评估。

## 运行与结果

具体命令、训练参数和结果路径见各配置README。历史模型兼容说明见 [原有方法细节](docs/method_details.md)。

## 参考文献

- [A³PFN](../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [FeatEnHancer](../REFERENCES.md#featenhancer)：https://github.com/khurramHashmi/FeatEnHancer
- [PFE-Net](../REFERENCES.md#pfe)：https://doi.org/10.1007/s44443-026-00927-9
- [SDCL](../REFERENCES.md#sdcl)：https://github.com/yangbincv/SDCL
- [ViT-CoMer](../REFERENCES.md#comer)：https://github.com/Traffic-X/ViT-CoMer
- [TransReID-SSL](../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

本包将上述模块用于Night600 ReID；项目特有的组合与对照设计在配置README单独说明。
