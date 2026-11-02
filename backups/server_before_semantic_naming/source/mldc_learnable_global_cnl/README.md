# mldc_learnable_global_cnl

2026-10-22：可学习融合保留，余弦学习率调度改为与固定等权CNL相同的逐参数组比例调度。BN和分类头5倍倍率在预热及最低学习率处均保留；融合logit仍为1倍。CNL当前配置和等价别名改用独立的`deep_four_uniform_cross_camera_cnl_fixed_equal_schedule`输出目录。旧检查点与历史指标保持原记录。

运行`python -m mldc_learnable_global_cnl.verify_schedule`核对全部120轮的调度轨迹；检查仅使用独立的少量参数，不训练模型。

## 实验思路

由相同SSL初始化ViT；第9—12层CLS以均匀起步的可学习Softmax权重融合，联合优化ID、Triplet和按配置开关的跨相机CNL。记忆库在成功更新优化器后更新；检索使用768维BN后全局描述子，不恢复优化器或采样器状态。

## 模块与配置

| 目录 | 实验 | 配置性质 |
|---|---|---|
| [legacy_random_start_config](variants/legacy_random_start_config/README.md) | 旧随机初始化配置（当前入口不支持） | 历史配置 |
| [deep_four_uniform_cnl_alias](variants/deep_four_uniform_cnl_alias/README.md) | 可学习deep4＋CNL等价配置别名 | 当前配置 |
| [deep_four_uniform_cross_camera_cnl](variants/deep_four_uniform_cross_camera_cnl/README.md) | 可学习deep4＋跨相机CNL | 当前配置 |
| [deep_four_uniform_without_cnl](variants/deep_four_uniform_without_cnl/README.md) | 可学习deep4：关闭CNL对照 | 当前配置 |

每个配置目录包含独立的 `config.yml` 与 README；不同配置的输出按语义名称隔离。

## 实验流程

由相同SSL初始化ViT；第9—12层CLS以均匀起步的可学习Softmax权重融合，联合优化ID、Triplet和按配置开关的跨相机CNL。记忆库在成功更新优化器后更新；检索使用768维BN后全局描述子，不恢复优化器或采样器状态。

准备数据和初始化 → 选择配置 → 使用对应训练入口 → 保存配置、指标与检查点 → 使用同一读出协议评估。

## 运行与结果

具体命令、训练参数和结果路径见各配置README。历史模型兼容说明见 [原有方法细节](docs/method_details.md)。

## 参考文献

- [A³PFN](../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [SDCL](../REFERENCES.md#sdcl)：https://github.com/yangbincv/SDCL
- [TransReID-SSL](../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

本包将上述模块用于Night600 ReID；项目特有的组合与对照设计在配置README单独说明。
