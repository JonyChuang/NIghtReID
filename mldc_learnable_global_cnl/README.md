# mldc_learnable_global_cnl

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
