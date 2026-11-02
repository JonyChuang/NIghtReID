# mldc_patch_grm

## 实验思路

加载已训练deep4初始化，固定全局骨干、融合和原BN；前5轮直接训练局部读出，随后启用GRM残差及最终特征监督，评估冻结全局、局部和融合读出。

## 模块与配置

| 目录 | 实验 | 配置性质 |
|---|---|---|
| [frozen_global_patch_grm](variants/frozen_global_patch_grm/README.md) | Patch直接监督＋GRM v2 | 当前配置 |

每个配置目录包含独立的 `config.yml` 与 README；不同配置的输出按语义名称隔离。

## 实验流程

加载已训练deep4初始化，固定全局骨干、融合和原BN；前5轮直接训练局部读出，随后启用GRM残差及最终特征监督，评估冻结全局、局部和融合读出。

准备数据和初始化 → 选择配置 → 使用对应训练入口 → 保存配置、指标与检查点 → 使用同一读出协议评估。

## 运行与结果

具体命令、训练参数和结果路径见各配置README。历史模型兼容说明见 [原有方法细节](docs/method_details.md)。

## 参考文献

- [A³PFN](../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [TAR](../REFERENCES.md#tar)：https://github.com/VectorYangYiStar/TAR
- [TransReID-SSL](../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

本包将上述模块用于Night600 ReID；项目特有的组合与对照设计在配置README单独说明。
