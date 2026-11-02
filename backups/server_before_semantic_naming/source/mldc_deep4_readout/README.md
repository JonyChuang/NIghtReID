# mldc_deep4_readout

## 实验思路

从SSL初始化完整ViT，仅融合第9—12层原始CLS；随机或均匀初始化层权重，随后与骨干共同学习，保存层权重与检索指标。

## 模块与配置

| 目录 | 实验 | 配置性质 |
|---|---|---|
| [deep_four_random_learnable](variants/deep_four_random_learnable/README.md) | 后四层随机起步（独立副本） | 当前配置 |
| [deep_four_uniform_learnable](variants/deep_four_uniform_learnable/README.md) | 后四层等权起步（独立默认） | 当前配置 |

每个配置目录包含独立的 `config.yml` 与 README；不同配置的输出按语义名称隔离。

## 实验流程

从SSL初始化完整ViT，仅融合第9—12层原始CLS；随机或均匀初始化层权重，随后与骨干共同学习，保存层权重与检索指标。

准备数据和初始化 → 选择配置 → 使用对应训练入口 → 保存配置、指标与检查点 → 使用同一读出协议评估。

## 运行与结果

具体命令、训练参数和结果路径见各配置README。历史模型兼容说明见 [原有方法细节](docs/method_details.md)。

## 参考文献

- [A³PFN](../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [TransReID-SSL](../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

本包将上述模块用于Night600 ReID；项目特有的组合与对照设计在配置README单独说明。
