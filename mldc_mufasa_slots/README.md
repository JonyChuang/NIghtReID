# mldc_mufasa_slots

## 实验思路

从SSL开始，读取后四层patch执行Slot Attention、跨层匹配和融合；MLDC引导整体slot残差，门控融合，并加入残差度量及patch重建辅助目标。

## 模块与配置

| 目录 | 实验 | 配置性质 |
|---|---|---|
| [slot_residual_joint_bnneck](variants/slot_residual_joint_bnneck/README.md) | 联合训练 BNNeck 修正版 v3 | 当前配置 |

每个配置目录包含独立的 `config.yml` 与 README；不同配置的输出按语义名称隔离。

## 实验流程

从SSL开始，读取后四层patch执行Slot Attention、跨层匹配和融合；MLDC引导整体slot残差，门控融合，并加入残差度量及patch重建辅助目标。

准备数据和初始化 → 选择配置 → 使用对应训练入口 → 保存配置、指标与检查点 → 使用同一读出协议评估。

## 运行与结果

具体命令、训练参数和结果路径见各配置README。历史模型兼容说明见 [原有方法细节](docs/method_details.md)。

## 参考文献

- [MUFASA](../REFERENCES.md#mufasa)：https://github.com/visinf/mufasa
- [A³PFN](../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [TransReID-SSL](../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

本包将上述模块用于Night600 ReID；项目特有的组合与对照设计在配置README单独说明。
