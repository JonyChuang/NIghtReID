# mldc_last_cls_dual_supervision

## 实验思路

构造后四层固定均值和末层LayerNorm后CLS两个分支；以0.7/0.3监督；推理时两路分别归一化后等能量拼接，得到1536维描述子。

## 模块与配置

| 目录 | 实验 | 配置性质 |
|---|---|---|
| [fixed_global_last_cls_unit_concat](variants/fixed_global_last_cls_unit_concat/README.md) | 固定deep4＋末层双监督 v2 | 当前配置 |

每个配置目录包含独立的 `config.yml` 与 README；不同配置的输出按语义名称隔离。

## 实验流程

构造后四层固定均值和末层LayerNorm后CLS两个分支；以0.7/0.3监督；推理时两路分别归一化后等能量拼接，得到1536维描述子。

准备数据和初始化 → 选择配置 → 使用对应训练入口 → 保存配置、指标与检查点 → 使用同一读出协议评估。

## 运行与结果

具体命令、训练参数和结果路径见各配置README。历史模型兼容说明见 [原有方法细节](docs/method_details.md)。

## 参考文献

- [A³PFN](../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [TransReID-SSL](../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

本包将上述模块用于Night600 ReID；项目特有的组合与对照设计在配置README单独说明。
