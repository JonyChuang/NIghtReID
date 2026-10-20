# mldc_boq_local_readout

## 实验思路

全局使用可学习后四层MLDC；从第4/6/8层patch提取BoQ多查询局部描述子，形成全局残差，并对局部读出提供辅助监督。所有活动模块联合训练。

## 模块与配置

| 目录 | 实验 | 配置性质 |
|---|---|---|
| [middle_patch_multi_query_boq](variants/middle_patch_multi_query_boq/README.md) | MLDC＋BoQ 多查询联合 v2 | 当前配置 |

每个配置目录包含独立的 `config.yml` 与 README；不同配置的输出按语义名称隔离。

## 实验流程

全局使用可学习后四层MLDC；从第4/6/8层patch提取BoQ多查询局部描述子，形成全局残差，并对局部读出提供辅助监督。所有活动模块联合训练。

准备数据和初始化 → 选择配置 → 使用对应训练入口 → 保存配置、指标与检查点 → 使用同一读出协议评估。

## 运行与结果

具体命令、训练参数和结果路径见各配置README。历史模型兼容说明见 [原有方法细节](docs/method_details.md)。

## 参考文献

- [BoQ](../REFERENCES.md#boq)：https://github.com/amaralibey/Bag-of-Queries
- [A³PFN](../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [TransReID-SSL](../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

本包将上述模块用于Night600 ReID；项目特有的组合与对照设计在配置README单独说明。
