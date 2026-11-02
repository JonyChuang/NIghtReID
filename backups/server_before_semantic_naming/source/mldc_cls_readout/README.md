# mldc_cls_readout

## 实验思路

同一SSL初始化ViT，收集原始CLS，按配置执行多层加权或末层对照；使用ID和Triplet联合训练，再提取BN前/后描述子评估。

## 模块与配置

| 目录 | 实验 | 配置性质 |
|---|---|---|
| [last_raw_cls_control](variants/last_raw_cls_control/README.md) | 末层原始 CLS 对照 | 当前配置 |
| [all_cls_random_initialization](variants/all_cls_random_initialization/README.md) | 12层随机初始化 MLDC | 当前配置 |
| [deep_four_uniform_learnable](variants/deep_four_uniform_learnable/README.md) | 后四层等权起步（主包） | 当前配置 |
| [deep_cls_prior_initialization](variants/deep_cls_prior_initialization/README.md) | 深层先验初始化 | 当前配置 |
| [all_cls_raw_residual_readout](variants/all_cls_raw_residual_readout/README.md) | 随机初始化＋原始残差检索 | 当前配置 |
| [all_cls_residual_supervision](variants/all_cls_residual_supervision/README.md) | 残差特征直接监督 | 当前配置 |
| [all_cls_uniform_initialization](variants/all_cls_uniform_initialization/README.md) | 12层等权初始化 | 当前配置 |
| [all_cls_uniform_fusion_lr_five](variants/all_cls_uniform_fusion_lr_five/README.md) | 等权初始化＋融合LR×5 | 当前配置 |

每个配置目录包含独立的 `config.yml` 与 README；不同配置的输出按语义名称隔离。

## 实验流程

同一SSL初始化ViT，收集原始CLS，按配置执行多层加权或末层对照；使用ID和Triplet联合训练，再提取BN前/后描述子评估。

准备数据和初始化 → 选择配置 → 使用对应训练入口 → 保存配置、指标与检查点 → 使用同一读出协议评估。

## 运行与结果

具体命令、训练参数和结果路径见各配置README。历史模型兼容说明见 [原有方法细节](docs/method_details.md)。

## 参考文献

- [A³PFN](../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [TransReID-SSL](../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

本包将上述模块用于Night600 ReID；项目特有的组合与对照设计在配置README单独说明。
