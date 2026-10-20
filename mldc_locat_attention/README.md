# mldc_locat_attention

## 实验思路

按配置控制GAug、读出LayerNorm和PRR；融合完整多层token后读取CLS，使用相同ID/Triplet和评估设置比较各组件。

## 模块与配置

| 目录 | 实验 | 配置性质 |
|---|---|---|
| [gaussian_attention_with_layernorm](variants/gaussian_attention_with_layernorm/README.md) | GAug＋LN | 当前配置 |
| [gaussian_attention_raw_cls](variants/gaussian_attention_raw_cls/README.md) | 仅 GAug | 当前配置 |
| [original_mldc_control](variants/original_mldc_control/README.md) | 原 MLDC 路径对照 | 当前配置 |
| [gaussian_attention_prr](variants/gaussian_attention_prr/README.md) | GAug＋LN＋PRR 主实验 | 当前配置 |
| [readout_layernorm_control](variants/readout_layernorm_control/README.md) | 仅 LN 对照 | 当前配置 |
| [patch_refinement_only](variants/patch_refinement_only/README.md) | LN＋PRR | 当前配置 |

每个配置目录包含独立的 `config.yml` 与 README；不同配置的输出按语义名称隔离。

## 实验流程

按配置控制GAug、读出LayerNorm和PRR；融合完整多层token后读取CLS，使用相同ID/Triplet和评估设置比较各组件。

准备数据和初始化 → 选择配置 → 使用对应训练入口 → 保存配置、指标与检查点 → 使用同一读出协议评估。

## 运行与结果

具体命令、训练参数和结果路径见各配置README。历史模型兼容说明见 [原有方法细节](docs/method_details.md)。

## 参考文献

- [LocAtViT](../REFERENCES.md#locat)：https://github.com/sinahmr/LocAtViT
- [A³PFN](../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [TransReID-SSL](../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

本包将上述模块用于Night600 ReID；项目特有的组合与对照设计在配置README单独说明。
