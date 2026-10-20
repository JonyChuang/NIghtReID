# ffvt_token_fusion

## 实验思路

原始ViT在第3/6/9/11层按MAWS各选12个patch，组成融合序列；原全局末层和独立融合末层分别监督，推理拼接两路BN后特征。

## 模块与配置

| 目录 | 实验 | 配置性质 |
|---|---|---|
| [maws_global_and_fused_tokens](variants/maws_global_and_fused_tokens/README.md) | 跨层 Token 融合双分支 | 当前配置 |

每个配置目录包含独立的 `config.yml` 与 README；不同配置的输出按语义名称隔离。

## 实验流程

原始ViT在第3/6/9/11层按MAWS各选12个patch，组成融合序列；原全局末层和独立融合末层分别监督，推理拼接两路BN后特征。

准备数据和初始化 → 选择配置 → 使用对应训练入口 → 保存配置、指标与检查点 → 使用同一读出协议评估。

## 运行与结果

具体命令、训练参数和结果路径见各配置README。历史模型兼容说明见 [原有方法细节](docs/method_details.md)。

## 参考文献

- [FFVT](../REFERENCES.md#ffvt)：https://github.com/Markin-Wang/FFVT
- [TransReID-SSL](../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

本包将上述模块用于Night600 ReID；项目特有的组合与对照设计在配置README单独说明。
