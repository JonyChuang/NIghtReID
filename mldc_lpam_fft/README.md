# mldc_lpam_fft

## 实验思路

全局始终取第9—12层原始CLS均值；LPAM读取第3/6/9/12层并执行FFT路径；分别BN后直接拼接四路描述子进行检索。

## 模块与配置

| 目录 | 实验 | 配置性质 |
|---|---|---|
| [fixed_global_lpam_fft_concat](variants/fixed_global_lpam_fft_concat/README.md) | LPAM＋FFT v4 | 当前配置 |

每个配置目录包含独立的 `config.yml` 与 README；不同配置的输出按语义名称隔离。

## 实验流程

全局始终取第9—12层原始CLS均值；LPAM读取第3/6/9/12层并执行FFT路径；分别BN后直接拼接四路描述子进行检索。

准备数据和初始化 → 选择配置 → 使用对应训练入口 → 保存配置、指标与检查点 → 使用同一读出协议评估。

## 运行与结果

具体命令、训练参数和结果路径见各配置README。历史模型兼容说明见 [原有方法细节](docs/method_details.md)。

## 参考文献

- [A³PFN](../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [TransReID-SSL](../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL

本包将上述模块用于Night600 ReID；项目特有的组合与对照设计在配置README单独说明。
