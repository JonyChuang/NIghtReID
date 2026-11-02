# patch_compensation_zero_initialized

## 实验思路

MLDC跨层局部补偿实验（2026-10-16）

## 实验流程与使用方式

这是修改前源码或部分文件快照。它保留对应阶段的模块和配置，用于恢复、比较旧实现；不直接把备份中的入口当作当前训练入口。恢复时结合 [目录映射](../../PATH_MAPPING.md) 恢复配套文件，再使用对应配置训练或评估。

原始实验流程、思路和已记录参数保留在 [原说明](docs/original_notes.md)。

## 参考文献

- [A³PFN](../../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [TAR](../../REFERENCES.md#tar)：https://github.com/VectorYangYiStar/TAR
- [TransReID-SSL](../../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL
