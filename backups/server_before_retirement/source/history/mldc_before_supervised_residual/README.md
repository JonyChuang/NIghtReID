# mldc_before_supervised_residual

## 实验思路

按目录名保留修改前源码；未额外推断方法差异。

## 实验流程与使用方式

这是修改前源码或部分文件快照。它保留对应阶段的模块和配置，用于恢复、比较旧实现；不直接把备份中的入口当作当前训练入口。恢复时结合 [目录映射](../../PATH_MAPPING.md) 恢复配套文件，再使用对应配置训练或评估。

该快照只有修改前文件，未记录独立训练流程和独立论文。涉及的主模块来源如下；不把缺失说明补写成已完成实验。

## 参考文献

- [A³PFN](../../REFERENCES.md#a3pfn)：https://github.com/zhguoqing/A3PFN
- [TransReID-SSL](../../REFERENCES.md#ssl)：https://github.com/damo-cv/TransReID-SSL
