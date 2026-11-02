# MLDC+CNL：P12原型注意力蒸馏

本副本按用户选择由ConvStream纹理蒸馏改为PAD适配，目录名继续使用`mldc_convstream_texture_cnl`。当前版本标识为`mldc_pad_p12_20261029`。

- K、V只来自最后一层P12，共128个空间token。
- 学生以当前图像的MLDC表示初始化查询，教师以训练身份原型初始化查询。
- 3层MHCA＋FFN逐层更新查询，每层8头、隐藏维度256。教师／学生对应层共享参数，各层之间使用独立参数。
- 注意力逐层逐头计算KL后平均。学生输出进入最终768维检索描述子，ID、Triplet、0.1 CNL约束最终同一路径。
- 单身份原型、学生热启动、教师EMA更新；没有独立RGB CNN教师。测试只使用学生头，不访问身份原型。

详细结构、损失和记忆更新规则见[PAD_METHOD.md](docs/PAD_METHOD.md)。新增损失系数尚未通过完整NightReID实验验证。

## 当前配置

| 配置 | 内容 |
|---|---|
| `configs/pad_p12_nightreid.yml` | P12＋原型教师＋逐层逐头蒸馏，默认入口 |
| `configs/pad_prototype_no_distill_nightreid.yml` | 同结构原型教师，不使用注意力KL |
| `configs/pad_head_only_nightreid.yml` | 仅学生注意力头及其身份监督 |
| `configs/baseline_nightreid.yml` | 关闭PAD，原MLDC+CNL对照 |

原型前5轮由学生建立，第6轮起按有效支持条件启用教师；注意力权重随后10轮增加至设定值。原学习率预热仍为20轮。数据集NightReID、seed1234、120轮、SIE关闭、CNL权重0.1及阈值0.9保持一致。

## 验证

从项目根目录执行：

```powershell
python -m mldc_convstream_texture_cnl.verify
```

该入口验证真实训练图像的前向、梯度、记忆更新、序列化和推理导出，不进行优化器训练。可通过`--baseline`、`--train-dir`、`--report`指定验证资源。

## 显式训练和评估入口

下面的命令是使用说明。本次代码修改不会自动执行训练。

```powershell
python -m mldc_convstream_texture_cnl.train --config_file mldc_convstream_texture_cnl/configs/pad_p12_nightreid.yml
python -m mldc_convstream_texture_cnl.test --config_file mldc_convstream_texture_cnl/configs/pad_p12_nightreid.yml TEST.WEIGHT /path/to/pad_checkpoint.pth
```

新结果写入本包独立的`pad/`结果子目录；非空结果目录会拒绝覆盖。旧ConvStream日志与检查点不改动。新加载器按架构标识严格区分PAD与旧ConvStream模型。

`export_inference.py`移除训练记忆与优化器状态，保留学生注意力头；导出后仍可通过同一PAD模型评估。

旧`train_teacher.py`及两份旧流水线入口已停用，避免误启动之前停止的ConvStream实验。旧教师模块和旧YAML仍作为历史代码保留，不属于当前训练入口支持的配置。修改前完整源码另有备份，位置记录在本轮部署报告中。
