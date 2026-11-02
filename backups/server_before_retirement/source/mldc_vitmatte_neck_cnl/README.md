# 每个Transformer Block内的ViTMatte卷积Neck + DCM + MLDC/CNL

独立包：`mldc_vitmatte_neck_cnl`。从上一版`mldc_vitmatte_cnl`复制，保留其完整DCM分支、MLDC和CNL，只在本副本加入残差卷积Neck。原MLDC/CNL实验和上一版DCM实验保持独立。

当前只交付代码和不更新参数的验证，不启动训练。随机种子固定1234。

## 结构

```text
RGB (256 x 128)
    ├─ 原卷积Stem + Patch Embedding -> 16 x 8 patch tokens + CLS
    │     └─ Transformer Block 1 ... 12，每个Block内部：
    │           原LN/MHSA + 残差 -> 原LN/MLP + 残差
    │           分离CLS和patch tokens
    │           patch -> 二维16 x 8网格
    │                 -> [1x1 768→384 + LN + GELU
    │                     3x3 384→384 + LN + GELU
    │                     1x1 384→768 + LN(零初始化)] + 残差
    │                 -> 恢复patch tokens，与原CLS拼接
    │           -> 下一个Block（末层patch进入DCM）
    │
    │     ├─ CLS9/10/11/12 -> 可学习MLDC -> BN -> 768D全局单位向量
    │     └─ 第12层patch网格 ----------------------------┐
    └─ ConvStream (48/96/192，1/2、1/4、1/8分辨率)       │
          -> 四级融合 (256/128/64/32，全分辨率输出) <-----┘
          -> MLDC原始向量引导池化 -> 256D细节单位向量

最终描述子 = [sqrt(0.9)·全局768D, sqrt(0.1)·细节256D] = 1024D
损失 = 原全局ID/Triplet + 融合ID/Triplet + 0.1·原全局768D CNL
```

卷积Neck只处理patch tokens，CLS单独保留。默认遵循原文的Block内顺序：自注意力、MLP之后再做残差卷积。补充后的patch由下一个Block的自注意力读入CLS；第12层Neck的输出patch进入DCM。第12层Neck不会直接改变第12层CLS，这不妨碍前11层Neck通过后续注意力影响MLDC。

## 和ViTMatte原文的关系

Neck结构、dim/2瓶颈宽度、LN/GELU、MSRA卷积初始化及末级LN零初始化来自官方`ResBottleneckBlock`。固定版本：`8cd7ef068380977c3962c4cb733cb1fe7f2241a5`，文件`modeling/backbone/vit.py`，MIT许可保留在`licenses/ViTMatte-MIT.txt`。参考原文件在`references/vitmatte_vit.py.txt`。移除了Detectron2/fvcore依赖，改用等价PyTorch基础层。

原文在Transformer组末、MLP之后接Neck；本版默认同样放在MLP之后，按用户要求增加到全部12个Block。扩展的是插入密度，Block内部的位置沿用原文。第3/6/9/12层MLP后插入的组末配置也保留。原来的自注意力前方案移为独立消融配置，不能预先声称优于原位置。所有配置仍使用当前骨干的全局自注意力，不代表完整复现原文窗口/全局混合注意力。trimap、抠图头、抠图损失和ViTMatte模型权重不迁移。

零初始化让Neck初始为恒等映射，保留预训练行为。第一轮反向时末级LN先获得梯度；当其权重离开零后，上游卷积可以获得非零梯度。不能用初始恒等前向来证明训练后无效或有效。

## 参数和对照配置

每个Neck新增1,920,000个参数；12个Neck合计23,040,000个。分组4个Neck为7,680,000个。该数量不含上一版已有的DCM分支。

| 配置 | 卷积Neck | DCM | 用途 |
|---|---|---|---|
| `configs/neck_all.yml`（默认） | 全部12层，MLP后 | 引导池化 | 每Block方案，沿用原文Block内顺序 |
| `configs/neck_grouped.yml` | 第3/6/9/12层，MLP后 | 引导池化 | 组末插入策略对照 |
| `configs/neck_all_pre_attn.yml` | 全部12层，自注意力前 | 引导池化 | 固定层数，只比较插入位置 |
| `configs/dcm.yml` | 关闭 | 引导池化 | 上一版DCM对照 |
| `configs/baseline.yml` | 关闭 | 关闭 | 原MLDC/CNL对照 |
| `configs/dcm_mean.yml` | 关闭 | 均值池化 | 原DCM池化对照 |

`CONV_NECK_LAYERS`使用从1开始的层号；`CONV_NECK_PLACEMENT`支持`pre_attn`与`post_mlp`。可以独立切换层数和位置；配置项相同时才允许严格恢复本实验检查点。没有改变原120轮、batch64、20轮预热及已对齐的分组学习率调度；Neck卷积权重使用骨干学习率，归一化偏置沿用原bias规则。新增Neck使用独立1234 RNG上下文，保留原骨干、分类器、DCM权重及调用方随机流。

## 入口（当前未启动训练）

在服务器项目根目录，后续用户明确要求训练时可以使用：

```bash
python -m mldc_vitmatte_neck_cnl.train
python -m mldc_vitmatte_neck_cnl.train --config_file mldc_vitmatte_neck_cnl/configs/neck_grouped.yml
python -m mldc_vitmatte_neck_cnl.train --config_file mldc_vitmatte_neck_cnl/configs/dcm.yml
python -m mldc_vitmatte_neck_cnl.test --weight <本独立实验检查点>
```

所有配置使用新的独立输出目录；不会覆盖原实验。严格检查点包含Neck层号、插入位置和原DCM结构。旧全局检查点只能通过显式`load_global_checkpoint`做诊断性权重导入，不会作为新分支已训练的结果。

代码验证入口没有训练开关，也不调用优化器更新：

```bash
python -m mldc_vitmatte_neck_cnl.verify --report mldc_vitmatte_neck_cnl/verification_cpu.json
```

验证包含官方Neck计算等价、矩形patch网格、CLS旁路、两种插入位置的CLS读取时机、零初始化梯度、原SSL/DCM初始化一致、12层MLP后Neck前向覆盖、优化器参数包含、检查点严格恢复、旧全局权重导入、组末对照前向。回归对照依赖服务器现有上一版包和原SSL/全局检查点，训练实现本身不导入上一版包。小型合成模块只做反向检查，不更新参数；完整模型只做推理。当前未测量新的mAP/Rank-1，未验证CUDA batch64显存容量。
