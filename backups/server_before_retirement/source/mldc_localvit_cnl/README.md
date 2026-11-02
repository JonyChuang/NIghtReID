# NightReID：前四层LocalViT FFN + MLDC + CNL

独立包：`mldc_localvit_cnl`。仅替换第1、2、3、4个Transformer Block的FFN；第5–12层保持原FFN。后四层CLS仍经可学习MLDC得到768维描述子，沿用原ID、Triplet及全局CNL。当前只完成代码和零优化器更新验证，没有启动训练。

## 模型结构

```text
RGB 256×128
  -> 原卷积Stem / Patch Embedding
  -> 16×8 patch tokens + CLS
  -> Block 1–4：
       LN -> MHSA -> 原残差
          ├─ CLS：绕过整个局部FFN ---------------------┐
          └─ patch：16×8二维网格                     │
               -> 1×1 Conv 768→3072 + BN + h-swish   │
               -> 3×3 DWConv(groups=3072) + BN + h-swish
               -> SE (3072→4→3072)
               -> 1×1 Conv 3072→768 + BN
               -> patch残差相加 -> 恢复tokens --------┤
                                  拼接CLS和patch <---┘
  -> Block 5–12：原LN/MHSA及LN/MLP
  -> CLS9、10、11、12 -> 可学习MLDC -> BN -> 768维检索

损失：原ID + 原Triplet + 0.1 × 全局CNL
```

与LocalViT官方一致，局部FFN的patch残差在模块内部完成，不再额外叠加原MLP、norm2或第二个DropPath。CLS通过后续Block的注意力读取局部信息。第5–12层继续保留CLS原MLP路径。这版没有DCM、ViTMatte卷积Neck、额外纹理描述子或辅助损失。

## 官方模块与预训练映射

来源：[LocalViT官方代码](https://github.com/ofsoundof/LocalViT)，固定版本见`provenance.json`。`model/localvit_official.py`中的五个官方类原样提取，保留MIT许可。局部FFN使用扩展比4、hs+se及SE reduction=768，使压缩后SE通道为4，沿用作者轻量模型的设置。原方形网格示例适配为256×128输入对应的16×8网格。

先按原模型加载`pretrain_model/vit_base_ics_cfs_lup.pth`，随后替换前四层：

- 将每层`mlp.fc1.weight`、`mlp.fc2.weight`分别重排到局部FFN两端的1×1卷积。
- 官方卷积无偏置，因此替换层原Linear偏置和第二个LayerNorm不继续使用；对应模块被移除，不会留在优化器中。
- DWConv及BN按官方基础层默认方式初始化；SE Linear使用骨干同类型的截断正态初始化。所有新增模块在独立seed1234随机数上下文中创建，保留公共参数初始化与调用方随机流。
- 没有改成零初始化残差或额外门控；BN、激活、CLS路径改变意味着新模型初始输出不等于原SSL模型。此次属于模块迁移，不能声称与原模型初始函数等价。

## 数据和训练配置

默认`configs/localvit_first4_nightreid.yml`：NightReID，500个训练ID、15514张训练图、1394张查询图、26111张图库图。使用已有训练/query/gallery划分，SIE camera/view关闭。原来的MLDC+CNL为主要对照；独立`configs/baseline_nightreid.yml`提供禁用LocalViT的控制。

seed1234、120轮、batch64、20轮预热、原SSL初始化；基础LR=0.0004，沿用已对齐的分组学习率调度，头部5倍、MLDC权重1倍，偏置规则保持原设置。CNL权重0.1、gamma0.90、temperature0.05、momentum0.2。没有使用Night600检查点继续训练。

## 手动入口

从服务器项目根目录执行；以下命令只作为入口说明，本次没有执行训练：

```bash
cd /root/TransReID-SSL-main/transreid_pytorch
/root/miniconda3/envs/QREID/bin/python -u -m mldc_localvit_cnl.train \
  --config_file mldc_localvit_cnl/configs/localvit_first4_nightreid.yml
```

结果目录：`/root/log/transreid/nightreid/mldc_localvit_cnl/first4_hs_se_gamma090_seed1234_matched_schedule_120`。

训练过程记录配置、架构、初始化映射、逐轮损失、MLDC权重和评估结果；保存best及最终检查点。它们不包含可恢复训练的完整优化器状态，不能用于精确断点续训。

评估本实验检查点：

```bash
/root/miniconda3/envs/QREID/bin/python -m mldc_localvit_cnl.test \
  TEST.WEIGHT /path/to/transformer_best.pth
```

检查点包含前四层LocalViT结构信息；严格恢复不会把旧基线或其他架构的权重当作本实验权重。

代码验证：

```bash
/root/miniconda3/envs/QREID/bin/python -m mldc_localvit_cnl.verify \
  --report mldc_localvit_cnl/verification_cpu.json
```

验证使用CPU，包含官方模块AST/计算等价、CLS绕过FFN、矩形网格、原SSL权重映射、公共参数/RNG一致、禁用控制输出一致、四个LocalFFN前向、优化器覆盖、真实图片反向及检查点严格恢复。为适配当前2GB CPU验证环境，临时反向检查只为新LocalFFN与MLDC/BN/分类器分配参数梯度，但梯度通过后续所有原Block传播；此限制不用于实际训练。优化器更新0次。临时训练态前向会更新临时模型BN缓冲区，验证后不保留其权重或检查点。没有新的mAP/Rank-1结果，CUDA batch64显存容量未验证。
