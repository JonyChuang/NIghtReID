# NightReID：Convpass并联旁路 + MLDC + CNL

独立包：`mldc_convpass_cnl`。保留原SSL骨干的卷积Stem、LayerNorm、Attention、GELU、FFN、偏置及CLS的FFN路径。在选定Block的Attention和FFN两处分别并联一个Convpass，原FFN不被替换。

当前只交付代码及零优化器更新验证，没有启动训练。默认继续此前的前四层范围；同时准备联合微调与全层冻结对照，均固定seed1234。

## 结构与官方来源

```text
每个选定Block：
                  ┌─ 原Attention ─────────────┐
x ── LN1 ────────┤                           ├─ + x -> y
                  └─ Convpass × 0.1 ──────────┘
                  ┌─ 原FFN（完整保留）─────────┐
y ── LN2 ────────┤                           ├─ + y -> 输出
                  └─ Convpass × 0.1 ──────────┘

Convpass：768 → Linear降维8 → QuickGELU
              → 3×3普通卷积（8通道）
              → QuickGELU → Dropout0.1 → Linear升维768

CLS9–12 -> 可学习MLDC -> 原BN检索头 -> 768维
损失 = 原ID + 原Triplet + 0.1×全局CNL
```

保留原Block对原分支和旁路分别使用DropPath的方式。CLS在旁路内部被当作独立1×1图像，使用同一卷积；patch恢复为16×8网格。这里的3×3是普通低维卷积，不是LocalViT的深度卷积。

官方代码：[JieShibo/PETL-ViT](https://github.com/JieShibo/PETL-ViT/tree/main/convpass)，固定版本见`provenance.json`。`QuickGELU`和`Convpass`类原样保存在`model/convpass_official.py`，MIT许可随包保存。`RectangularConvpass`仅将原来固定14×14的patch网格改为实际16×8。

## 初始化与冻结

先按原基线加载`pretrain_model/vit_base_ics_cfs_lup.pth`，再添加旁路；原参数名和数值完整保留。新增模块在独立seed1234随机数上下文中初始化，不改变公共参数初始化及调用方随机流。

沿用官方初始化：降维Linear使用Xavier，3×3卷积中心为通道恒等映射，其余位置为零，升维Linear权重与偏置为零。新旁路初始输出为零。需要核对初始eval特征与原基线一致，不能只检查形状或梯度。

初始反向时升维投影先获得梯度，降维和卷积梯度为零；升维权重离开零后，上游可以学习。不能把整个旁路都置零。验证中的非零投影小模块测试专门检查后续梯度通路。

默认冻结原骨干，只训练Convpass、MLDC权重及检索BN/分类头。当前骨干额外包含卷积Stem，因此冻结其BN运行统计；原DropPath和旁路Dropout仍按训练状态执行。检索头BN继续学习。此策略保留骨干权重和持久缓冲区，但旁路学习后仍会改变最终表示，不保证性能提升。

联合微调配置保留原骨干活动参数的训练。在该模式下，零初始化保证接入时eval函数一致，并不意味着训练后骨干权重不变。新增随机分支会增加训练时的随机抽样，不能声称两种网络训练轨迹逐步相同。

## 配置

| 配置 | 旁路层 | 原骨干 | 用途 |
|---|---|---|---|
| `convpass_first4_frozen_nightreid.yml`（默认） | 1–4，每层两个旁路 | 冻结，Stem BN统计冻结 | 优先检验保留预训练骨干的路线 |
| `convpass_first4_joint_nightreid.yml` | 1–4，每层两个旁路 | 联合微调 | 与当前联合训练基线比较接入方式 |
| `convpass_all_frozen_nightreid.yml` | 1–12，每层两个旁路 | 冻结，Stem BN统计冻结 | 更接近原文全层PETL插入方式 |
| `baseline_nightreid.yml` | 无旁路 | 联合微调 | 独立基线控制 |

上述配置均使用NightReID原划分，SIE关闭，768维检索，seed1234、batch64、120轮、20轮预热、基础LR0.0004、头部5倍、MLDC权重1倍。CNL权重0.1、gamma0.90。旁路权重使用基础学习率，偏置沿用原2倍规则。

本次迁移保留现有SGD及已对齐调度，没有照搬论文的AdamW、VTAB增强和超参数搜索。旁路宽度8、缩放0.1是预先指定值，尚未证明是NightReID最优值。冻结与联合微调属于不同训练策略，比较结果时应明确标注。

## 手动命令

```bash
cd /root/TransReID-SSL-main/transreid_pytorch
/root/miniconda3/envs/QREID/bin/python -u -m mldc_convpass_cnl.train \
  --config_file mldc_convpass_cnl/configs/convpass_first4_frozen_nightreid.yml
```

联合微调对照：

```bash
/root/miniconda3/envs/QREID/bin/python -u -m mldc_convpass_cnl.train \
  --config_file mldc_convpass_cnl/configs/convpass_first4_joint_nightreid.yml
```

训练日志和权重位于`/root/log/transreid/nightreid/mldc_convpass_cnl/`下的对应独立目录。测试使用与训练相同的配置，例如：

```bash
/root/miniconda3/envs/QREID/bin/python -m mldc_convpass_cnl.test \
  --config_file mldc_convpass_cnl/configs/convpass_first4_frozen_nightreid.yml \
  TEST.WEIGHT /path/to/transformer_best.pth
```

检查点严格记录层号、冻结策略、缩放、MLDC及完整模型参数。旧基线权重仅能通过明确的诊断导入方法加载，不能作为新旁路已经训练的结果。完整优化器断点续训未提供。

## 验证

```bash
/root/miniconda3/envs/QREID/bin/python -m mldc_convpass_cnl.verify \
  --report mldc_convpass_cnl/verification_cpu.json
```

核对官方非零旁路计算、矩形网格、零投影梯度启动、真实图片上原SSL输出及全部12层CLS、原参数和RNG、冻结权重与Stem BN统计、优化器覆盖、真实图片反向、严格检查点恢复，以及向已训练基线插入零旁路后的输出一致性。训练态检查会更新临时检索头BN缓冲区，但不更新模型参数或CNL记忆库；临时检查点随后删除。未测量新mAP/Rank-1，未验证CUDA batch64容量。
