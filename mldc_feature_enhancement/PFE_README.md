# PFE渐进增强：同一副本内的可选实验

新增独立入口 `train_pfe.py` 和配置 `pfe.yml`。原方案B的 `train.py`、`config.yml`、模型、主干与处理器保持不变；安装和验证不会启动PFE训练，也不排队接替B。

## 模型

原图进入相同SSL ViT。第2、4、6、8个Transformer块之后，各自用一个PFE适配器更新patch；CLS在适配器内保持原样，由后续自注意力读取增强后的patch。最后仍对第9–12层原始CLS做精确算术平均，再经原BNNeck，得到768维描述符。

每个适配器：LayerNorm → 768降至192维 → 论文MSRFConv及PFE增量变换 → 192升至768维 → 以0.1系数加回原patch。网格始终16×8。MSRFConv使用3×3深度分组卷积产生每通道9个感受野分量，展开为48×24网格，应用高宽池化注意力后以stride3聚合还原。PFE变换使用1×1、3×3深度卷积、1×1，内部比例0.5。升维层小幅非零初始化，使所有模块从第一步即可获得梯度。

这是依据[论文](https://link.springer.com/article/10.1007/s44443-026-00927-9)公式独立实现的核心模块迁移，不是完整PFE-Net检测器。未找到官方实现链接。固定ViT网格、投影维度、插入深度与残差系数属于本次适配选择；不迁移检测金字塔、C3k2-DCFF、框头与IoU损失。具体对应与公式维度解释见 `pfe_provenance.json`。

本实验不叠加A输入端增强或B局部读出。所有活动主干和新增模块从第1轮共同训练，不冻结、不detach活动路径、不划分人体区域；目标仍为完整ID＋Triplet，系数各1。使用相同SSL路径，不能从A/B训练好的ReID检查点初始化。

## 启动

待当前B实验结束后，在服务器项目根目录运行：

```bash
/root/miniconda3/envs/QREID/bin/python -u mldc_feature_enhancement/train_pfe.py
```

默认输出：

`/root/log/transreid/night600/vit_base_ics_featenhancer_mldc_20261017/pfe_patch2468_fixed_equal_120`

匹配控制组关闭PFE，其他设置相同：

```bash
/root/miniconda3/envs/QREID/bin/python -u mldc_feature_enhancement/train_pfe.py --config_file mldc_feature_enhancement/variants/progressive_enhancement_disabled/config.yml
```

控制组输出 `pfe_fixed_equal_baseline_120`。两组均为seed1234、SGD4e-4、batch64、120轮、20轮学习率warmup，每10轮验证；主干和分类头相同种子初始权重匹配。相同种子不保证两个结构的全部随机训练序列相同。

若需后续消融，可显式覆盖 `MODEL.PFE_LAYERS [8]` 并指定新的 `OUTPUT_DIR`，本次不自动启动消融。

## 日志、可视化与复测

逐轮记录真实batch数、AMP跳步和固定融合权重。每10轮对全部查询/图库计算检索指标，记录每个插入阶段的实际patch修正比例，以及后四层CLS在patch上的条件注意力熵。固定查询810、1544、11、2059、160、24保存PFE前后范数、实际修正幅度、后四层CLS注意力的原图叠加图。范数图不是身份贡献；PFE前后在同一阶段共享色标，修正图和CLS注意力使用各自色标。

```bash
/root/miniconda3/envs/QREID/bin/python -u mldc_feature_enhancement/test_pfe.py --weight CHECKPOINT_PATH
```

安装验证使用单线程、低优先级CPU随机输入，不加载SSL/数据集，不执行优化器更新，不占用GPU。验证成绩不作为实验性能。
