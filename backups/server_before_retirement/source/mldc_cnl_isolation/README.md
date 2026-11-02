# CNL 独立收益的严格对照

这组对照从“对齐调度的可学习 MLDC＋CNL”出发，将读出改为第 12 个 block 的 CLS。CNL 作用位置、BN 后检索、分类头 5 倍学习率、分组余弦调度、基础 ID＋Triplet 损失、混合精度和所有数据设置保持一致。

实际不进行多层融合，也不训练融合权重。沿用的模型类仍构造一个冻结且不参与前向的融合权重对象，用来保留原来的初始化随机数顺序。

| 版本 | 实际读出 | 基础损失 | CNL 权重 |
| --- | --- | --- | ---: |
| last_cls_no_cnl | CLS12 → BN | ID＋Triplet | 0 |
| last_cls_cnl01 | CLS12 → BN | ID＋Triplet | 0.1 |

两个配置仅 `SOLVER.CNL_WEIGHT` 和 `OUTPUT_DIR` 不同。二者的差值才衡量当前训练设置下不依赖 MLDC 的 CNL 收益。

这里的末层 CLS 是最终 LayerNorm 之前的读出，与当前 MLDC 实现相同。它是“当前方案去掉融合”的消融，不等同于“原版 TransReID-SSL 所有配置”的复现。

## 手动训练

依次运行，不要同时运行两条训练命令：

```bash
cd /root/TransReID-SSL-main/transreid_pytorch
/root/miniconda3/envs/QREID/bin/python -u mldc_cnl_isolation/train.py --config_file mldc_cnl_isolation/configs/last_cls_no_cnl.yml
/root/miniconda3/envs/QREID/bin/python -u mldc_cnl_isolation/train.py --config_file mldc_cnl_isolation/configs/last_cls_cnl01.yml
```

均为 Night600、seed=1234、120 轮、batch=64，从原始 SSL 预训练重新训练，不从本次 9.3439% 的模型继续训练。结果分别保存到：

```text
/root/log/transreid/night600/mldc_cnl_isolation/last_cls_no_cnl_matched_schedule_120
/root/log/transreid/night600/mldc_cnl_isolation/last_cls_cnl01_matched_schedule_120
```

此代码准备不启动新训练，也不修改已安排的 NightReID、FeatEnHancer 顺序。

## 已完成结果应如何解释

- 原版配置的 TransReID-SSL＋CNL：第 120 轮 mAP=9.3439102553%、Rank-1=19.8165133595%。
- 用户报告的 TransReID-SSL 基线：mAP≈9.5%、Rank-1≈20%。下降约 0.1561、0.1835 个百分点；基线尚缺完整配置和同代码复现记录。
- 对齐调度的可学习 MLDC＋CNL：第 120 轮 mAP=12.2581008687%、Rank-1=26.0550469160%。

前两项说明现有单次结果没有观察到提升，不能用它们证明 CNL 稳定有害。后两项除了融合，还有读出中的最终 LayerNorm、BN 前后特征、head 学习率及调度、Triplet 数值计算路径的差别，不能把全部差距归于融合。

现有原版配置的训练记录显示，第 10–40 轮 CNL 正邻居覆盖率为 100%，第 120 轮下降到约 29.18%；说明早期相似度筛选基本没有排除同身份跨摄像头正样本。该现象与特征空间变化一致，但不足以独立证明指标下降原因。

如果要严格回答“CNL 对完全原版 TransReID-SSL 的影响”，还需与已完成版本完全配套的 CNL=0 运行；它属于另一组问题。现有入口可以直接运行：

```bash
cd /root/TransReID-SSL-main/transreid_pytorch
/root/miniconda3/envs/QREID/bin/python -u transreid_ssl_cnl/train.py SOLVER.CNL_WEIGHT 0.0 OUTPUT_DIR /root/log/transreid/night600/transreid_ssl_cnl/global_no_cnl_original_config_120
```

以上待训练配置均没有检索指标，不应以 CPU 梯度验证代替完整训练结果。
