# PFE v2：学习率与实际残差幅度

在当前 `a3pfn_mldc_featenhancer_20261017` 包内增加可选入口。原 A/B/PFE v1 源码、正在运行的进程和输出不变。本版本没有自动启动、排队或终止训练。

## 调整

- 主干基础学习率仍为 4e-4；新增 PFE 参数乘 10，即参考学习率 4e-3。头部仍乘 5；原 bias 的 2 倍规则保留。
- 各组使用同一个标量学习率曲线，10 倍关系在预热、余弦阶段、最低学习率均保持。预热起点主干 4e-6 / PFE 4e-5，bias 另乘 2。原调度器的统一预热起点和最低值无法严格保持这些倍率。
- up 投影初始化 std 从 .001 提至 .01。核心 MSRFConv/PFEBlock、192维瓶颈、L2/4/6/8 后插入位置不变。
- 原始方向 `D=up(PFE_increment(down(LN(X))))`；新残差 `Delta=.05*RMS(X)*D/max(RMS(D),1e-8)`，RMS 对每张图片的全部 token 和通道统计。正常非零输出满足 `||Delta||/||X||=5%`，零输出仍严格为零。按用户反馈由1%提高至5%；这是每处插入的实际范数比例，不是单纯的外乘系数。
- 统一缩放保留各 patch 修正幅度的相对空间差异，不逐 patch 强制等范数。归一化及残差加法用 FP32，整条路径可微，无主干冻结或 detach。
- 仍从原始 `pretrain_model/vit_base_ics_cfs_lup.pth` 初始化，联合训练120轮；CLS9–12固定各 .25，ID+Triplet、768维检索保持原协议。v2 是一次新实验，不接续正在运行的旧权重。

这控制了“实际作用量”，不证明随机初始方向有益，更不保证指标提升。需要看模块方向学习、CLS变化和最终检索指标。

## 启动

当前旧实验完成后，在服务器项目根目录执行：

```bash
/root/miniconda3/envs/QREID/bin/python -u mldc_feature_enhancement/train_pfe_v2.py
```

默认读取同目录 `pfe_v2.yml`，输出到：

```text
/root/log/transreid/night600/vit_base_ics_featenhancer_mldc_20261017/pfe_v2_rms5pct_lr10_fixed_equal_120
```

入口拒绝非空输出目录，避免覆盖已有结果。不要使用旧 `train_pfe.py` 启动 v2。测试入口为 `test_pfe_v2.py --weight <v2检查点>`。

## 直接观察改变

- 固定训练图片在任何更新前生成 `effect_probes/epoch_000/`，显示 PFE 开关前后的 CLS 注意力、实际增量和共享色标前后 patch 图。该组图片仅作尺寸变换及归一化，不随机擦除，便于观察人体；正常训练仍按原数据增强执行。
- 每轮首批在控制台和 `pfe_learning_diagnostics.jsonl` 记录真实学习率、四处残差比例、down/core/up 梯度、实际参数更新，以及去除统一缩放后的方向更新。
- 每10轮完整评估，`evaluation_diagnostics.jsonl` 记录全评估集四层实际比例，预期约 .05。
- 每10轮对同一组训练图片生成 `effect_probes/epoch_NNN/`，记录关闭 PFE 时最终描述符和 CLS 的改变；这是同模型功能诊断，不是独立训练无PFE基线。
- 配对热图共享色标，差值图使用自己的对称色标并标注绝对差值，避免把微小变化画成大变化。

预检在低优先级单线程 CPU 上读取原SSL和8张真实训练图片，完成全网前后向、一次丢弃的优化器更新、倍率全调度检查、CPU AMP及零输出、检查点严格重载。没有使用验证/测试集校准幅度，没有改动运行中模型或初始化CUDA。
