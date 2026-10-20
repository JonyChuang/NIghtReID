# V4：作者 LPAM＋FFT，最后四层 CLS 固定等权融合

实验包：`/root/TransReID-SSL-main/transreid_pytorch/a3pfn_mldc_lpam_fft_v4_20261017`。
新包独立于历史 V4（MLDC 引导读取）和停止的 BoQ 实验。

全局特征严格为 `(CLS9 + CLS10 + CLS11 + CLS12) / 4`。没有可学习层权重，训练中不会偏离25%；这不同于之前仅以等权初始化、随后学习权重的 deep4 实验。主干12个block与LPAM从第1轮共同训练，不冻结活动主干、不使用局部分支独立预热。20轮warmup仅为原有学习率调度。

LPAM直接取自作者归档提交`5f1457f4cd18ee6194771392c3665aec3d22df3c`的`model/backbones/vit_pytorch7_0+FFT.py`，来源blob见provenance.json。三个LPAM类原样保留，输入CLS3/6/9/12、8头、共享卷积金字塔、FFT→einsum→IFFT公式、作者全通道缩放以及**无残差**的FFT ConvBlock均不改动。不能把无FFT主分支里的残差加进这个FFT版本。

原始单CLS路径N=1，softmax恒为1。Q/K的任务梯度应为零，FFT注意力输出等价于V投影乘头权重；Q/K仍可能受SGD权重衰减更新，但这不表示它们学到了注意力选择。当前实验可以检验这套CLS金字塔的联合训练效果，不能证明有效使用了空间频域信息。

保留作者四分支共享BN/分类器、四路ID取均值、Triplet仅作用于全局原始特征。不会混入自由Query、patch编码器、自适应融合等改造。

检索适配：主结果为四路BN后直接拼接3072维，再整体L2；同步报告同检查点全局768维、局部2304维和f3/f5/f6各768维，并统计融合修复/破坏的query数。拼接是Night600适配选择：作者after路径仅返回全局，另一路径返回元组，不能将本次拼接描述成原文默认检索的严格复现。保留global结果可检查局部作为辅助监督的作用。

训练：同一SSL初始化`pretrain_model/vit_base_ics_cfs_lup.pth`，不加载训练好的ReID；seed1234，256×128，batch64，SGD4e-4，bias2倍，BN/分类头5倍，120轮，每10轮完整评估，不重排序。每轮保存training.csv、layer_weights.csv；验证保存evaluation.csv、branch_evaluation.csv、evaluation_diagnostics.jsonl；保留best及120轮checkpoint。

从项目根目录启动V4：

```bash
/root/miniconda3/envs/QREID/bin/python mldc_lpam_fft/launch.py
```

launch会检查当前GPU和容器内存，成功后后台运行并写console文件；未满足条件立即返回，不进行CPU训练、不常驻等待GPU、不恢复旧实验。

控制组：同一启动器加`--baseline`，使用相同固定等权全局，不创建LPAM，损失为原全局ID＋Triplet，输出独立目录。控制组应在另一实验结束后单独启动。旧learnable deep4的10.7185/21.8349只作为历史参考，不能替代此固定等权控制组。

查看控制台：

```bash
tail -F /root/log/transreid/night600/vit_base_ics_lpam_fft_v4_20261017/deep4_fixed_equal_120.console.log
```

启动器和训练入口均拒绝覆盖非空输出目录。没有配置监控器或自动等待GPU。

手动复测已训练检查点：

```bash
/root/miniconda3/envs/QREID/bin/python mldc_lpam_fft/test.py --weight /root/log/transreid/night600/vit_base_ics_lpam_fft_v4_20261017/deep4_fixed_equal_120/transformer_best.pth
```

verify检查768维FFT读取的退化性质、真实ViT-Base前向、局部金字塔反向、固定四层精确均值及控制组输出。它不读取数据集或SSL权重，不执行优化器更新，不能当作Night600性能结果。
