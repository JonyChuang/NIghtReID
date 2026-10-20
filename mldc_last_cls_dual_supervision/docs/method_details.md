# 后四层等权MLDC＋末层双监督＋分别归一化拼接（恢复版）

共享ViT、原SSL初始化、后四层固定等权。恢复NightReID迁移前的两个监督分支和检索拼接方式。

- `g = mean(raw_CLS9:12)`；`c = base.norm(raw_CLS12)`，末端LayerNorm参与训练。
- g和c各有独立768维BNNeck和分类器；ID监督BN后特征，Triplet监督各自BN前特征。
- `L = 0.7*(ID_g+Tri_g) + 0.3*(ID_c+Tri_c)`。`MODEL.LAST_LOSS_WEIGHT=0.3`，融合比例自动为1-w。
- 移除独立1536维拼接BN、分类器和拼接损失。拼接仅用于检索。
- 默认`TEST.NECK_FEAT=after`，恢复原双监督设计的BN后读出。两支分别L2归一化后拼接，并除以sqrt(2)，得到1536维单位向量。
- 拼接余弦精确等于两支余弦的各50%；不受两支原始范数差异控制。0.3是监督系数，检索仍各占50%。
- 主干和两支从第1轮共同训练，保留原SSL、SGD、120轮、20轮warmup与原cosine factory；两套头使用相同标称5倍LR，记录实际LR。

每10轮在同一检查点、同一图库上评估fusion、last、concat三种读出，记录各自mAP/Rank-1、拼接纠正/破坏/净变化、两支正确并集流失。三种评估不等于三套训练损失。

新变体为`mldc_fixed4_lastln_dual_supervision_unit_concat_v2_20261019`。
默认结果目录：`/root/log/transreid/night600/vit_base_ics_mldc_dual_20261019/dual_lastw03_unit_concat_after_v2_120`。
历史NightReID式实验仍在原结果目录，恢复版不会覆盖其日志、权重或接管已有训练进程。

## 手动训练入口

仅在用户决定开始新实验后执行：

```text
cd /root/TransReID-SSL-main/transreid_pytorch
/root/miniconda3/envs/QREID/bin/python -u mldc_last_cls_dual_supervision/train.py
```

不会自动开始或重启训练。新实验从`pretrain_model/vit_base_ics_cfs_lup.pth`开始；不将三监督旧检查点当成双监督继续训练。

后续0.5对照需使用独立目录：

```text
/root/miniconda3/envs/QREID/bin/python -u mldc_last_cls_dual_supervision/train.py MODEL.LAST_LOSS_WEIGHT 0.5 OUTPUT_DIR /root/log/transreid/night600/vit_base_ics_mldc_dual_20261019/dual_lastw05_unit_concat_after_v2_120
```

## 同检查点三种读出

```text
/root/miniconda3/envs/QREID/bin/python -u mldc_last_cls_dual_supervision/test.py --checkpoint /absolute/path/transformer_120.pth
```

新版本默认使用保存配置的after；追加`--neck-feat before`可比较BN前的两支分别归一化拼接。测试入口按变体选择模型并严格加载。

历史NightReID三头检查点由独立的`nightreid_legacy`模型/评估器读取，保留原始cat(g,c)及其保存的before/after配置，不会默默改写旧实验的读出定义。完整历史源码快照在服务器包内`source_snapshots/nightreid_three_heads_20261019`，只保留源文件及部署记录，不重复复制训练权重。

## 日志和验证

`training.csv`只记录两支ID/Triplet、0.7/0.3贡献、分类准确率、实际更新及AMP跳步；`evaluation.csv`仍每轮三行；`joint_gradients.jsonl`记录活动主干和两套头。保留三种逐查询CSV、描述符缓存、同序元数据、实际拼接能量比例和各读出最佳检查点。缓存及模型只留服务器。

验证覆盖不同原始范数下的50/50拼接、整体单位范数、平均余弦恒等式、同身份同摄像头过滤、双分支损失和精确系数梯度；服务器验证进一步使用真实SSL、真实图片和完整ViT前后向、严格检查点重载。历史70轮三头检查点另检查严格加载及原读出重现。验证不执行优化器更新。
