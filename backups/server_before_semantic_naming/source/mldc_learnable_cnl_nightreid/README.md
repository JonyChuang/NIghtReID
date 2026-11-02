# 对齐调度的可学习 MLDC＋CNL：NightReID

本版本复制自原服务器当前 `mldc_learnable_global_cnl`，仅将配置中的数据集切换为 `nightreid`，结果写入独立目录；入口和结果元信息相应改名。原版 Night600 实验与当前 TransReID＋CNL 代码均不修改。

- ViT-Base，原始 SSL 权重 `pretrain_model/vit_base_ics_cfs_lup.pth` 重新初始化。
- CLS 9–12 的可学习 softmax 融合，初始权重均为 0.25，温度 0.9。
- BN 后全局特征用于检索及归一化后的 CNL。
- ID＋Triplet＋0.1×CNL；CNL gamma=0.9、temperature=0.05、momentum=0.2。
- 沿用已对齐的分组余弦调度、head LR factor=5、fusion LR factor=1。
- 120 轮、batch=64、seed=1234、基础学习率 0.0004、warmup=20。
- 每 10 轮评估，保留最佳 mAP 模型和第 120 轮模型。

## 数据协议

使用服务器已有 `data/NightReID`：

| 部分 | 目录 | 图像数 |
| --- | --- | ---: |
| Train | bounding_box_train | 15514 |
| Query | query | 1394 |
| Gallery | bounding_box_test | 26111 |

训练集有 500 个身份、6 个摄像头，500 个身份都有跨摄像头样本。没有切换为 query-1000 或无 distractor 的 gallery 版本。数据驱动的历史提示字符串仍可能显示 “Night600 loaded”；实际数据来源和数量由上表及 `run_metadata.json` 记录。

## 运行与结果

接续任务使用 `run_after_current.py`，运行条件及被顺延的旧等待任务记录在 `queue_plan.json`。它等待指定 TransReID＋CNL 进程完成，并确认第 120 轮训练记录、最终评估与检查点齐全，再等待 GPU 0 空闲。接着用完整 batch 64 做一次 GPU 显存验证（无优化器更新），验证通过后从 SSL 权重开始训练 NightReID。

NightReID 完成后恢复先前等待的 FeatEnHancer 任务。若前序或本实验失败，状态写入 `queue_status.json`，同时恢复旧等待任务。

结果目录：

```text
/root/log/transreid/nightreid/mldc_learnable_global_cnl/deep_four_uniform_cross_camera_cnl_fixed_equal_schedule
```

输出 `training.csv`、`evaluation.csv`、`best_metrics.json`、`transformer_best.pth`、`transformer_120.pth`、层权重与热图。`queue_status.json` 完成时也会记录最终与最佳指标。

仅在没有接续任务、需要手动运行时使用：

```bash
cd /root/TransReID-SSL-main/transreid_pytorch
/root/miniconda3/envs/QREID/bin/python -u mldc_learnable_cnl_nightreid/train.py --config_file mldc_learnable_cnl_nightreid/configs/nightreid.yml
```

已有接续任务时不要重复手动启动。复测第 120 轮：

```bash
cd /root/TransReID-SSL-main/transreid_pytorch
/root/miniconda3/envs/QREID/bin/python -u mldc_learnable_cnl_nightreid/test.py --config_file mldc_learnable_cnl_nightreid/configs/nightreid.yml TEST.WEIGHT /root/log/transreid/nightreid/mldc_learnable_global_cnl/deep_four_uniform_cross_camera_cnl_fixed_equal_schedule/transformer_120.pth
```

CPU 验证已通过：配置仅 dataset/output 两项变化，初始化与特征输出一致，真实 NightReID 数据加载、跨摄像头正邻居及融合参数梯度正常，完整 121 个调度阶段与等权调度一致。CPU 验证没有执行优化器更新，也不是检索指标测量。
