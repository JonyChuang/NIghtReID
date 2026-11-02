# 本轮已停训：2026-10-27

已按用户要求停止，不再继续训练教师或学生。最后完整训练为第82轮，第83轮仅有部分日志；已保存的最佳学生检查点来自第70轮。原训练与控制器记录的PID均已不存在，流水线状态为 `stopped_by_user`。

已用第70轮检查点完成CPU静态分析，参数更新0次。结论：蒸馏实际参与优化，当前同图对齐充分，但没有形成净身份检索增益；已定位明显的卷积边界响应和跨相机目标不足。

完整分析：[纹理分支作用分析](C:/Users/Larmes/Documents/ChatGPT/NIghtReID/texture_branch_effect_analysis_20261027/analysis.md)。

第80轮当前 mAP=50.36%、Rank-1=65.14%；原基线同轮为51.55%、66.86%。修正41个query、损坏65个query。

---

以下保留启动时的模型结构与训练配置，作为历史记录；它不表示训练仍在运行。

# 当前训练：独立纹理教师 → 最终特征 InfoNCE

2026-10-26 已启动 GPU 后台流水线：**教师 120 轮 → GPU 检查 → 学生 120 轮**，两阶段 seed 均为 **1234**。启动信息见 `training_launch_full.json`，实时状态以服务器日志为准。

10 轮先导教师 mAP=0.3108518%，Rank-1=0.0717360%。训练集内跨摄像头诊断同样弱，未发现明显的评估路径错误。尚不能确定是训练预算、表示能力还是泛化问题。另开 120 轮教师，保留先导结果；延长训练是否有效须由实际结果判断。

## 教师

复用旧 DCM 第 40 轮的 `detail_branch.convstream.*`，独立 48/96/192 ConvStream。各尺度生成高频、带通及平滑梯度能量，在 8×4 网格池化；拼接为 336 通道，经 Conv1×1(336→128)+GroupNorm 编码。身份读出：GAP→Linear256→BNneck→500 类分类头。完整教师 413,872 个参数。

卷积和轻量头可训练，原 ConvStream BN 运行统计量冻结。两个同几何光照视图的 ID+Triplet，加 0.1 倍局部光照一致性、0.1 倍空间关系一致性。固定 120 轮，5 轮 warmup 后 cosine，LR=0.0004，新增头 5 倍 LR。重新从相同旧 DCM 起点和 seed1234 初始化轻量头，保留 10 轮先导结果。每轮保存最新快照，固定末轮保存教师并进行一次完整检索评估，不以测试集选择教师轮次。

教师输出：`/root/log/transreid/nightreid/mldc_convstream_texture_cnl/teachers/texture_full120_seed1234`。

## 学生与实际蒸馏

```text
原 SSL → 原 Stem / PatchEmbed / ViT1–12 → CLS9–12 → MLDC → BNneck → L2 → u:768
冻结 CNN 教师 → 纹理编码 / 身份读出 → L2 → t:256
训练期 Ps(u):128 → L2；Pt(stopgrad(t)):128 → L2
                       ↓
                 跨模型 InfoNCE
```

教师 CNN、纹理编码和原身份头冻结；两个独立蒸馏映射头可学习，均为无偏置 Linear，LR 为学生基准 LR 的 5 倍。温度 0.1，学生为 anchor、教师为 key。正对为同一原图索引，负对仅包含不同训练身份；同图重复采样保留多正对，其他同身份图片不进入分母。当前不添加跨相机同身份正对，原 CNL 继续提供跨相机约束。

总损失：`ID + Triplet + 0.1 CNL + 0.1 min(1,epoch/20) InfoNCE`。

参考 [FBT（ICCV 2025）的最终特征对比监督](https://yuan-gao.net/pdf/ICCV25_fuse.pdf)，维度映射参考 [CRD（ICLR 2020）](https://arxiv.org/abs/1910.10699)。未迁移 FBT 完整融合网络；这里 batch 内 softmax InfoNCE 也不是 CRD 原始 NCE/记忆库实现。

约束原 BNneck 后最终 768 维检索特征，梯度沿原路径传回 MLDC 和全部有效 ViT。辅助模块在独立 RNG 作用域初始化。学生仍为 120 轮、20 轮原 warmup、原 fixed-equal 学习率调度、SGD LR=0.0004、原头 5 倍 LR、batch64、SIE 关闭、CNL weight=0.1 / gamma=0.9。每 10 轮评估并保存 best，第 120 轮另存固定末轮。

学生输出：`/root/log/transreid/nightreid/mldc_convstream_texture_cnl/students/texture_xkd_teacher120_seed1234_matched120`。

推理只保留原学生 Stem、ViT、MLDC、BNneck 和 L2，导出时移除教师和两个蒸馏映射头，仍为原 768 维表示。学生没有中间 Patch 蒸馏或额外 CNN 融合。旧局部匹配、判别差 KL 不在本次训练中使用；历史说明见包内 `README_legacy_relation.md`。

## 已完成的 GPU 检查

RTX 4090 D，真实 batch64，完整主干 AMP 前反向，优化器更新 0。用已完成的 10 轮教师验证代码；120 轮教师完成后，会使用实际教师再次检查，成功后才启动学生。

- 原 SSL 初始化的最终输出、原始 MLDC 特征、全部 CLS 最大误差均为 0，原学生 RNG 一致。
- 64 个 anchor 均激活，64 个同图正对、3,840 个不同身份负对，同身份负对为 0。
- KD 梯度到达最终特征、BNneck、MLDC、ViT 首末层及两侧映射头。
- 教师无梯度，状态和 BN 缓冲不变，不在学生优化器内。
- ID+Triplet+CNL+KD 联合反向有限，峰值分配显存约 5.21GiB。
- 推理不调用教师；移除训练模块后，状态键与原模型一致。

这些检查验证实现和显存，不能证明指标提升。已有 DCM、先导 10 轮和当前教师训练都是额外成本，只有学生预算与原 MLDC+CNL 匹配。

流水线目录：`/root/log/transreid/nightreid/mldc_convstream_texture_cnl/pipeline_texture_xkd_teacher120_seed1234_20261026`。其中 `status.json`、`teacher.console.log`、`student_preflight.console.log`、`student.console.log` 记录各阶段。任一步失败会停止后续阶段。断开本地会话不影响后台进程；服务器重启会终止训练，当前学生入口不支持直接断点续训。请勿重复启动。

当前配置：`configs/texture_teacher120_nightreid.yml`、`configs/texture_xkd_nightreid.yml`；入口依次为 `train_teacher`、`verify_xkd_gpu`、`train`。新数据流见 `dataflow/module_dataflow_xkd.md`（包内 `docs/module_dataflow_xkd.md`）；旧 `module_dataflow.*` 图形属于历史关系蒸馏方案。
