# 独立 ConvStream 纹理教师与最终检索特征蒸馏

本版按用户给定结构修改：以旧 DCM 的 **48/96/192 ConvStream 及已训练权重**为起点，独立建立训练期纹理教师；学生保留原始 TransReID-SSL 的 CNN Stem、Patch Embed、所有 Transformer Block 和预训练加载路径。教师通过跨图像判别关系约束最终 **BNneck 后、L2 归一化的 768 维检索特征**。

数据集沿用 **NightReID**，不是参考图上标注的 Night600。seed 固定 **1234**。原版本 `mldc_texture_distill_cnl` 保留，本版位于新目录 `mldc_convstream_texture_cnl`。

本轮仅完成代码、CPU 功能验证和数据流。**未进行教师专门化或学生训练，没有本轮 mAP / Rank-1 结果。**

## 1. 本版的实际接入位置

```text
同一张 RGB 训练图像
    ├─ 原始 TransReID-SSL → CLS9–12 → 可学习 MLDC → BNneck → L2 → 最终检索特征 u
    │       原 ID / Triplet / CNL 继续训练学生
    │
    └─ 独立 CNN 纹理教师（冻结）
            → 多尺度局部纹理描述子
            → 跨图像局部匹配相似度 s_T
            → 正负样本判别差 Delta_T
                    ↓
        与最终检索特征的判别差 Delta_F 做关系蒸馏
```

学生没有中间 Patch Hook、Patch 适配器、额外 CNN 融合或新增特征投影。两个网络的输入图像相同，Stem 和参数完全独立。蒸馏梯度通过最终特征的原有路径传回 BNneck、MLDC 与整个有效 ViT 主干。

## 2. 阶段 A：复用小型 DCM ConvStream，进行纹理专门化

ConvStream 类和参数键保持与旧 DCM 一致：

| 输出 | 尺寸，不含 batch |
|---|---|
| D1 | 48 × 128 × 64 |
| D2 | 96 × 64 × 32 |
| D3 | 192 × 32 × 16 |

权重来源：

```text
/root/log/transreid/nightreid/mldc_vitmatte_neck_cnl/
neck_all_post_mlp_dcm_guided_alpha01_gamma090_seed1234_matched_schedule_120/
transformer_best.pth
```

该检查点为旧实验第 40 轮；只严格加载 `detail_branch.convstream.*`。旧 DCM 的融合层、Neck 和 ViT 权重不进入新教师，学生另从原始 SSL 检查点初始化。

每个尺度提取高频、带通及平滑后梯度的局部能量，统一池化到 8 × 4，得到 48/96/192 通道的空间表示。拼接为 336 通道，再用轻量 `Conv1×1(336→128) + GroupNorm` 编码成 32 个 128 维局部描述子。身份头为 GAP → Linear256 → BNneck → ID 分类器。**完整教师仅 413,872 个参数**；ConvStream 复用旧权重，新空间投影和身份头从固定种子初始化。

教师默认固定专门化 **10 轮**：微调原 ConvStream 的卷积及 BN 仿射参数，同时训练轻量头；保留旧 ConvStream 的 BN 运行统计量。基准学习率 0.0004，新增头 5 倍，2 轮 warmup + cosine。

两个同几何视图共享身份标签，第二视图改变曝光系数 `[0.7,1.3]` 与 gamma `[0.8,1.2]`：

\[
L_T=\frac12\sum_{v=1}^{2}(L_{ID}^{v}+L_{Tri}^{v})
 +0.1L_{illum}+0.1L_{struct}.
\]

`L_illum` 匹配对应位置局部描述子的余弦方向；`L_struct` 匹配两个光照视图的空间 Patch 余弦关系矩阵，去掉对角项。纹理 bank 本身是固定可微变换，梯度能传回已有 ConvStream。

保存固定第 10 轮教师，然后全部冻结。最终教师只进行一次完整检索评估用于报告，不使用测试集选择专门化轮次。已有 DCM 40 轮训练及新增 10 轮专门化都是额外教师成本，不能写成与无教师基线总算力相同。

## 3. 阶段 B：局部纹理转成最终特征的关系监督

冻结教师对学生同一增强输入，以及固定曝光 0.7 的同几何输入，分别计算局部描述子。每个位置的置信度由**光照方向稳定性 × 纹理能量充足度**构成。

每张图保留 8 × 4 局部网格。图像间匹配限制在相同水平带，每带 2 行；使用互为最近邻的局部匹配，取 Top-4。匹配两端的置信度均需至少 0.5，至少保留 2 对匹配，得到图像间教师相似度 `s_T(i,j)` 及匹配置信度。这里没有人体分割掩码，置信度表示这些明确的匹配条件，不等同于已经定位了人体纹理。

沿用原 PK 采样。在当前训练 batch 内，选择跨相机同身份正样本，以及不同身份的跨相机困难负样本，排除重复图像索引。负样本按学生当前相似度选择，每个 anchor 最多 8 个。

\[
\Delta_T=s_T(a,p)-s_T(a,n),\qquad
\Delta_F=u_a^\top u_p-u_a^\top u_n.
\]

只有正负匹配置信度都至少 0.25，且教师 `Delta_T > 0.05` 的三元组参与蒸馏。以温度 0.2 将两个判别差转成正负偏好概率，计算加权二元 KL。它比较的是跨图像判别关系，不要求教师 128 维局部特征与学生 768 维全局特征逐通道相等。

\[
L=L_{ID}+L_{Tri}+0.1L_{CNL}
 +0.1\min(1,e/20)L_{KD}.
\]

日志包含有效跨相机正对数、候选三元组数、可靠三元组数、anchor 激活率、三元组覆盖率、加权 KD、KD / 原损失比例。每批日志还报告候选教师判别差的均值、最大值，区分匹配不足与教师判别不足。没有可靠目标时，该批 KD 为零，主任务继续优化。

## 4. 对照配置

| 配置 | 教师 | 学生监督 |
|---|---|---|
| `baseline_nightreid.yml` | 无 | 原 MLDC + CNL |
| `raw_relation_nightreid.yml` | 相同旧 ConvStream、轻量头与 10 轮预算，直接池化尺度归一化特征 | 最终检索特征的关系蒸馏 |
| `texture_relation_nightreid.yml` | 相同起点，使用多尺度纹理能量 | 最终检索特征的关系蒸馏，主方案 |

学生共享原协议：原始 SSL 初始化，seed1234，120 轮、20 轮相同 warmup、SGD 基准 LR 0.0004、batch64、SIE 关闭、CNL 权重 0.1 / gamma0.9、分类头 5 倍 LR，原有偏置倍率保留。正式代码默认训练整个有效主干。优先比较固定第 120 轮结果，同时保留相同评估频率下的 best。

Raw 与 Texture 的比较用于区分普通 CNN 关系监督和纹理变换的作用。三组是配置与待执行实验，不是已经取得的结果。

## 5. 训练与推理入口

以下命令本轮均未执行。工作目录为 `/root/TransReID-SSL-main/transreid_pytorch`，使用原 QREID 环境。该独立包复用主项目的数据集、优化器与评估工具。

```bash
# 分别专门化两个小型教师，各固定 10 轮
python -m mldc_convstream_texture_cnl.train_teacher --config_file mldc_convstream_texture_cnl/configs/raw_relation_nightreid.yml
python -m mldc_convstream_texture_cnl.train_teacher --config_file mldc_convstream_texture_cnl/configs/texture_relation_nightreid.yml

# 三组匹配学生实验
python -m mldc_convstream_texture_cnl.train --config_file mldc_convstream_texture_cnl/configs/baseline_nightreid.yml
python -m mldc_convstream_texture_cnl.train --config_file mldc_convstream_texture_cnl/configs/raw_relation_nightreid.yml
python -m mldc_convstream_texture_cnl.train --config_file mldc_convstream_texture_cnl/configs/texture_relation_nightreid.yml

# 完整本实验权重评估 / 导出原模型兼容的推理权重
python -m mldc_convstream_texture_cnl.test --config_file mldc_convstream_texture_cnl/configs/texture_relation_nightreid.yml TEST.WEIGHT /path/to/transformer_120.pth
python -m mldc_convstream_texture_cnl.export_inference /path/to/transformer_120.pth /path/to/inference.pth
```

输出位于 `/root/log/transreid/nightreid/mldc_convstream_texture_cnl/teachers/` 与 `students/`，不覆盖旧实验。入口要求空输出目录；当前不支持断点续训。学生训练会检查教师已完成专门化、seed / bank / 网格 / 维数及训练集指纹一致。

最终推理只执行原 CNN Stem + ViT → MLDC → BNneck → L2，导出时移除独立教师权重。没有新增学生推理参数。

## 6. 本轮验证及其实际边界

4 张真实 NightReID 训练图像、2 个跨相机身份，CPU，优化器更新 **0**。通过的功能检查包括：

- ConvStream 类与旧 DCM 一致，已训练流权重严格加载；原检查点未修改。
- 原 ViT Stem / Block / FFN 保持一致；全部 CLS 和最终输出与原 SSL 初始化完全相等，最大误差 0；已训练基线加载后输出也完全相等。
- 教师身份、光照与结构损失能传回旧 ConvStream；BN 统计量保持原值。
- 受控局部匹配和跨相机判别关系计算正确；同相机、重复图像及教师不可靠三元组被排除。
- 用受控教师判别矩阵验证，KD 梯度到达真实图像产生的最终 768 维检索特征、BNneck、MLDC、最后一个 ViT Block；教师无梯度且不在优化器内。
- ID + Triplet + CNL + KD 联合反向有限；完整学生保存/加载及移除教师后由原模型加载的输出完全一致。

**真实静态样本的结果也保留**：旧 ConvStream 加尚未训练的新纹理头时，这 4 张图有 4 个候选跨相机三元组，但教师最大判别差约 **0.02485**，小于设定的 0.05，可靠三元组为 **0**。不能把受控目标的梯度通过当成真实纹理教师已经有效，也不能由这 4 张图推断专门化后的效果。本轮没有为得到非零 KD 而降低筛选阈值。

CPU 的学生反向检查仅临时打开最后一个 Block 和原学习头以控制内存；正式训练默认全部有效主干可训练。当前没有测试 CUDA 混合精度、正式 batch 显存或长程指标。

代码数据流见 `docs/module_dataflow.md`；验证报告见 `docs/verification.json`。
