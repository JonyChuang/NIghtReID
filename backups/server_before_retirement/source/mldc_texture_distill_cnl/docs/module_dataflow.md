# 本轮模块数据流：mldc_texture_distill_cnl

对应代码主配置：`configs/texture_structure_nightreid.yml`。所有张量均省略 batch 维，随机种子固定为 1234。当前为代码验证阶段，教师与学生均未训练。

## 1. 离线纹理教师

```mermaid
flowchart LR
    X["训练图像<br/>3 × 256 × 128"] --> S["原始 SSL CNN Stem<br/>冻结参数与 BN 统计量<br/>64 × 128 × 64"]
    S --> B["纹理能量 bank<br/>池化后 64 × 16 × 8"]
    B --> E["两层 Conv3×3 + GN + ReLU<br/>可学习编码器<br/>128 × 16 × 8"]
    E --> H["GAP → Linear256<br/>BNneck → ID 分类器"]
    H --> L["ID + Triplet<br/>只更新编码器与学习头"]
    L --> C["固定训练 40 轮<br/>保存最终教师，供学生阶段冻结使用"]
```

教师 Stem 结构是 `Conv7×7,s2 → IBN → ReLU → Conv3×3 → IBN → ReLU → Conv3×3 → BN → ReLU`。输入尺寸 256 × 128，输出 64 × 128 × 64；Stem 从原始 TransReID-SSL 预训练权重初始化。

纹理 bank 先计算每通道的局部能量：

\[
G_1=\mathcal G_{5\times5,\sigma=1}(F),\quad
G_2=\mathcal G_{9\times9,\sigma=2}(F),
\]
\[
E=(F-G_1)^2+(G_1-G_2)^2+
  (S_x*G_1)^2+(S_y*G_1)^2,
\]
\[
B_{texture}=\sqrt{\max(\operatorname{AvgPool}_{16\times8}(E),10^{-12})}.
\]

`Sx / Sy` 为固定 Sobel 核，Gaussian / Sobel 使用逐通道卷积和 reflect padding。能量先池化再开方，保留每个通道，不在 bank 中压成一张单通道热力图。

`raw` 对照仅把 bank 替换为 `AvgPool16×8(F)`，其余教师网络、初始化、数据和预算相同。两种教师分别训练，学生运行时只使用其中一个。

## 2. 学生训练：主任务与纹理蒸馏

```mermaid
flowchart TB
    X["同一张增强后的图像 x<br/>3 × 256 × 128"]
    X --> PE["原 CNN Patch Embed<br/>CLS + 128 Patch + 位置编码<br/>129 × 768"]
    PE --> B4["原 ViT Block 1–4<br/>129 × 768"]
    B4 --> B12["原 ViT Block 5–12"]
    B12 --> CLS["第 9 / 10 / 11 / 12 层 CLS<br/>每层 768 维"]
    CLS --> M["可学习 MLDC<br/>融合特征 f：768 维"]
    M --> BN["原 BNneck<br/>z：768 维"]
    BN --> ID["原分类头 → L_ID"]
    M --> TRI["L_Triplet"]
    BN --> N["L2 归一化 → 原 CNL 记忆库<br/>L_CNL，gamma = 0.9"]
    ID --> BASE["L_base = L_ID + L_Triplet + 0.1 L_CNL"]
    TRI --> BASE
    N --> BASE

    B4 --> P["取第 4 层 Patch，排除 CLS<br/>128 × 768"]
    P --> A["训练期适配器<br/>LN + Linear768→128<br/>重排得到 S：128 × 16 × 8"]
    X --> T["阶段 A 训练完成的教师<br/>Stem → texture bank → 编码器<br/>eval + no_grad，全冻结"]
    T --> TF["教师空间特征 T<br/>128 × 16 × 8"]
    A --> KD["L_KD = L_rel + 0.1 L_att<br/>Patch 相似关系 + 空间能量分布"]
    TF --> KD
    KD --> TOTAL["L = L_base + 0.1 r(e) L_KD<br/>r(e) = min(1, e / 20)"]
    BASE --> TOTAL

    classDef original fill:#edf3fb,stroke:#587899,color:#26323e;
    classDef teacher fill:#f0ecf6,stroke:#87749b,color:#26323e;
    classDef auxiliary fill:#eaf4ec,stroke:#729578,color:#26323e;
    classDef objective fill:#fbeee1,stroke:#ba8d5e,color:#26323e;
    class PE,B4,B12,CLS,M,BN original;
    class T,TF teacher;
    class P,A auxiliary;
    class ID,TRI,N,BASE,KD,TOTAL objective;
```

教师直接接收学生同一增强输入，空间位置一一对应。图中教师的 GAP、BNneck 和分类器用于阶段 A 学习身份；阶段 B 的蒸馏目标使用编码器输出的空间特征，教师全部冻结。

将学生和教师空间特征展开为 `S,T ∈ R^(128×128)`：第一维为 128 个空间 Patch，第二维为 128 个通道。对每个 Patch 沿通道做 L2 归一化，得到 `s_i,t_i`。

| 损失 | 实际计算 | 主方案权重 |
|---|---|---|
| `L_rel` | 学生、教师各自构建 128 × 128 的 Patch 余弦关系矩阵，去对角项后取 MSE | 1.0 |
| `L_att` | 每位置的通道均方能量归一化成 128 点概率分布，计算 KL(教师‖学生) | 0.1 |
| `L_feature` | 每 Patch 归一化通道向量直接取 MSE | 0；仅 feature 对照启用 |

`L_att` 的名字表示空间能量分布匹配，**不是**取 Transformer Attention 矩阵做监督。

完整损失为：

\[
L=L_{ID}+L_{Triplet}+0.1L_{CNL}
  +0.1\min(1,e/20)\left(L_{rel}+0.1L_{att}\right).
\]

梯度范围：主任务训练原有全部有效主干、MLDC、BNneck 和分类器；蒸馏项训练适配器及其之前的 Block1–4 和 Patch Embed。教师无梯度。适配器初始为随机映射，随学生共同学习；它只参与辅助损失计算。

## 3. 最终推理

```mermaid
flowchart LR
    X["行人图像"] --> V["原 CNN Patch Embed<br/>原 ViT Block 1–12"]
    V --> C["CLS9、CLS10、CLS11、CLS12"]
    C --> M["可学习 MLDC"]
    M --> B["原 BNneck<br/>768 维描述子"]
    B --> N["L2 归一化"]
    N --> R["Query / Gallery 距离与排名"]
```

`eval` 模式下教师与适配器不执行；`export_inference.py` 移除它们的权重，保留原 ViT + MLDC 的参数结构。纹理的作用来自学生训练时收到的梯度，推理时使用经过该监督训练的主干。

## 4. 对应代码位置

| 数据流节点 | 实现 |
|---|---|
| 冻结 SSL Stem、纹理 bank、身份教师 | `model/texture_teacher.py` 中 `TextureBank / TextureTeacher` |
| 第 4 层取 Patch、LN + Linear 适配 | `model/make_model_mldc.py` 中 `MLDCReID` |
| 关系、能量、直接特征损失 | `model/texture_teacher.py` 中 `distillation_terms` |
| 损失权重与 20 轮 ramp | `model/make_model_mldc.py` 中 `distillation_loss` |
| ID + Triplet + CNL + KD | `processor/processor_mldc.py` 中 `do_train` |
| 教师训练与固定第 40 轮权重 | `train_teacher.py` |
| 推理辅助权重移除 | `export_inference.py` |

附图：[PNG](module_dataflow.png)、[矢量 PDF](module_dataflow.pdf)、[可编辑 draw.io](module_dataflow.drawio)。PNG/PDF 与 draw.io 使用同一组图元坐标生成；本机没有 draw.io 命令行，PNG/PDF 为直接矢量绘制导出。
