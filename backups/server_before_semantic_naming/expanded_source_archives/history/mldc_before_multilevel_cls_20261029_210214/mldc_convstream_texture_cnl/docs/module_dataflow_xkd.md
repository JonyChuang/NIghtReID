# 当前模块数据流：纹理教师 → 最终特征 InfoNCE

```mermaid
flowchart TB
  subgraph T[阶段一：教师 120 轮，seed 1234]
    X1[RGB 训练图像与身份标签] --> C1[旧 DCM ConvStream 48/96/192]
    C1 --> E1[高频 + 带通 + 梯度能量]
    E1 --> S1[8×4 池化 / 拼接 / 336→128 编码]
    S1 --> H1[GAP → Linear256 → BNneck → ID 头]
    H1 --> LT[ID + Triplet]
    S1 --> LC[双光照视图的局部光照一致性与空间关系一致性]
    LT --> OT[优化独立教师]
    LC --> OT
    OT --> CK[固定末轮检查点，随后冻结]
  end
  subgraph S[阶段二：学生 120 轮，seed 1234，原学习率调度]
    X2[同一增强图像与训练标签] --> V[原 Stem + PatchEmbed + ViT1–12]
    X2 --> TC[冻结 CNN 纹理教师]
    V --> CLS[CLS9–12 → 可学习 MLDC]
    CLS --> BN[原 BNneck → L2 → 最终检索特征 u：768维]
    CLS --> TRI[原 Triplet]
    BN --> ID[原 ID + CNL]
    TC --> TV[身份读出 → L2 → t：256维，停止梯度]
    BN --> PS[训练期 Ps：768→128 → L2]
    TV --> PT[训练期 Pt：256→128 → L2]
    PS --> KD[跨模型 InfoNCE，同图为正对、不同身份为负对]
    PT --> KD
    KD --> LS[ID + Triplet + 0.1CNL + 渐增权重 InfoNCE]
    TRI --> LS
    ID --> LS
    LS --> U[更新原学生与两侧蒸馏映射头]
  end
  CK --> TC
  subgraph I[阶段三：检索推理]
    XT[测试图像] --> VM[原 TransReID-SSL + MLDC + BNneck + L2]
    VM --> F[原 768 维身份特征]
    F --> R[跨摄像头 ReID 检索]
  end
  U -. 移除教师与蒸馏映射头 .-> VM
```

教师内部空间一致性只发生在独立 CNN；学生没有中间 Patch 蒸馏。教师网络冻结，两侧独立蒸馏映射头可学习。其他同身份图片不作 InfoNCE 负样本，原 CNL 提供跨相机约束。KD 权重为 `0.1 min(1,epoch/20)`，温度为 0.1。

10 轮先导教师很弱，120 轮教师和学生结果尚待训练。数据流表示实际执行路径，不代表已取得指标提升。
