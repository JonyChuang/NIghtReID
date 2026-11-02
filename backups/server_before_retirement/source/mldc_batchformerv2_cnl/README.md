# MLDC+CNL + BatchFormerV2

独立实验包 `mldc_batchformerv2_cnl`。原 `mldc_convstream_texture_cnl` BatchFormer 实验和 `mldc_mvi_cross_cnl` 实验保持原样。

训练时在第8个原始ViT块之后，对完整 `[B,129,768]` token（CLS和128个patch）逐位置做batch注意力。原路和增强路共享第9—12块、MLDC、BNNeck与身份分类器；测试只走原单图路径。

```bash
cd /root/TransReID-SSL-main/transreid_pytorch
/root/miniconda3/envs/QREID/bin/python -u -m mldc_batchformerv2_cnl.train_batchformerv2 --config_file mldc_batchformerv2_cnl/configs/batchformerv2_nightreid.yml
```

输出：`/root/log/transreid/nightreid/mldc_batchformerv2_cnl/from_ssl_seed1234_120`。

新模块从第1轮参与训练；原优化器20轮学习率预热保留。沿用服务器最新BatchFormer设置：`CE_original + 0.5*CE_augmented + Triplet_original + 0.1*CNL_original`，原路CE系数保持1，不新增蒸馏损失。SSL初始化、seed1234、120轮、PK64/K4及CNL配置保持一致。

实现、适配范围与评估命令见 [docs/BATCHFORMERV2.md](docs/BATCHFORMERV2.md)。本次只做源码和配置静态检查及上传后文件核对，没有执行模型或启动训练。
