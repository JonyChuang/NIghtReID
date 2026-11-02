# 环境与运行入口

仓库保存实验源码、配置、方法说明和历史实现。数据集、SSL预训练以及已训练检查点保留在实验服务器，Git仓库不包含这些大文件。

## 在现有服务器继续研究

工程根目录：`/root/TransReID-SSL-main/transreid_pytorch`。现有Python环境：`/root/miniconda3/envs/QREID/bin/python`。

从根目录选择各包 `variants/<实验名称>/README.md`，按其中的训练入口、配置、特征读出和输出位置操作。`experiment_index.csv` 列出全部配置，`PATH_MAPPING.md` 对照整理前后的名称。

同一配置的新一轮训练使用独立的、有明确含义的输出目录，例如 `seed_1234_repeat_2`。已有实验结果通过语义路径访问，原检查点与指标保留。

## 在其他机器使用源码

先准备与CUDA兼容的PyTorch环境，再按 `requirements.txt` 安装项目依赖。该文件沿用原工程的依赖列表；不同研究模块还可能需要NumPy、SciPy、scikit-learn、Matplotlib等其入口实际导入的库。

数据集应按 `datasets/night600.py`、`datasets/nightreid.py` 和对应YAML的 `DATASETS` 设置准备。初始化权重由YAML的 `MODEL.PRETRAIN_PATH` 指定。包内初始化权重（如 `mldc_patch_grm/initialization/`）也需要单独从实验服务器获取。

YAML保留服务器上的路径以便继续实验。其他机器需要调整数据根目录、初始化路径和输出路径。支持尾部配置覆盖的入口可以这样运行：

```bash
python mldc_learnable_global_cnl/train.py \
  --config_file mldc_learnable_global_cnl/variants/deep_four_uniform_cross_camera_cnl/config.yml \
  MODEL.PRETRAIN_PATH /path/to/vit_base_ics_cfs_lup.pth \
  DATASETS.ROOT_DIR /path/to/data \
  OUTPUT_DIR /path/to/results/learnable_deep_four_cnl_seed_1234
```

## 历史实现

`history/` 中部分内容是修改前的文件快照，恢复时需要共享工程和相应配置。各README说明模块来源和使用状态。历史配置的存在不表示其旧入口和全部依赖仍完整；具体限制已写入相应说明。

文献标题、作者仓库与模块来源见 [REFERENCES.md](REFERENCES.md)。每个实验的README分别说明所采用模块与本项目的组合方式。
