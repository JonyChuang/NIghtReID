# GLTrans Night600 adapter

This is an independent experiment copy placed alongside the project's other
experiment variants. Its files are `model/make_model_gltrans_night.py`,
`model/gltrans_modules.py`, `model/backbones/vit_pytorch_gltrans_night.py`,
`processor/processor_gltrans_night.py`, and `loss/make_loss_gltrans_night.py`.
It uses the original TransReID-SSL ViT implementation as the basis of its
separate backbone copy, plus the existing Night600 dataset loader, optimizer,
and evaluator. Existing experiment files are unchanged.

The module flow follows the official GLTrans source at
https://github.com/AWangYQ/GLTrans : class tokens 10/11/12 feed GAE; patch
tokens 10/11/12 feed PAM and PTF; the GAE feature guides PTF tokens through
sigmoid GMA; three horizontal PTL branches produce the local descriptor. The
final retrieval descriptor concatenates local (3D), global (D), and final CLS
(D) features. For ViT-B, that is 3840 dimensions.

The source assumes a 21x10 patch grid. This version uses the actual 16x8 grid
from the existing stride-16 / convolutional-stem config and partitions 16 rows
as 5/5/6. All other source-style module operations are retained. The official
code's GAE is one linear layer (the GeLU mentioned in the paper is commented
out in the official implementation). Training uses the paper's Eq. 16 rather
than repeating the final CLS triplet loss three times as the public code's
feature list does.

The loss calculation runs in FP32 even when the backbone forward uses CUDA
mixed precision. This avoids FP16 overflow and unstable gradients in the
project's Euclidean triplet distance. The batch-hard soft-margin triplet uses
`softplus(positive_distance - negative_distance)`, which is mathematically the
same objective as the project's `SoftMarginLoss` but does not overflow near a
distance gap of 88 in FP32. A non-finite loss stops immediately and names the
affected branch instead of logging `inf`/`nan` for more epochs.

From the project root on the server:

```bash
python train_gltrans_night.py --config_file experiment_variants/gltrans_global_local/config.yml
```

For a separately trained checkpoint, set `TEST.WEIGHT` to its path:

```bash
python test_gltrans_night.py --config_file experiment_variants/gltrans_global_local/config.yml TEST.WEIGHT ../../log/transreid/night600/vit_base_ics_gltrans_night/transformer_120.pth
```

The output directory is separate from every earlier experiment. No bright
image pairs, teacher model, Q-norm, or LayerScale are used by this version.
