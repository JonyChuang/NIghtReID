# Current default: shared four-layer CNL

Run from the project root:

```bash
python mldc_information_preservation/train.py
```

Config: `configs/night600_shared_cnl.yml`. Output: `./output/mldc_information_preservation/deep4_uniform_shared_cnl09_120`.

The existing learnable MLDC and ID/Triplet are retained. The supervised preservation loss is replaced by PID-free CNL. Each layer mines same-layer neighbors at 0.9 times its own maximum, takes the image-index union, and all layers learn that set. One [N,4,768] memory is updated after successful optimizer steps.

See `SHARED_CNL_20261021.md` for definitions, checkpoints and verification. Old configs and completed output directories remain available. This code update does not start training.
