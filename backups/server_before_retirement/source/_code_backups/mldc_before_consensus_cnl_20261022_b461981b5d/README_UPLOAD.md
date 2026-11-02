# Current default: multilayer-mined final-retrieval CNL

Run from the project root:

```bash
python mldc_information_preservation/train.py
```

Config: `configs/night600_shared_cnl.yml`.
Output: `./output/mldc_information_preservation/deep4_uniform_shared_cnl_retrieval09_120`.

The existing learnable MLDC and original ID/Triplet remain. Each layer mines same-layer neighbors at 0.9 times its own eligible maximum. Their image-index union supplies pseudo positives; CNL contrasts only the final fused, BN-after, L2-normalized retrieval descriptors against global image candidates. It directly updates the fusion logits, BN and backbone. The mining references and all candidate keys are detached. CNL reads cameras and image indices, never ground-truth PID.

One logical image-index memory stores [N,4,768] mining keys and [N,768] final retrieval keys, synchronized after successful optimizer steps. It covers all training image indices and fills online; uninitialized slots never participate. Cross-camera candidates, self exclusion and duplicate coalescing remain.

See `SHARED_CNL_20261021.md` for loss definitions, checkpoints and verification. Old result directories remain. An already-running process keeps its loaded old code; this update applies at the next launch. This code update does not start or stop training.
