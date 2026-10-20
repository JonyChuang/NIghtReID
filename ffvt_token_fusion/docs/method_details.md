# Night600 FFVT global / cross-layer fusion experiment

This updates the existing `alf_end2end` experiment in place. File and entry-point
names remain compatible with the existing project layout; the model is now
FFVT-style fusion rather than ALF learned-query attention.

## Architecture

- Shared TransReID-SSL blocks 1-11 process all 128 patches and CLS.
- Blocks 3, 6, 9 and 11 supply post-block patch features. For each selected
  block, MAWS uses its scaled QK logits before attention dropout: softmax over
  the CLS row and separately over the CLS column, average each across heads,
  then multiply. CLS is explicitly excluded from the top-k candidates.
- Select 12 patches per layer, concatenate the 48 tokens, and prepend raw
  block-11 CLS. Hard selection is discrete; gathered features retain gradients.
- The global branch runs the original block 12 and final LayerNorm.
- The fusion branch runs an independent copy of block 12 and final LayerNorm
  over the 49-token sequence and takes its CLS output. Both copies are made
  AFTER loading the configured SSL checkpoint. Parameters are not shared.
- No ALF query, expanded attention width, ALF projection, external residual
  scaling, teacher, or Q-norm is used. Transformer blocks retain their normal
  internal residual connections. Both branches train the shared backbone.

The main configuration uses 768D descriptors and 12-head fusion. The fusion
block inherits the backbone dropout / stochastic-depth settings. Fusion block
and copied LayerNorm use backbone learning rates. Newly initialized BN and
identity heads use `SOLVER.FFVT_HEAD_LR_FACTOR: 5.0`; the existing bias multiplier
still applies. `MODEL.FFVT_LAYERS` must contain layers preceding the final block;
`MODEL.FFVT_TOKENS_PER_LAYER` must not exceed the patch count.

## Supervision and retrieval

Two independent BNNecks and linear identity classifiers:

```
ID = (ID(CLS) + ID(fusion)) / 2
Triplet = (Triplet(CLS) + Triplet(fusion)) / 2
Loss = MODEL.ID_LOSS_WEIGHT * ID + MODEL.TRIPLET_LOSS_WEIGHT * Triplet
```

ID uses BN -> Dropout -> classifier; Triplet uses pre-BN descriptors in FP32.
There is no third concatenation classifier. The one-entry `parts` container is
retained only for the existing loss/component API.

Default retrieval is `concat(BN_cls(CLS), BN_fusion(fusion))`, 1536 dimensions,
followed by the evaluator's global L2 normalization. There is no retrieval
Dropout or classifier. `TEST.NECK_FEAT before` selects pre-BN descriptors.
`TEST.FEATURE_BRANCH` supports `concat`, `cls`, and `fusion` on the SAME FFVT
checkpoint, permitting branch contribution evaluation without retraining.

## Run in the existing QREID environment

From the shared project root:

```bash
python ffvt_token_fusion/train.py
```

The output directory remains `../../log/transreid/night600/vit_base_ics_alf_end2end`.
Start a fresh run from the configured SSL backbone. Old ALF checkpoints cannot
resume this architecture; they are rejected explicitly. Updates do not change
models already loaded by a running process. Existing output filenames may be
reused by a new training run; retain any older results you need before starting.

To test each branch of a trained compatible checkpoint:

```bash
python ffvt_token_fusion/test.py TEST.WEIGHT path/to/ffvt_checkpoint.pth TEST.FEATURE_BRANCH concat
python ffvt_token_fusion/test.py TEST.WEIGHT path/to/ffvt_checkpoint.pth TEST.FEATURE_BRANCH cls
python ffvt_token_fusion/test.py TEST.WEIGHT path/to/ffvt_checkpoint.pth TEST.FEATURE_BRANCH fusion
```

## Logging and visualization

Logs include CLS `Acc`, `Fusion Acc`, each branch's ID / Triplet loss, actual
`Base Lr`, `Fusion Lr`, `Head Lr` (representative weight learning rates), and
`Fusion/CLS norm` (pre-BN, latest training batch; not retrieval branch weight).

With the default evaluation interval, every 10 epochs three random same-ID,
cross-camera pairs produce original query / reference / global CLS / fusion CLS
panels in `ffvt_heatmaps/epoch_NNN/`. Attribution uses the selected before/after-BN
space, absolute gradient-times-activation at input patch embeddings, and smooth
original-image overlays. Hard MAWS selection is held fixed for these gradients.
Colors are independently scaled and do not measure retrieval gains.

Each PNG has a companion NPZ storing both raw attribution maps, patch grid,
query/reference indices, selected layers, actual MAWS scores and zero-based
patch indices for the QUERY. The images are not direct MAWS attention maps.

## Provenance and adaptation

- FFVT paper: https://www.bmva-archive.org.uk/bmvc/2021/assets/papers/0685.pdf
- Author implementation: https://github.com/Markin-Wang/FFVT/blob/main/models/modeling.py

The paper uses token fusion before its last block. This experiment adapts that
mechanism to a separate ReID branch, uses four selected layers, excludes CLS
from candidate tokens, initializes the fusion block from SSL weights, retains
an ordinary global branch, and applies independent identity / triplet losses.
These changes are a proposed Night600 adaptation, not paper-reproduction results.
