# MLDC + BoQ joint training, revision 2 (2026-10-17)

The package replaces the single-query local reader with BoQ-style multi-query aggregation. This is an adaptation to multiple ViT depths, not an exact reproduction of BoQ's VPR architecture or results. Reference: https://github.com/amaralibey/Bag-of-Queries and https://arxiv.org/abs/2405.07364. The reference code is MIT licensed; its notice is included in BOQ_LICENSE.txt.

## Architecture

- Global descriptor: trainable MLDC fusion of post-block CLS from layers 9–12, initialized uniformly.
- Local inputs: full-grid post-block patch tokens from layers 4/6/8, without region masks, part labels, norm filtering or query competition.
- Each selected depth has independent normalization/projection to 192 channels and one BoQ block. The block encodes patches with a Transformer encoder, applies self-attention and LayerNorm to four learned queries, and reads patches through four-head cross-attention.
- Query parameters are shared across images and independently learned at each depth. They are not added to the cross-attention outputs.
- Three depths produce 12 query slots. A learned linear projection across these slots produces four rows of 192 channels, flattened and normalized to one 768-dimensional descriptor. Query/depth outputs are not averaged prematurely.
- Attention diagnostics retain B × heads × depths × queries × patches. Query overlap is measured but not penalized.
- Fusion: f = g + 0.1 * norm(g) * normalize(a), with gradients through every term.
- Loss: ID(f) + Triplet(f) + 0.1 * (ID(a) + Triplet(a)). The active backbone, MLDC, local reader and final BN/classifier all train from epoch 1. There is no frozen active backbone or local-only phase.

The SSL initialization, augmentation, batch size 64, seed 1234, SGD, 20-epoch learning-rate warmup and 120-epoch schedule match the independent deep-four baseline. Unused baseline output modules and the conventional constant BN offset retain the baseline implementation.

## Server commands

Run from `/root/TransReID-SSL-main/transreid_pytorch`:

```text
python mldc_boq_local_readout/train.py
python mldc_boq_local_readout/test.py --checkpoint /root/log/transreid/night600/vit_base_ics_mldc_boq_20261017/middle468_q4_120/transformer_120.pth
```

The default output directory is `/root/log/transreid/night600/vit_base_ics_mldc_boq_20261017/middle468_q4_120`. Training refuses to reuse a populated output directory. Old experiment weights are not loaded or overwritten. Epoch 120 is the primary endpoint; the test-mAP-selected best checkpoint is diagnostic only.

## Verification and diagnostics

Verification covers syntax/import/config, small-backbone joint gradients, BN updates, CPU FP32/bfloat16, CUDA float16 when available, checkpoint roundtrip, normalized attention and absence of a query residual. Real ViT/data integration is validated by the actual training start.

Full-gallery readouts include fused, local BN and local-removal using the same final fused BN. The last is a removal counterfactual, not an independently trained global baseline. Evaluation records Rank-1 corrections and new errors, per-head/averaged-head entropy, query-map overlap and residual size before/after BN.
