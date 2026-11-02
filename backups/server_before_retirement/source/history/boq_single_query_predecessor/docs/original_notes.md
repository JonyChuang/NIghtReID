# Joint MLDC and unpartitioned local readout

This is an untrained experiment revision, not a measured improvement. All computation stays on the server. The existing trained experiments and weights are not modified.

The active ViT backbone, deep-four CLS fusion logits, final BN/classifier and local reader/BN/classifier all train together from epoch 1. Initialization uses the same SSL checkpoint as the deep-four baseline, not the trained ReID checkpoint. Ordinary learning-rate warmup remains; there is no local-only training phase. Unused baseline output modules and the conventional constant BN offset retain the baseline implementation.

The global descriptor uses CLS layers 9–12. The local descriptor reads post-block image tokens from layers 4/6/8. Each depth has its own normalization and key/value projection, and a single query per head. Attention is normalized within each depth; depth summaries receive fixed equal weights. There are no region masks, part assignments, competing part queries, norm-based patch filtering or extra final-CLS addition.

Cosine query/key logits use temperature 0.2 to remove query magnitude from the softmax scale. This is a proposed remedy for the latest reader's nearly uniform averaged attention, not a claim that cosine attention alone improves ReID. A mean-pooling mode is supplied to test whether spatial selection helps.

Fusion is f = g + 0.1 * norm(g) * normalize(a), with gradients through every term. The coefficient is fixed from epoch 1; the first revision removes the nearly constant 1% learned gate as a variable. It does not guarantee useful angular changes after BN.

The objective is ReID(f) + 0.1 * ReID(a), with ID cross-entropy and the baseline soft-margin batch-hard triplet. The final classifier is trained on exactly the fused feature used for retrieval. There is no separate global classification head or three-head objective.

## Server commands

Run from `/root/TransReID-SSL-main/transreid_pytorch` on a machine with a visible CUDA GPU and enough memory:

```text
python mldc_boq_local_readout/train.py
python mldc_boq_local_readout/train.py MODEL.LOCAL_READER mean OUTPUT_DIR /root/log/transreid/night600/vit_base_ics_mldc_joint_local_20261016/middle468_mean_120
python mldc_boq_local_readout/train.py MODEL.LOCAL_LAYERS '[9,10,11]' OUTPUT_DIR /root/log/transreid/night600/vit_base_ics_mldc_joint_local_20261016/deep91011_cosine_120
python mldc_boq_local_readout/test.py --checkpoint /path/to/this/experiment/transformer_120.pth
```

These commands are provided for later execution; installing or verifying this package does not launch training. The trainer refuses to reuse a populated output directory. The 120th epoch is the primary endpoint; a test-mAP-selected best checkpoint is diagnostic only.

## Diagnostics and comparison

Evaluation retains the original full-gallery protocol, normalized descriptors and no reranking. It records the fused descriptor, local BN descriptor and local-removal counterfactual using the final fused BN. That last readout is explicitly not an independently trained MLDC baseline. It counts Rank-1 corrections and new errors, and records per-head and averaged-head attention entropy plus residual size before and after BN. Training logs total-objective gradients for the backbone, fusion logits, local reader and heads.

The first three comparisons are the existing independent deep-four baseline, middle-depth mean pooling and middle-depth cosine reading. The deep 9/10/11 option tests the depth hypothesis with the same number of source layers. No test-set coefficient search is intended. Layer 4/6/8 and strength 0.1 remain hypotheses until trained comparisons support them.

Small substitute-backbone tests cover gradient flow, BN updates, readout shape, attention normalization, checkpoint roundtrip and CPU bfloat16 autocast. They do not validate full ViT training, CUDA autocast, data loading or retrieval metrics.
