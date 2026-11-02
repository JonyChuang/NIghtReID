# Training-time batch NFC center contrast

Implementation only. This change does not start, stop or restart training, and does not change a running class-KL process. Use `train_nfc_center` with `configs/nfc_center_nightreid.yml` for this separate training mode in the same source package.

## Objective

`L = ID + Triplet + 0.1*CNL + lambda(epoch)*NFC_center`.

The student is initialized from the original TransReID-SSL weights, with uniform MLDC and a new trainable500-class classifier. Original ViT, MLDC, BN and classifier all train. No extra head, frozen teacher, offline targets, classifier KL or NFC memory is used. The original CNL retains its own memory and update rules; the new NFC loss never reads it.

Defaults:120 epochs, original SGD base LR4e-4,20 LR-warmup epochs, PK64/K4, seed1234, original CNL gamma0.9/all_nonself. The existing sampler and data augmentation are preserved. Cross-camera support availability is measured, not assumed.

## Per-batch computation

1. Normalize final BN features. Detach a copy and coalesce duplicate stable image indices into one mean-direction graph vertex. Duplicate metadata must agree.
2. Build original mutual top2 squared-Euclidean neighbors over current-batch vertices. Identity and camera are NOT used for neighbor selection or filtering. Self edges are excluded. Add the owner's own vertex to each neighbor support set.
3. Choose same-ID, other-camera positive owners for each anchor, using normal supervised training labels. Remove the anchor image vertex from each candidate center support set. This excludes all duplicate augmentations of that image. Retain centers only when at least two independent images remain.
4. Normalize the detached sum to form a local center. Center membership is not identity-filtered: impurity is a diagnostic, not a hidden change to NFC.
5. Compare the live single-image anchor to each positive center, against all different-ID detached batch keys. Each valid anchor-center pair has its own positive in the denominator plus all batch negatives. Average over valid pairs, not over feature dimensions. Support vectors and negative keys receive no gradient from their key role. They receive gradients when acting as anchors in their own rows.

Loss per pair is `softplus(logsumexp(negative cosine/tau) - positive_center_cosine/tau)`, tau0.05. No valid center, fewer than3 distinct images or missing negatives gives differentiable zero. There is no fallback that presents a single image as a multi-image center.

## Activation

Epochs1-20:lambda0; no graph computation or additional loss. Epoch21:.02,22:.04,23:.06,24:.08,25onward:.1. This is independent of the ordinary learning-rate warmup schedule.

## Diagnostics and output

Step logs report raw/weighted NFC loss, currentlambda, active-anchor fraction, positive-center count and mean support size. Epoch CSV also records graph coverage, same-ID edge purity and cross-camera edge fraction for diagnosis, available cross-camera positive-owner pairs, and average positive-center/hard-negative margin. Warmup diagnostics are zero because the graph is not computed.

The default output directory is `.../mldc_convstream_texture_cnl/nfc_center/from_ssl_seed1234_120`; the entry refuses a nonempty output directory. This preserves previous KL results. Checkpoints retain original model and CNL state, omit unused optimizer resume state, and are compatible with the package's single-image inference path. No added inference parameters exist.

This is an unvalidated implementation. Neither performance gains nor cross-camera graph quality have been established for this mode. No training or experiment was run as part of this code-only change.
