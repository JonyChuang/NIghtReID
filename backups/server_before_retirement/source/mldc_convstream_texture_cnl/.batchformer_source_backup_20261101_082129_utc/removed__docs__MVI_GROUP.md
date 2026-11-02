# Shared-tail multi-view training

Entry: `python -u -m mldc_convstream_texture_cnl.train_mvi_group --config_file mldc_convstream_texture_cnl/configs/mvi_group_nightreid.yml`.

This is a ViT adaptation inspired by multi-view integration, not an exact MVI2P
reproduction. One existing backbone, MLDC, BN affine and ID classifier are reused.
There are no additional trainable parameters, teacher network, CAM classifier,
patch classifier, prototype memory or NFC objective in this mode.

## Forward paths

The ordinary image batch runs through the unchanged 12-block backbone and
CLS9-12 MLDC. When the auxiliary objective is active, post-block-8 full tokens
are also returned from that same forward; the stem and first eight blocks are
not recomputed. Distinct same-ID images are paired within the current batch,
requiring different cameras by default. Repeated augmentations of the same image
are never counted as two support images. Images without an eligible partner
still receive every original single-image loss.

A group starts with the arithmetic mean of its two block-8 CLS tokens, followed
by both sets of patch tokens. For 128 patches per image this is 257 tokens. The
existing blocks9-12 process that sequence with ordinary self-attention and FFN.
Their four group CLS outputs use the original MLDC weights. No positional
embedding is added a second time; each patch already carries its source-image
positional context. Groups include their member images, not leave-one-out
targets. Training labels define groups; this is not an unlabeled NFC graph.

## Normalization and losses

The original single-image BN call is unchanged and is the only path updating
BN running_mean, running_var and num_batches_tracked. Group features use a
functional BatchNorm operation with the current single-image batch mean and
biased variance, detached from autograd, and the same BN affine parameters.
The group path never runs a second stem or other BatchNorm module.

The total loss is:

`original_ID + original_Triplet + 0.1*CNL + ramp*(alpha*group_ID + lambda*IP)`.

Original ID and Triplet coefficients remain 1, with no averaging or halving of
the original identity objective. Group ID uses the original classifier. IP is
`mean(1 - single_unit dot stopgrad(group_unit))` over eligible images. Only IP
detaches the group descriptor; group ID backpropagates through the shared tail,
MLDC, BN affine, classifier, and each support image's earlier backbone blocks.
BN moment detachment prevents group ID from updating the single-image path
indirectly through normalization statistics. All shared parameters remain
trainable; stop-gradient does not freeze a second model.

Defaults: alpha=1, lambda=0.1, original SSL initialization, 120 epochs, seed1234,
PK64/K4, original learning-rate schedule and CNL gamma0.9. Both auxiliary terms
are off in epochs1-20 and ramp to full strength in epochs21-25. The YAML disables
old NFC mean shift, center contrast, class KL, PAD and PEM. Original CNL memory
receives only original single-image descriptors after successful optimizer steps.

Only ordinary independent single-image inference is run during evaluation.
Group fusion and label access are disabled in evaluation mode. Results use
`mvi_group/from_ssl_seed1234_120`, separate from previous experiments. This
entry expects an empty output directory and does not implement resume.

## Logs

`mvi_group_id`, `mvi_ip`, `mvi_weighted`, `mvi_ramp`, active image coverage,
cross-camera group fraction, group accuracy, single/group cosine and group/mean
cosine are recorded in training.csv. Cosine diagnostics describe representations,
not retrieval gains. All reported validation metrics use single-image inference.

## Verification scope

This revision was checked by static Python parsing, YAML parsing and source
review only. No model forward/backward, runtime tests, GPU checks, training or
experiment was executed. Runtime behavior, memory demand and retrieval benefit
remain to be measured when the user starts training.
