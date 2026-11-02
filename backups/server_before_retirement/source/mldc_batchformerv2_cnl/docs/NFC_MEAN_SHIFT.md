# Reciprocal-neighbor mean shift: replacement for the extra center contrast term

The existing `train_nfc_center` entry and default `configs/nfc_center_nightreid.yml` now select `NFC_CENTER.LOSS: mean_shift`. No new source experiment copy is needed. Old center contrast is retained only for historical checkpoint compatibility and explicit comparisons; it is not added on top of mean shift.

## Exact objective

For each live normalized MLDC/BN anchor q_i, build its own reciprocal top2 neighbor set R(i) from current-batch detached features. Use:

`L_i = sum_{j in R(i)} 0.5 * sum_dim((q_i - stopgrad(k_j))^2) / |R(i)|`.

Average over active anchors. Sum feature dimensions; do not divide by768. This is equivalent in anchor gradient to half-squared distance to the raw arithmetic mean `mu_i = mean(stopgrad(k_j))`, plus a detached neighbor-variance term. The arithmetic mean is NOT renormalized. Ambient feature gradient is `q_i - mu_i`; the network additionally applies the feature-normalization Jacobian and shared-parameter update.

There is no positive-owner lookup, negative softmax denominator, temperature or class-probability KL in the new term. Original ID/Triplet/CNL still provide identity separation.

## Neighborhood and metadata

- Current-batch only: no extra memory, cache, frozen teacher or test features.
- Duplicate stable image indices are coalesced to one detached mean-direction graph vertex. This excludes self and all duplicated augmentations from the anchor's neighbor candidates.
- Top2 squared-Euclidean reciprocal selection is identity/camera blind. Identities/cameras are used only for duplicate metadata validation and logging; they do not filter neighbor attraction. Graph endpoints therefore need not belong to the same identity.
- Each original anchor row, including different augmentations, learns from its own graph neighbors. At least one independent mutual neighbor is required. One neighbor is pair alignment, not a multi-neighbor center; one- and multi-neighbor active fractions are logged separately. Fewer than3 unique images cannot run exact top2 and returns zero, as does an empty mutual set.
- Only the live anchor role receives mean-shift gradients. Neighbor keys and their arithmetic mean are detached. All images can receive gradients when acting as anchors.

## Schedule and losses

`ID + Triplet + 0.1*CNL + lambda(epoch)*L_MS`.

Unchanged baseline settings: SSL initialization, uniform MLDC, trainable original classifier,120 epochs, LR4e-4,20 LR-warmup epochs, PK64/K4, seed1234, CNL gamma0.9/all_nonself. Additional-loss weight is0 for epochs1-20, then0.02/0.04/0.06/0.08/0.10 over epochs21-25 and0.10 afterwards. This reuses0.1 only as an initial setting; its gradient strength is not equivalent to the old center contrast weight.

Default output is `.../mldc_batchformerv2_cnl/nfc_mean_shift/from_ssl_seed1234_120` to avoid mixing different CSV fields or overwriting center-experiment results. The entry still refuses nonempty result directories. No results were deleted or migrated by this source change.

## Monitoring

Step logs are labeled `MeanShift`. Epoch columns use `nfc_ms_*`: raw/weighted loss, weight, active fraction, average neighbors, single/multi-neighbor fraction, raw mean norm, ambient drift norm, unit-sphere tangent drift norm, neighbor cosine and variance, and graph/anchor-weighted neighbor identity and camera diagnostics. Drift diagnostics are not measured network parameter gradient norms. There is no center-positive/hard-negative margin in this new objective.

Single-image inference is unchanged. A distinct mean-shift architecture variant is saved; the old configuration without a LOSS field defaults to contrastive when loading historical checkpoints.

## Scope of this change

Code and configuration edits only, plus local static syntax parsing. No loss forward/backward, model inference, training or experiment was run. No existing training process was stopped or restarted. A running process continues using its previously loaded contrastive loss until the user launches a new process.
