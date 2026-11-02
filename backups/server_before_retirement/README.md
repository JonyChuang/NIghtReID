# Server source backup before experiment retirement

This is the server source snapshot taken before removing experiment families 05–16 at the owner's request.

- `source/`: current server source, configurations, README files, licenses, and existing source history.
- `expanded_source_archives/`: source-only contents extracted from two historical source archives.
- `snapshot_manifest.json`: original server paths, content digests, and file modes.

The snapshot excludes datasets, pretrained/trained weights, image features, caches, and result image directories. It is a recovery archive, not a claim that every historical entry can run unchanged.

Selected families: Patch+GRM, BoQ, TAR, MUFASA/Slot Attention, LocAt, Patch/CNN enhancement, input FeatEnHancer, LocalViT, Convpass, ViTMatte, texture teachers/distillation, and ConvNeXt relation distillation.

Other families (MLDC/CNL, PAD, NFC, PEM, BatchFormer, BatchFormerV2, MVI, etc.) remain outside the retirement scope. Shared runtime code must be retained where still required.

## Retirement completed

The selected server experiments have been removed after remote confirmation of the backup. See [retirement_result.json](retirement_result.json) for the completion record and remaining packages. Source recovery is supported by this backup; deleted trained weights are not included.
