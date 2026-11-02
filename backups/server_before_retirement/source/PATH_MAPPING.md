# 实验目录与配置映射

旧检查点中的架构variant标识保持原值；本表记录文件路径迁移。已有结果保留原位置，通过语义结果路径访问。

## 实验包

| 原目录 | 新目录 |
|---|---|
| `a3pfn_mldc` | `mldc_cls_readout` |
| `a3pfn_mldc_deep4_random` | `mldc_deep4_readout` |
| `a3pfn_mldc_learnable_cnl_20261020` | `mldc_learnable_global_cnl` |
| `a3pfn_mldc_dual_supervision_20261019` | `mldc_last_cls_dual_supervision` |
| `a3pfn_mldc_featenhancer_20261017` | `mldc_feature_enhancement` |
| `a3pfn_mldc_joint_local_20261016` | `mldc_boq_local_readout` |
| `a3pfn_mldc_locat` | `mldc_locat_attention` |
| `a3pfn_mldc_lpam` | `mldc_tar_refinement` |
| `a3pfn_mldc_lpam_fft_v4_20261017` | `mldc_lpam_fft` |
| `a3pfn_mldc_mufasa` | `mldc_mufasa_slots` |
| `a3pfn_mldc_patch_compensation_20261016` | `mldc_patch_grm` |
| `alf_end2end` | `ffvt_token_fusion` |

## 配置

| 原配置 | 新配置 |
|---|---|
| `a3pfn_mldc/configs/Night600/vit_base_ics_last_cls.yml` | `mldc_cls_readout/variants/last_raw_cls_control/config.yml` |
| `a3pfn_mldc/configs/Night600/vit_base_ics_mldc.yml` | `mldc_cls_readout/variants/all_cls_random_initialization/config.yml` |
| `a3pfn_mldc/configs/Night600/vit_base_ics_mldc_deep4.yml` | `mldc_cls_readout/variants/deep_four_uniform_learnable/config.yml` |
| `a3pfn_mldc/configs/Night600/vit_base_ics_mldc_deep_prior.yml` | `mldc_cls_readout/variants/deep_cls_prior_initialization/config.yml` |
| `a3pfn_mldc/configs/Night600/vit_base_ics_mldc_random_residual.yml` | `mldc_cls_readout/variants/all_cls_raw_residual_readout/config.yml` |
| `a3pfn_mldc/configs/Night600/vit_base_ics_mldc_supervised_residual.yml` | `mldc_cls_readout/variants/all_cls_residual_supervision/config.yml` |
| `a3pfn_mldc/configs/Night600/vit_base_ics_mldc_uniform_init.yml` | `mldc_cls_readout/variants/all_cls_uniform_initialization/config.yml` |
| `a3pfn_mldc/configs/Night600/vit_base_ics_mldc_uniform_lr5.yml` | `mldc_cls_readout/variants/all_cls_uniform_fusion_lr_five/config.yml` |
| `a3pfn_mldc_deep4_random/configs/Night600/vit_base_ics_mldc_deep4_random.yml` | `mldc_deep4_readout/variants/deep_four_random_learnable/config.yml` |
| `a3pfn_mldc_deep4_random/configs/Night600/vit_base_ics_mldc_deep4_uniform.yml` | `mldc_deep4_readout/variants/deep_four_uniform_learnable/config.yml` |
| `a3pfn_mldc_dual_supervision_20261019/config.yml` | `mldc_last_cls_dual_supervision/variants/fixed_global_last_cls_unit_concat/config.yml` |
| `a3pfn_mldc_featenhancer_20261017/baseline.yml` | `mldc_feature_enhancement/variants/patch_residual_disabled/config.yml` |
| `a3pfn_mldc_featenhancer_20261017/config.yml` | `mldc_feature_enhancement/variants/patch_fen_safa_residual/config.yml` |
| `a3pfn_mldc_featenhancer_20261017/cppi.yml` | `mldc_feature_enhancement/variants/global_cls_cross_camera_cnl/config.yml` |
| `a3pfn_mldc_featenhancer_20261017/cppi_baseline.yml` | `mldc_feature_enhancement/variants/global_cls_without_cnl/config.yml` |
| `a3pfn_mldc_featenhancer_20261017/cppi_block8.yml` | `mldc_feature_enhancement/variants/legacy_cnn_injection_block_eight/config.yml` |
| `a3pfn_mldc_featenhancer_20261017/cppi_no_cnl.yml` | `mldc_feature_enhancement/variants/global_cls_without_cnl_alias/config.yml` |
| `a3pfn_mldc_featenhancer_20261017/cppi_vit_source.yml` | `mldc_feature_enhancement/variants/legacy_vit_pyramid_injection/config.yml` |
| `a3pfn_mldc_featenhancer_20261017/pfe.yml` | `mldc_feature_enhancement/variants/progressive_patch_enhancement/config.yml` |
| `a3pfn_mldc_featenhancer_20261017/pfe_baseline.yml` | `mldc_feature_enhancement/variants/progressive_enhancement_disabled/config.yml` |
| `a3pfn_mldc_featenhancer_20261017/pfe_v2.yml` | `mldc_feature_enhancement/variants/progressive_enhancement_norm_five_percent/config.yml` |
| `a3pfn_mldc_joint_local_20261016/config.yml` | `mldc_boq_local_readout/variants/middle_patch_multi_query_boq/config.yml` |
| `a3pfn_mldc_learnable_cnl_20261020/configs/Night600/vit_base_ics_mldc_deep4_random.yml` | `mldc_learnable_global_cnl/variants/legacy_random_start_config/config.yml` |
| `a3pfn_mldc_learnable_cnl_20261020/configs/Night600/vit_base_ics_mldc_deep4_uniform.yml` | `mldc_learnable_global_cnl/variants/deep_four_uniform_cnl_alias/config.yml` |
| `a3pfn_mldc_learnable_cnl_20261020/configs/Night600/vit_base_ics_mldc_deep4_uniform_cnl.yml` | `mldc_learnable_global_cnl/variants/deep_four_uniform_cross_camera_cnl/config.yml` |
| `a3pfn_mldc_learnable_cnl_20261020/configs/Night600/vit_base_ics_mldc_deep4_uniform_no_cnl.yml` | `mldc_learnable_global_cnl/variants/deep_four_uniform_without_cnl/config.yml` |
| `a3pfn_mldc_locat/configs/Night600/gaug_norm.yml` | `mldc_locat_attention/variants/gaussian_attention_with_layernorm/config.yml` |
| `a3pfn_mldc_locat/configs/Night600/gaug_raw.yml` | `mldc_locat_attention/variants/gaussian_attention_raw_cls/config.yml` |
| `a3pfn_mldc_locat/configs/Night600/legacy.yml` | `mldc_locat_attention/variants/original_mldc_control/config.yml` |
| `a3pfn_mldc_locat/configs/Night600/locat.yml` | `mldc_locat_attention/variants/gaussian_attention_prr/config.yml` |
| `a3pfn_mldc_locat/configs/Night600/norm_control.yml` | `mldc_locat_attention/variants/readout_layernorm_control/config.yml` |
| `a3pfn_mldc_locat/configs/Night600/prr_only.yml` | `mldc_locat_attention/variants/patch_refinement_only/config.yml` |
| `a3pfn_mldc_lpam/configs/Night600/vit_base_ics_mldc_lpam.yml` | `mldc_tar_refinement/variants/tar_cls_residual_three_heads/config.yml` |
| `a3pfn_mldc_lpam_fft_v4_20261017/config.yml` | `mldc_lpam_fft/variants/fixed_global_lpam_fft_concat/config.yml` |
| `a3pfn_mldc_mufasa/configs/night600.yml` | `mldc_mufasa_slots/variants/slot_residual_joint_bnneck/config.yml` |
| `a3pfn_mldc_patch_compensation_20261016/configs/Night600/vit_base_ics_mldc_patch_compensation.yml` | `mldc_patch_grm/variants/frozen_global_patch_grm/config.yml` |
| `alf_end2end/configs/Night600/vit_base_ics_alf_end2end.yml` | `ffvt_token_fusion/variants/maws_global_and_fused_tokens/config.yml` |
| `configs/Night600/vit_base_ics.yml` | `experiment_variants/patch_cls_fusion/config.yml` |
| `configs/Night600/vit_base_ics_gltrans_night.yml` | `experiment_variants/gltrans_global_local/config.yml` |
| `configs/Night600/vit_base_ics_init.yml` | `experiment_variants/dce_qnorm_student/config.yml` |
| `configs/Night600/vit_base_ics_qcat_rank.yml` | `experiment_variants/qnorm_context_rank/config.yml` |
| `configs/Night600/vit_base_ics_qcat_rank_hinge..yml` | `experiment_variants/qnorm_context_rank_hinge/config.yml` |
| `configs/Night600/vit_base_ics_qcat_raw.yml` | `experiment_variants/qnorm_context_raw/config.yml` |
| `configs/Night600/vit_base_ics_qnorm_sfm.yml` | `experiment_variants/qnorm_spatial_feature_mask/config.yml` |
| `configs/NightReID/vit_base_ics.yml` | `experiment_variants/nightreid_dataset_legacy/config.yml` |

## 历史源码

| 原目录 | 新目录 |
|---|---|
| `.mufasa_bnneck_review_20261013` | `history/mufasa_joint_bnneck_review` |
| `.mufasa_joint_review_20261013` | `history/mufasa_joint_supervision_review` |
| `codex_backup_20261006_fusion_heatmaps` | `history/patch_cls_before_heatmap_changes` |
| `_code_backups/mldc_before_weight_controls_20261014_110920` | `history/mldc_before_weight_lr_controls` |
| `_code_backups/mldc_before_uniform_init_20261014_100720` | `history/mldc_before_uniform_initialization` |
| `_code_backups/mldc_before_residual_retrieval_20261014_124035` | `history/mldc_before_residual_retrieval` |
| `_code_backups/mldc_before_supervised_residual_20261014_125944` | `history/mldc_before_supervised_residual` |
| `_code_backups/mldc_before_deep4_20261014_145452` | `history/mldc_before_deep_four` |
| `_code_backups/mldc_deep4_uniform_start_20261014_190552` | `history/mldc_deep_four_before_uniform_start` |
| `_code_backups/mldc_joint_before_boq_v2_20261017_052816` | `history/boq_single_query_predecessor` |
| `_code_backups/mufasa_before_bnneck_20261013` | `history/mufasa_joint_without_fusion_bn` |
| `_code_backups/mufasa_before_joint_20261013` | `history/mufasa_frozen_increment` |
| `_code_backups/a3pfn_mldc_patch_compensation_20261016_incomplete_20261016_024438` | `history/patch_compensation_incomplete` |
| `_code_backups/mldc_patch_before_grm_v2_20261016_070501` | `history/patch_compensation_zero_initialized` |
| `a3pfn_mldc_dual_supervision_20261019/source_snapshots/nightreid_three_heads_20261019` | `mldc_last_cls_dual_supervision/history/nightreid_three_head_supervision` |
| `a3pfn_mldc_lpam/.codex_backups/aligned_training_20261011_153157` | `mldc_tar_refinement/history/lpam_joint_aligned` |
| `a3pfn_mldc_lpam/.codex_backups/before_tar_20261012_160539` | `mldc_tar_refinement/history/mldc_guided_spatial_readout` |
| `a3pfn_mldc_lpam/.codex_backups/epochs120_20261011_143626` | `mldc_tar_refinement/history/lpam_training_schedule` |
| `a3pfn_mldc_lpam/.codex_backups/guided_readout_20261012_111623` | `mldc_tar_refinement/history/guided_cross_layer_reader` |
| `a3pfn_mldc_lpam/.codex_backups/guided_readout_20261012_112359` | `mldc_tar_refinement/history/guided_cross_layer_evaluation_fix` |
| `a3pfn_mldc_lpam/.codex_backups/joint_20261011_135700` | `mldc_tar_refinement/history/lpam_joint_unaligned` |
| `a3pfn_mldc_lpam/.codex_backups/lpam_v3_adaptive_20261011_192424` | `mldc_tar_refinement/history/lpam_pyramid_adaptive` |
| `a3pfn_mldc_lpam/.codex_backups/no_initial_eval_20261011_141444` | `mldc_tar_refinement/history/lpam_without_initial_evaluation` |
| `a3pfn_mldc_lpam/.codex_backups/standard_eval_20261011_143003` | `mldc_tar_refinement/history/lpam_standard_evaluation` |
