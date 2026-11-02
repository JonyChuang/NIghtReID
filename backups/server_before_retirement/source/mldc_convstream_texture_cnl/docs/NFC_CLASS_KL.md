# NFC identity probability distillation

## Current requested protocol: full120-epoch training from SSL

```bash
cd /root/TransReID-SSL-main/transreid_pytorch
/root/miniconda3/envs/QREID/bin/python -u -m mldc_convstream_texture_cnl.train_nfc_kl --config_file mldc_convstream_texture_cnl/configs/nfc_class_kl_from_ssl_nightreid.yml
```

This is now the default configuration for `train_nfc_kl`. Student backbone loads `pretrain_model/vit_base_ics_cfs_lup.pth`; MLDC starts uniformly and the student500-class classifier is newly initialized and trainable. No student parameters are copied from epoch110. The frozen teacher still loads the trained original epoch110 baseline solely to generate train-only NFC probabilities before training, then is released.

KD uses `softmax(student ID logits/T)`, with gradient through the student's own classifier and ViT/MLDC/BN. Teacher and student share the same500 training identity indices, not classifier weights. Teacher NFC probabilities remain fixed and detached. The objective is ID+Triplet+0.1CNL+1.0T-squared KL at T4. This differs intentionally from the earlier frozen/fine-tuning probe's fixed teacher projection.

Defaults:120 epochs, SGD base LR4e-4,20 warmup epochs, PK64/K4, seed1234, evaluation every10 epochs, checkpoints every40 epochs plus best. Output: `/root/log/transreid/nightreid/mldc_convstream_texture_cnl/nfc_class_kl/from_ssl_seed1234_120`. It requires an empty directory. Checkpoints contain model and CNL state but omit unused optimizer/scheduler/scaler copies because this entry has no resume implementation. Test inference remains the original single-image student.

## Previous optional protocol: baseline fine-tuning

Train inside the existing `mldc_convstream_texture_cnl` package. No new experiment source copy is required.

```bash
cd /root/TransReID-SSL-main/transreid_pytorch
/root/miniconda3/envs/QREID/bin/python -u -m mldc_convstream_texture_cnl.train_nfc_kl --config_file mldc_convstream_texture_cnl/configs/nfc_class_kl_nightreid.yml
```

The command requires a visible CUDA GPU. It prepares train-only targets before training; it does not start from SSL or resume an older optimizer. Both teacher and student initialize from the original MLDC+CNL epoch110 best checkpoint configured in the YAML. The output directory must be empty.

## Method

- Re-extract every clean training image in deterministic evaluation mode. Training indices are verified against the baseline dataset fingerprint before generating targets. Query and gallery never enter teacher construction.
- Apply original unlabeled mutual top2 NFC with simultaneous, unweighted raw summation. There is no identity/camera filtering and no CNL memory input.
- Normalize the sum's direction for classification and multiply by a common fixed scale: median BN feature norm over the complete clean training set. This replaces the 32-image scale calibration used by the frozen probe. Both teacher and student use this same scale.
- The original frozen 500-class classifier generates teacher soft probabilities at T=4. Targets and classifier weights are frozen. Every image retains its own NFC target, including its own feature when no reciprocal neighbor exists.
- Fine-tune the original student ViT, MLDC and BN for20 epochs. The existing ID classifier is also frozen by default, retaining its shared identity coordinate system. No extra student feature head is added. BN continues training normally.
- Optimize `ID + Triplet + 0.1*CNL + 1.0*T^2*KL(teacher || student)`. Original CNL uses gamma0.9 and `all_nonself`; its own training memory remains part of the baseline loss. It is not used to construct NFC targets. No frozen-probe feature preservation penalty is added; the original supervised losses provide the joint training objective.
- Inference uses the original single-image student. No teacher, NFC aggregation or target artifact is required.

Defaults: SGD base LR4e-5, original head LR multipliers,20 epochs,2 warmup epochs, PK batch64/K4, seed1234, evaluation every5 epochs, checkpoint at20 epochs and best mAP. This is short fine-tuning, not the original SSL-initialized120-epoch schedule.

The initial single-image baseline is evaluated before the first optimizer step. Logs include `nfc_class_kl_T2`, `nfc_probability_kl`, `nfc_weighted` and teacher entropy with eight-decimal step logging. Target metadata, nearest indices and fixed teacher class log probabilities are saved in the run directory. The targets do not refresh online. AMP-skipped optimizer steps keep the original CNL update behavior. This entry does not implement resume.

## Same-budget raw teacher control

Append to the training command:

```bash
NFC.TARGET raw OUTPUT_DIR /root/log/transreid/nightreid/mldc_convstream_texture_cnl/nfc_class_kl/raw_seed1234_ft20
```

This uses original clean single-image class probabilities with exactly the same fine-tuning budget.

## Verification

```bash
/root/miniconda3/envs/QREID/bin/python -m mldc_convstream_texture_cnl.verify_nfc_kl --integration
```

Checks the original NFC sum, classwise KL reduction and temperature factor, teacher gradient isolation, stable repeated indices, real baseline initialization, official dataset ordering, full original ID/Triplet/CNL plus KD backward, and teacher-free inference. CPU checks do not establish CUDA/AMP training performance.
