# Support-only multi-view cross-attention experiment

Dedicated fork of the current MLDC+CNL support-only cross-attention version. Original BatchFormer experiments remain in mldc_convstream_texture_cnl.

Train from the server project root:

```bash
/root/miniconda3/envs/QREID/bin/python -u -m mldc_mvi_cross_cnl.train_mvi_cross --config_file mldc_mvi_cross_cnl/configs/mvi_cross_nightreid.yml
```

The default configuration enables group ID (1.0) and one-way IP (0.1) fully from epoch1; original learning-rate warmup20, SSL initialization, seed1234, 120 epochs and batch64/K4 remain. One rotating anchor per ID receives support-only supervision from 2-3 other independent images. Blocks9-12 projections, norms, FFN, MLDC and classifier are shared; no teacher or EMA copy. Test input is an independent single image. Results: /root/log/transreid/nightreid/mldc_mvi_cross_cnl/from_ssl_seed1234_120.

This copy and restoration were checked by static source/config parsing and file readback only. No model execution or training was performed.
