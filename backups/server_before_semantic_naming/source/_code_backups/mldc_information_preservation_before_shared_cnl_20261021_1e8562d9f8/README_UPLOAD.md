# MLDC information preservation

Uploaded as an independent module. Original ID + Triplet + 0.1 * preservation.
No added cross-camera contrastive CNL; positive camera unrestricted.
Train explicitly from the existing project root:

```bash
cd /root/TransReID-SSL-main/transreid_pytorch
python mldc_information_preservation/train.py
```

Config: mldc_information_preservation/configs/night600.yml.
This upload does not launch training. Local verification uses generated CPU images, not retrieval results.
