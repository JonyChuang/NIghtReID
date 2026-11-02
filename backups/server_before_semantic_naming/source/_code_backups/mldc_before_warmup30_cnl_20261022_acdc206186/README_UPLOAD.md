# Current default: fused-space nomination, three-layer consensus, uncertain exclusion

Run from the project root:

```bash
python mldc_information_preservation/train.py
```

Config: `configs/night600_shared_cnl_consensus.yml`.
Output: `./output/mldc_information_preservation/deep4_uniform_cnl_fused09_support3_uncertain08_120`.

The existing learnable MLDC and original ID/Triplet remain. A positive must pass the fused 0.9 relative threshold and at least three same-layer 0.9 relative thresholds. Candidates passing any layer or fused relaxed 0.80 threshold, but not positive, are uncertain and excluded from the CNL denominator. Remaining eligible candidates are negatives. Only spaces with positive maxima nominate neighbors. CNL contrasts final fused BN-after L2 descriptors and directly updates fusion logits, BN and backbone; all mining/key paths are detached. CNL never reads ground-truth PID.

One logical image-index memory stores [N,4,768] mining keys and [N,768] final retrieval keys, synchronized after successful optimizer steps. It covers all training image indices and fills online; uninitialized slots never participate. Cross-camera candidates, self exclusion and duplicate coalescing remain.

See `FUSED_CONSENSUS_CNL_20261022.md` for definitions and validation. The historical `night600_shared_cnl.yml` retains the union policy and its old output directory; all completed results remain. This code update does not start training.
