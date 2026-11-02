"""Evaluate only an explicitly selected PFE checkpoint."""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    import json
    import torch
    from config import cfg as base_cfg
    from mldc_feature_enhancement.pfe_v2_model import VARIANT, config_from_dump, PFEV2ReID
    from mldc_feature_enhancement.pfe_processor import evaluate
    saved = torch.load(args.weight, map_location='cpu', mmap=True, weights_only=False)
    if saved['variant'] != VARIANT:
        raise ValueError('Use test_pfe_v2.py only with a PFE checkpoint')
    cfg = config_from_dump(base_cfg, saved['config']); cfg.MODEL.DEVICE = args.device; cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.MODEL.DEVICE_ID)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('No CUDA device; select CPU explicitly if desired')
    from datasets import make_dataloader
    _, _, loader, nq, nc, cameras, views = make_dataloader(cfg)
    model = PFEV2ReID(cfg, nc, cameras, views, initialize=False)
    assert model.architecture() == saved['architecture']
    model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    model.to(args.device)
    metrics, diagnostics = evaluate(cfg, model, loader, nq, args.device)
    report = dict(checkpoint=args.weight, epoch=saved['epoch'], metrics=metrics, diagnostics=diagnostics)
    target = Path(args.weight).parent/f'evaluation_pfe_v2_epoch{saved["epoch"]}.json'
    target.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
