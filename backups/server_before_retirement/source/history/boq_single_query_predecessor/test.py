import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    import json
    import torch
    from yacs.config import CfgNode
    from datasets import make_dataloader
    from mldc_boq_local_readout.model import build_model
    from mldc_boq_local_readout.processor import evaluate
    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if checkpoint.get('experiment') != 'mldc_joint_local_v1_20261016':
        raise ValueError('Use a checkpoint from this joint experiment')
    cfg = CfgNode.load_cfg(checkpoint['config']); cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    if not torch.cuda.is_available():
        raise RuntimeError('Full checkpoint evaluation requires CUDA; no local fallback')
    _, _, loader, nq, nc, cameras, views = make_dataloader(cfg)
    model = build_model(cfg, nc, cameras, views, initialize=False)
    if checkpoint['architecture'] != model.architecture():
        raise ValueError('Checkpoint architecture differs')
    model.load_state_dict(checkpoint['state_dict'], strict=True, assign=True)
    device = torch.device('cuda:0'); model.to(device)
    metrics, diagnostics = evaluate(cfg, model, loader, nq, device)
    print(json.dumps({'metrics':metrics, 'diagnostics':diagnostics}, indent=2))


if __name__ == '__main__':
    main()
