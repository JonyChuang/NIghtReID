"""Evaluate only an explicitly selected CPPI checkpoint."""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--compare_off', action='store_true', help='Full-query on/off metrics and corrected/broken query statistics')
    args = parser.parse_args()
    import json
    import torch
    from config import cfg as base_cfg
    from mldc_feature_enhancement.cppi_model import VARIANT, CNN_SDCL_VARIANT, CNN_FUSED_VARIANT, CNN_LEARNABLE_FUSED_VARIANT,CNN_CROSS_VARIANT, GLOBAL_CNL_VARIANT, config_from_dump, CPPIReID, CNNShallowDeepReID, MLDCCNLReID
    from mldc_feature_enhancement.cppi_processor import evaluate
    saved = torch.load(args.weight, map_location='cpu', mmap=True, weights_only=False)
    if saved['variant'] not in (VARIANT,CNN_SDCL_VARIANT,CNN_FUSED_VARIANT,CNN_LEARNABLE_FUSED_VARIANT,CNN_CROSS_VARIANT,GLOBAL_CNL_VARIANT):
        raise ValueError('Use test_cppi.py only with a CPPI checkpoint')
    cfg = config_from_dump(base_cfg, saved['config']); cfg.MODEL.DEVICE = args.device; cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.MODEL.DEVICE_ID)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('No CUDA device; select CPU explicitly if desired')
    from datasets import make_dataloader
    _, _, loader, nq, nc, cameras, views = make_dataloader(cfg)
    is_sd=saved['variant'] in (CNN_SDCL_VARIANT,CNN_FUSED_VARIANT,CNN_LEARNABLE_FUSED_VARIANT,CNN_CROSS_VARIANT)
    is_global=saved['variant']==GLOBAL_CNL_VARIANT
    model_class=MLDCCNLReID if is_global else CNNShallowDeepReID if is_sd else CPPIReID
    model=model_class(cfg,nc,cameras,views,initialize=False)
    assert model.architecture() == saved['architecture']
    model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    model.to(args.device)
    if is_sd or is_global:
        from mldc_feature_enhancement.cnn_sdcl_processor import evaluate as evaluate_sd
        metrics,diagnostics=evaluate_sd(cfg,model,loader,nq,args.device,query_details=True)
        report=dict(checkpoint=args.weight,epoch=saved['epoch'],metrics=metrics,diagnostics=diagnostics)
        tag='mldc_cnl' if is_global else 'cnn_sdcl'
        target=Path(args.weight).parent/f'evaluation_{tag}_epoch{saved["epoch"]}.json'
        target.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
        return
    metrics, diagnostics = evaluate(cfg, model, loader, nq, args.device, query_details=args.compare_off)
    report = dict(checkpoint=args.weight, epoch=saved['epoch'], metrics=metrics, diagnostics=diagnostics)
    if args.compare_off:
        from mldc_feature_enhancement.cppi_diagnostics import summarize_query_changes
        enabled = model.enabled
        try:
            model.enabled = False
            off_metrics, off_diagnostics = evaluate(cfg, model, loader, nq, args.device, query_details=True)
            report.update(off_metrics=off_metrics, off_diagnostics=off_diagnostics,
                query_changes=summarize_query_changes(diagnostics['query_statistics'], off_diagnostics['query_statistics']))
        finally:
            model.enabled = enabled
    target = Path(args.weight).parent/f'evaluation_cppi_epoch{saved["epoch"]}{"_on_off" if args.compare_off else ""}.json'
    target.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
