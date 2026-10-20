"""Real SSL/Night600 integration preflight, without launching training."""
import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    import torch
    from config import cfg as base_cfg
    from datasets import make_dataloader
    from solver import make_optimizer
    from mldc_locat_attention.model.make_model_locat import add_mldc_config, make_model, configure_head_learning_rates
    from mldc_locat_attention.loss.make_loss_mldc import make_loss
    from mldc_locat_attention.processor.processor_mldc import record_weights
    from mldc_locat_attention.processor.visualization_locat import save_visualizations
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='mldc_locat_attention/verification_real_ssl')
    args = parser.parse_args()
    torch.set_num_threads(8)
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    cfg = add_mldc_config(base_cfg.clone())
    cfg.merge_from_file('mldc_locat_attention/variants/gaussian_attention_prr/config.yml')
    # A small genuine P x K minibatch suffices for gradient and optimizer checks.
    # The training configuration remains batch=64 with four images per identity.
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.TEST.IMS_PER_BATCH = 8
    cfg.TEST.MLDC_VIS_SAMPLES = 2
    cfg.OUTPUT_DIR = args.output
    cfg.freeze()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('verification')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    train_loader, _, val_loader, nq, nc, cams, views = make_dataloader(cfg)
    model = make_model(cfg, nc, cams, views).to(args.device)
    loss_fn, center = make_loss(cfg, nc)
    optimizer, _ = make_optimizer(cfg, model, center)
    configure_head_learning_rates(cfg, model, optimizer)
    images, targets, cameras, view = next(iter(train_loader))
    model.train()
    scores, features = model(images.to(args.device), targets.to(args.device),
                             cam_label=cameras.to(args.device), view_label=view.to(args.device))
    loss = loss_fn(scores, features, targets.to(args.device))
    loss.backward()
    gradients = {}
    for index, block in enumerate(model.base.blocks, 1):
        gradients[index] = {name: float(p.grad.norm()) for name, p in block.attn.gaug.named_parameters()}
        assert all(np.isfinite(value) and value > 0 for value in gradients[index].values()), gradients[index]
    assert model.mldc.layer_weights.grad is not None
    # Do not step SGD: epoch-zero maps must correspond to the SSL initialization.
    model.zero_grad(set_to_none=True)
    # Restore initial BN running statistics after the gradient-only train forward.
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
    # Reload pretrained BN running statistics (stem) exactly; also preserves the
    # independently initialized, unchanged GAug and classifier parameters.
    model.base.load_param(cfg.MODEL.PRETRAIN_PATH, hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
    model.eval()
    record_weights(model, 0, output, logger)
    save_visualizations(cfg, model, val_loader, nq, 0, logger)
    with torch.no_grad():
        detail = model(images[:2].to(args.device), return_details=True)
    assert detail['retrieval'].shape == (2, 768) and torch.isfinite(detail['retrieval']).all()
    gaug_parameters = sum(p.numel() for block in model.base.blocks for p in block.attn.gaug.parameters())
    report = {'device': args.device, 'cuda_available': torch.cuda.is_available(), 'torch': torch.__version__,
              'checks': {'real_ssl_loaded': True, 'real_pk_minibatch_backward': True,
                         'all_12_gaug_layers_nonzero_finite_gradients': True,
                         'fixed_night600_query_visualizations': True, 'optimizer_parameter_groups': len(optimizer.param_groups)},
              'locat_spec': model.locat_spec, 'additional_gaug_parameters': gaug_parameters,
              'prr_parameters': sum(p.numel() for p in model.prr.parameters()),
              'losses': {key: float(value) for key, value in loss_fn.last_terms.items()},
              'gaug_gradient_norms': gradients, 'initial_layer_weights': model.readout_weights().detach().cpu().tolist(),
              'num_query': nq, 'num_gallery': len(val_loader.dataset) - nq, 'num_classes': nc,
              'full_training_started': False, 'full_retrieval_evaluated': False,
              'note': 'No optimizer step. Real data/SSL compatibility and diagnostics only; no trained result.'}
    (output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'gaug_gradient_norms'}, indent=2))


if __name__ == '__main__':
    main()
