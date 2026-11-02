SERVER_ROOT = '/root/TransReID-SSL-main/transreid_pytorch'
"""CPU-only correctness checks. Never call optimizer.step or a training loop."""
import ast
import gc
import hashlib
import importlib
import io
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

root = Path(SERVER_ROOT).resolve()
target = root / 'mldc_patch_grm'
sys.path.insert(0, str(root))
if (Path('/sys/fs/cgroup/memory.max').exists()
        and Path('/sys/fs/cgroup/memory.max').read_text().strip() != 'max'):
    remaining = int(Path('/sys/fs/cgroup/memory.max').read_text()) - int(Path('/sys/fs/cgroup/memory.current').read_text())
    if remaining < 3 * 1024 ** 3:
        raise RuntimeError('Insufficient server memory for model and checkpoint verification; need 3 GiB free')
torch.set_num_threads(2)
torch.manual_seed(1234)
package = target.name
module = importlib.import_module(package + '.model.make_model_mldc')
patch_module = importlib.import_module(package + '.model.patch_compensation')
processor = importlib.import_module(package + '.processor.processor_mldc')
from config import cfg as defaults
cfg = module.add_mldc_config(defaults.clone())
cfg.merge_from_file(str(target / 'variants/frozen_global_patch_grm/config.yml'))
cfg.MODEL.DEVICE = 'cpu'
cfg.freeze()
report = {'training_started': False, 'optimizer_steps': 0, 'device': 'cpu',
          'cuda_available': torch.cuda.is_available(), 'checks': {}}
checks = report['checks']
for p in target.rglob('*.py'):
    ast.parse(p.read_text(), filename=str(p))
checks['syntax_and_imports'] = True
model = module.make_model(cfg, 300, 8, 1)
trainable = [n for n,p in model.named_parameters() if p.requires_grad]
assert trainable and all(n.startswith(('patch_compensation.', 'classifier.')) for n in trainable)
report['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
report['patch_parameters'] = sum(p.numel() for p in model.patch_compensation.parameters())
assert model.checkpoint_epoch == 0 and model.baseline_epoch == 120
assert model.patch_compensation.layers == tuple(range(1, 13))
checks['baseline_epoch_and_trainable_scope'] = True
baseline_module = importlib.import_module('mldc_deep4_readout.model.make_model_mldc')
baseline_cfg = cfg.clone()
baseline_cfg.defrost()
baseline_cfg.MODEL.MLDC_TRAIN_FEATURE = 'mldc'
baseline_cfg.MODEL.PRETRAIN_PATH = '/root/log/transreid/night600/vit_base_ics_a3pfn_mldc/deep4_uniform_start_120/transformer_120.pth'
baseline_cfg.freeze()
baseline = baseline_module.make_model(baseline_cfg, 300, 8, 1).eval()
for name, tensor in baseline.state_dict().items():
    assert torch.equal(tensor, model.state_dict()[name]), name
checks['all_anchor_tensors_equal_original_checkpoint'] = True
images = torch.randn(4, 3, 256, 128)
camera = torch.tensor([0, 0, 1, 1])
view = torch.zeros(4, dtype=torch.long)
model.eval()
with torch.no_grad():
    expected, expected_raw, expected_layers = baseline(images, cam_label=camera, view_label=view, return_components=True)
    data = model(images, cam_label=camera, view_label=view, return_patch_diagnostics=True)
assert torch.equal(data['output'], expected)
assert torch.equal(data['anchor'], expected_raw)
assert torch.equal(data['cls_features'], expected_layers)
assert torch.count_nonzero(data['delta']).item() == 0
assert data['output'].shape == (4, 768)
assert data['patch_attention'].shape == (4, 4, 12, 128)
assert torch.allclose(data['layer_mass'].sum(-1) + data['null_mass'], torch.ones(4, 4), atol=1e-6)
report['initial_descriptor_max_abs_difference'] = (data['output'] - expected).abs().max().item()
report['initial_null_mass_per_head'] = data['null_mass'].mean(0).tolist()
report['loaded_cls_weights'] = model.readout_weights().tolist()
checks['zero_compensation_matches_original_descriptor_exactly'] = True
checks['attention_shape_and_probability_conservation'] = True
del baseline, expected, expected_raw, expected_layers
gc.collect()
original_bn = {n: t.clone() for n,t in model.bottleneck.state_dict().items()}
model.train()
assert model.training and model.patch_compensation.training and model.classifier.training
assert not model.base.training and not model.mldc.training and not model.bottleneck.training
assert all(not m.training for m in model.base.modules())
from solver import make_optimizer
loss_module = importlib.import_module(package + '.loss.make_loss_mldc')
loss_fn, center = loss_module.make_loss(cfg, 300)
optimizer, _ = make_optimizer(cfg, model, center)
module.configure_head_learning_rates(cfg, model, optimizer)
assert {id(p) for g in optimizer.param_groups for p in g['params']} == {
    id(p) for p in model.parameters() if p.requires_grad}
for group in optimizer.param_groups:
    p = group['params'][0]
    name = next(n for n, v in model.named_parameters() if v is p)
    expected_lr = cfg.SOLVER.BASE_LR * (cfg.SOLVER.BIAS_LR_FACTOR if 'bias' in name else 1)
    if name.startswith('classifier.'):
        expected_lr *= cfg.SOLVER.MLDC_HEAD_LR_FACTOR
    assert abs(group['lr'] - expected_lr) < 1e-12, name
checks['optimizer_contains_only_patch_and_classifier'] = True
labels = torch.tensor([0, 0, 1, 1])
scores, features = model(images, labels, cam_label=camera, view_label=view)
loss = loss_fn(scores, features, labels, camera)
loss.backward()
gradient = model.patch_compensation.output.weight.grad
assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0
assert model.classifier.weight.grad is not None and model.classifier.weight.grad.norm() > 0
assert all(p.grad is None for p in model.base.parameters())
assert all(p.grad is None for p in model.mldc.parameters())
assert all(p.grad is None for p in model.bottleneck.parameters())
assert all(torch.equal(original_bn[n], t) for n,t in model.bottleneck.state_dict().items())
report['finite_loss_without_update'] = loss.item()
checks['identity_triplet_backward_and_frozen_bn'] = True
model.zero_grad(set_to_none=True)
# Test later-stage gradient flow through a functional weight substitution. The
# module's real zero-initialized output weights are never changed or optimized.
model.eval()
with torch.no_grad():
    layers, patches = model.base(images, cam_label=camera, view_label=view,
                                return_patches=True, patch_layers=model.patch_compensation.layers)
    anchor = model.mldc(layers)
replacement = (torch.randn_like(model.patch_compensation.output.weight) * 0.01).requires_grad_()
delta = torch.func.functional_call(model.patch_compensation, {'output.weight': replacement}, (anchor, patches))
delta.square().mean().backward()
for name, p in model.patch_compensation.named_parameters():
    if name == 'output.weight':
        continue
    assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0, name
assert torch.isfinite(replacement.grad).all() and replacement.grad.norm() > 0
assert torch.count_nonzero(model.patch_compensation.output.weight).item() == 0
checks['query_key_value_layer_norm_and_null_receive_gradients'] = True
model.zero_grad(set_to_none=True)
# All-equal content logits: null and real-token pool each have mass 1/2.
probe = patch_module.PatchCompensation(layers=(9,10,11,12))
with torch.no_grad():
    probe.query.weight.zero_()
    probe.query_token.zero_()
    _, info = probe(anchor, patches[-4:], return_attention=True)
assert torch.allclose(info['null_mass'], torch.full((4,4), 0.5), atol=1e-6)
with torch.no_grad():
    probe.output.weight.normal_(std=0.01)
    probe.null_bias.fill_(80)
    suppressed, _ = probe(anchor, patches[-4:], return_attention=True)
assert suppressed.abs().max() < 1e-25
no_null = patch_module.PatchCompensation(layers=(9,10,11,12), use_null=False)
with torch.no_grad():
    _, info = no_null(anchor, patches[-4:], return_attention=True)
assert torch.count_nonzero(info['null_mass']) == 0
assert torch.allclose(info['layer_mass'].sum(-1), torch.ones(4,4), atol=1e-6)
checks['null_count_prior_can_abstain_and_tail_only_ablation'] = True
del probe, no_null, replacement, delta, anchor, patches, layers, scores, features, optimizer, center
gc.collect()
# Real serialization and strict loading, entirely in server RAM.
buffer = io.BytesIO()
processor.save_checkpoint(model, cfg, 0, buffer)
buffer.seek(0)
restored = module.make_model(cfg, 300, 8, 1)
restored.load_param(buffer)
assert all(torch.equal(v, restored.state_dict()[n]) for n,v in model.state_dict().items())
checks['new_checkpoint_roundtrip_strict_loading'] = True
buffer.seek(0)
restored.patch_compensation.heads = 8
try:
    restored.load_param(buffer)
except ValueError:
    checks['mismatched_attention_architecture_rejected'] = True
else:
    raise AssertionError('Mismatched patch architecture was accepted')
finally:
    restored.patch_compensation.heads = 4
buffer.close()
del restored
gc.collect()
model.zero_grad(set_to_none=True)
attribution_input = images[:1].clone().requires_grad_()
model(attribution_input, cam_label=camera[:1], view_label=view[:1]).square().mean().backward()
assert attribution_input.grad is not None and torch.isfinite(attribution_input.grad).all()
assert attribution_input.grad.norm() > 0
checks['frozen_backbone_still_supports_input_attribution'] = True
model.zero_grad(set_to_none=True)
for name in ('train', 'test'):
    importlib.import_module(package + '.' + name)
    result = subprocess.run([sys.executable, str(target / (name + '.py')), '--help'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    assert result.returncode == 0, result.stderr.decode()
checks['entrypoint_imports_and_help_without_training'] = True
manifest = json.loads((target / 'deployment_manifest.json').read_text())
source = Path(manifest['source_package'])
current = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
assert current == manifest['source_file_digests']
original = Path(manifest['baseline_checkpoint'])
assert [original.stat().st_size, original.stat().st_mtime_ns] == manifest['baseline_stat']
assert not list(Path(cfg.OUTPUT_DIR).glob('transformer_*.pth'))
checks['original_code_and_checkpoint_unchanged'] = True
report['status'] = 'passed'
verification = target / 'verification'
verification.mkdir(exist_ok=True)
(verification / 'report.json').write_text(json.dumps(report, indent=2))
print('VERIFICATION', json.dumps(report, indent=2))
