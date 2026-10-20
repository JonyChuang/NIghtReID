"""Strictly reproduce a historical checkpoint after archiving its model code."""
import json
from pathlib import Path
import numpy as np
import torch


def verify_legacy_checkpoint(root):
    from config import cfg as base_cfg
    from datasets import make_dataloader
    from .test import checkpoint_api
    from .verify import release
    run=Path('/root/log/transreid/night600/vit_base_ics_mldc_dual_20261019/nightreid_cat3_lastw03_catw10_before_120')
    checkpoint=run/'transformer_70.pth';cache=run/'evaluation/epoch_070/branch_features_server_only.npy'
    if not checkpoint.exists() or not cache.exists():
        result=dict(available=False,reason='Epoch70 checkpoint/cache unavailable; archived model retained')
        print('LEGACY_CHECKPOINT_CHECK',json.dumps(result),flush=True);return result
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    api,_=checkpoint_api(saved['variant']);cfg=api.config_from_dump(base_cfg,saved['config'])
    cfg.DATALOADER.NUM_WORKERS=0;cfg.TEST.IMS_PER_BATCH=4;cfg.freeze()
    _,_,loader,nq,nc,cameras,views=make_dataloader(cfg)
    model=api.DualMLDCReID(cfg,nc,cameras,views,initialize=False)
    model.load_state_dict(saved['state_dict'],strict=True,assign=True);model.eval()
    images,pid,cam,camera,view,paths=next(iter(loader))
    image_order=json.loads((cache.parent/'image_order_server_only.json').read_text())
    assert list(map(int,pid))==image_order['pids'][:len(images)]
    assert list(paths)==image_order['paths'][:len(images)]
    with torch.inference_mode():actual=model(images,cam_label=camera,view_label=view)
    reference=np.load(cache,mmap_mode='r')[:len(images),1536:]
    error=float(np.max(np.abs(actual.numpy()-reference)))
    assert error<1e-4,error
    result=dict(available=True,checkpoint=str(checkpoint),strict_load=True,
        original_readout_preserved=cfg.TEST.NECK_FEAT,real_images=len(images),max_absolute_error=error,
        archived_model_used=True,training_modified=False,optimizer_steps=0)
    del model,saved,loader,actual;release()
    print('LEGACY_CHECKPOINT_VERIFIED',json.dumps(result),flush=True)
    return result
