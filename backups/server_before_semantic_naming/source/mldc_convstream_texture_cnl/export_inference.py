"""Export PAD inference weights while retaining the learned student head."""
import argparse
from pathlib import Path
import torch

def export(source,destination):
    destination=Path(destination)
    if destination.exists():raise FileExistsError(str(destination))
    checkpoint=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
    if checkpoint.get('variant')!='mldc_pad_multilevel_cls_20261029':
        raise ValueError('Expected the multilevel CLS refinement variant; older checkpoints need their backed-up source')
    result={k:v for k,v in checkpoint.items() if k not in ['cnl_memory','prototype_memory','optimizer','scheduler','scaler']}
    result['inference_only']=True
    # Keep all four CLS refiners: they contribute to the final MLDC retrieval descriptor.
    torch.save(result,destination)
    return destination

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination')
    a=p.parse_args();print(export(a.source,a.destination))
