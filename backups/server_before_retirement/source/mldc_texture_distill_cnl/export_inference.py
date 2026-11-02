"""Remove training-only teacher and adapter weights from a student checkpoint."""
import argparse
from pathlib import Path
import torch


def export(source,destination):
    destination=Path(destination)
    if destination.exists(): raise FileExistsError(str(destination))
    checkpoint=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
    core={k:v for k,v in checkpoint['state_dict'].items()
        if not k.startswith(('texture_teacher.','texture_adapter.'))}
    result={k:v for k,v in checkpoint.items() if k not in ['state_dict','architecture','cnl_memory']}
    result.update(state_dict=core,training_architecture=checkpoint.get('architecture'),
        variant='texture_distilled_original_mldc_inference_20261026')
    torch.save(result,destination)
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('source'); parser.add_argument('destination')
    args=parser.parse_args(); print(export(args.source,args.destination))
