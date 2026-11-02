"""Code verification only: no optimizer updates or training entry invocation."""
import argparse
from pathlib import Path
import random
import re
import torch


def reset_seed():
    import numpy as np
    random.seed(1234);np.random.seed(1234);torch.manual_seed(1234)


def real_batch(cfg):
    import numpy as np
    from PIL import Image
    groups={}
    for p in sorted(Path('data/night600/bounding_box_train').glob('*.jpg')):
        match=re.match(r'(\d{6})_c([1-8])_',p.name)
        if match and int(match[1])>0:
            groups.setdefault(int(match[1]),{}).setdefault(int(match[2])-1,[]).append(p)
    eligible=sorted(p for p,c in groups.items() if len(c)>1)
    rng=np.random.RandomState(1234)
    chosen=[eligible[i] for i in rng.permutation(len(eligible))[:2]]
    relabel={p:i for i,p in enumerate(sorted(groups))}
    images,labels,cameras=[],[],[]
    for pid in chosen:
        for cam in sorted(groups[pid])[:2]:
            paths=groups[pid][cam];path=paths[int(rng.randint(len(paths)))]
            with Image.open(path) as im:
                arr=np.asarray(im.convert('RGB').resize((128,256),Image.Resampling.BICUBIC)).copy()
            images.append(torch.from_numpy(arr).permute(2,0,1).float()/255)
            labels.append(relabel[pid]);cameras.append(cam)
    images=torch.stack(images)
    mean=torch.tensor(cfg.INPUT.PIXEL_MEAN)[None,:,None,None]
    std=torch.tensor(cfg.INPUT.PIXEL_STD)[None,:,None,None]
    return (images-mean)/std,torch.tensor(labels),torch.tensor(cameras),len(groups)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    from .verify_readonly import run
    run(args.report)


if __name__=='__main__':main()
