"""Fixed unlabeled NFC targets; a frozen training-ID classifier for probability KD."""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from PIL import Image


def reciprocal_nfc(unit,k=2,block_size=256):
    """Original NFC: squared-Euclidean mutual top-2 and simultaneous raw sum."""
    if unit.ndim!=2 or len(unit)<=k or not torch.isfinite(unit).all():
        raise ValueError('NFC requires finite descriptors and more than k images')
    nearest=[];norm=unit.square().sum(1)
    for start in range(0,len(unit),block_size):
        stop=min(start+block_size,len(unit))
        dist=norm[start:stop,None]+norm[None,:]-2*unit[start:stop]@unit.T
        dist[torch.arange(stop-start,device=unit.device),torch.arange(start,stop,device=unit.device)]=1000
        nearest.append(dist.topk(k,largest=False).indices)
    nearest=torch.cat(nearest)
    mutual=(nearest[nearest]==torch.arange(len(unit),device=unit.device)[:,None,None]).any(2)
    return unit+(unit[nearest]*mutual[:,:,None]).sum(1),nearest,mutual


class CleanTrainingImages(Dataset):
    def __init__(self,entries,transform):
        # Only paths and stable indices enter target construction, not labels/cameras.
        self.paths=[str(r[0]) for r in entries];self.transform=transform
    def __len__(self):return len(self.paths)
    def __getitem__(self,index):
        with Image.open(self.paths[index]) as im:image=self.transform(im.convert('RGB'))
        return image,index


class ClassProbabilityKD(nn.Module):
    def __init__(self,weight,log_targets,scale,temperature):
        super().__init__()
        if temperature<=0 or scale<=0 or not torch.isfinite(log_targets).all():
            raise ValueError('Invalid NFC probability calibration')
        self.register_buffer('weight',weight.detach().float().clone())
        self.register_buffer('log_targets',log_targets.detach().float().clone())
        self.scale=float(scale);self.temperature=float(temperature)
    def forward(self,unit,indices,student_logits=None):
        # FP32 outside AMP: T^2 KL(teacher || student), not average over classes.
        with torch.autocast(device_type=unit.device.type,enabled=False):
            logits=(self.scale*F.linear(unit.float(),self.weight) if student_logits is None else student_logits.float())
            if logits.shape!=(len(indices),self.log_targets.shape[1]):
                raise ValueError('Student and teacher must share the same training identity class order')
            student=F.log_softmax(logits/self.temperature,dim=1)
            teacher=self.log_targets[indices].detach()
            kl=(teacher.exp()*(teacher-student)).sum(1).mean()*self.temperature**2
            entropy=-(teacher.exp()*teacher).sum(1).mean()
        return kl,{'teacher_entropy':float(entropy.detach()),'probability_kl':float(kl.detach()/self.temperature**2)}


@torch.no_grad()
def build_fixed_targets(cfg,model,dataset,clean_transform,device,logger):
    """Called once on baseline-initialized eval model, before any optimizer step."""
    was_training=model.training;model.eval()
    clean=CleanTrainingImages(dataset.entries,clean_transform)
    loader=DataLoader(clean,batch_size=cfg.NFC.EXTRACT_BATCH_SIZE,shuffle=False,
                      num_workers=cfg.DATALOADER.NUM_WORKERS,pin_memory=True)
    features=torch.empty((len(clean),model.descriptor_dim),device=device)
    lengths=torch.empty(len(clean),device=device);seen=torch.zeros(len(clean),dtype=torch.bool,device=device)
    for iteration,(images,indices) in enumerate(loader,1):
        indices=indices.to(device)
        neck=model(images.to(device))
        if not torch.isfinite(neck).all():raise FloatingPointError('Non-finite frozen teacher')
        features[indices]=F.normalize(neck.float(),dim=1);lengths[indices]=neck.float().norm(dim=1);seen[indices]=True
        if iteration==1 or iteration%50==0:logger.info('NFC clean extraction %d/%d',int(seen.sum()),len(clean))
    if not seen.all():raise RuntimeError('Incomplete teacher extraction')
    raw_sum,nearest,mutual=reciprocal_nfc(features,block_size=cfg.NFC.KNN_BLOCK_SIZE)
    target=F.normalize(raw_sum,dim=1) if cfg.NFC.TARGET=='nfc' else features
    scale=float(lengths.median());weight=model.classifier.weight.detach().float().clone()
    temperature=cfg.NFC.TEMPERATURE
    log_targets=F.log_softmax(scale*F.linear(target,weight)/temperature,dim=1)
    kd=ClassProbabilityKD(weight,log_targets,scale,temperature).to(device)
    info=dict(images=len(clean),classes=weight.shape[0],target=cfg.NFC.TARGET,k1=2,k2=2,
        mutual_edges=int(mutual.sum()),coverage=float(mutual.any(1).float().mean()),
        shared_feature_norm_scale=scale,temperature=temperature,weight=cfg.NFC.WEIGHT,
        source_checkpoint=cfg.NFC.TEACHER_CHECKPOINT,dataset_fingerprint=dataset.fingerprint,
        label_filter=False,camera_filter=False,cnl_memory_used=False,refresh='fixed before training',
        student_probability=cfg.NFC.STUDENT_PROBABILITY,
        scale_note=('Median clean norm restores teacher aggregate logits; student uses its native ID logits'
                    if cfg.NFC.STUDENT_PROBABILITY=='student_classifier' else
                    'Median norm over all clean training images; shared fixed scale for teacher/student'),
        inference='single-image original student; no NFC or teacher classifier')
    out=Path(cfg.OUTPUT_DIR)
    torch.save(dict(weight=weight.cpu(),log_targets=log_targets.cpu(),metadata=info,
                    nearest=nearest.cpu(),mutual=mutual.cpu()),out/'nfc_teacher_targets.pth')
    (out/'nfc_teacher_metadata.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    logger.info('Fixed NFC teacher: %s',json.dumps(info))
    model.train(was_training)
    return kd
