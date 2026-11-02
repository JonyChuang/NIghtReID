"""Synthetic GPU integration check, run only after the predecessor releases GPU."""
from pathlib import Path
import json
import torch
from config import cfg as defaults
from .model.make_model_mldc import add_mldc_config,make_model
from .loss.make_loss_mldc import make_loss
from .neighbor_memory import NeighborMemory
from .global_cnl import GlobalCNL


def main():
    cfg=add_mldc_config(defaults.clone())
    cfg.merge_from_file(str(Path(__file__).parent/'configs/neighbor_night600.yml'))
    torch.manual_seed(1234)
    device=torch.device('cuda:0')
    model=make_model(cfg,300,6,1,initialize=False).to(device).train()
    bank=NeighborMemory(8,768,candidate_k=7,neighbors=7,warmup_epochs=20).to(device)
    bank.update(torch.randn(8,768,device=device),torch.arange(8,device=device),0)
    all_labels=torch.tensor([0,0,1,1,0,0,1,1],device=device)
    all_cameras=torch.tensor([0,1,0,1,1,0,1,0],device=device)
    cnl=GlobalCNL(all_labels,all_cameras).to(device)
    criterion,_=make_loss(cfg,300)
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    scaler=torch.cuda.amp.GradScaler()
    images=torch.randn(4,3,256,128,device=device)
    indices=torch.arange(4,device=device);labels=all_labels[indices];cameras=all_cameras[indices]
    for epoch in [1,21]:
        bank.refresh_graph(epoch)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            score,raw,unit,terms=model(images,labels,cam_label=cameras,
                return_cnl_features=True,return_distillation=True,sample_indices=indices,
                epoch=epoch,neighbor_memory=bank)
            loss=criterion(score,raw,labels,batchformer_scores=terms['batchformer_scores'],
                           batchformer_rows=terms['batchformer_rows'])
            neighbor_loss,_=cnl(unit,indices,labels,cameras)
            loss=loss+.1*neighbor_loss
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
        applied=scaler.get_scale()>=old_scale
        bank.update(raw,indices,epoch,applied=applied)
        if applied:cnl.update(unit,indices,labels,cameras)
        assert len(terms['batchformer_scores'])==(0 if epoch==1 else 4)
    model.eval()
    with torch.inference_mode():
        descriptor=model(images[:1]);assert descriptor.shape==(1,768)
    report=dict(passed=True,device=str(device),synthetic_images=True,dataset_access=False,
                warmup_and_active_amp_steps=True,cnl_and_neighbor_memories=True,
                single_image_inference=True)
    (Path(__file__).parent/'gpu_smoke.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':main()
