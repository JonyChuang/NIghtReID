"""Meaningful NFC aggregation, identity-KL gradients, and optional real-model checks."""
import argparse,importlib.util,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
spec=importlib.util.spec_from_file_location('nfc_kd',Path(__file__).with_name('nfc_class_distillation.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--integration',action='store_true')
    parser.add_argument('--config_file',default=str(Path(__file__).parent/'configs/nfc_class_kl_from_ssl_nightreid.yml'))
    args=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(12)
    x=F.normalize(torch.randn(19,8),dim=1)
    summed,rank,mutual=module.reciprocal_nfc(x,block_size=4)
    dist=torch.cdist(x,x).square();dist.fill_diagonal_(1000);ref_rank=dist.topk(2,largest=False).indices
    expected=x.clone()
    for i in range(len(x)):
        for j in ref_rank[i]:
            if i in ref_rank[j]:expected[i]+=x[j]
    assert torch.equal(rank,ref_rank) and torch.allclose(expected,summed,atol=1e-6)
    weight=torch.randn(5,8,requires_grad=True);source=torch.randn(19,5,requires_grad=True)
    logt=F.log_softmax(source,dim=1);kd=module.ClassProbabilityKD(weight,logt,2.,4.)
    student=torch.randn(3,8,requires_grad=True);indices=torch.tensor([7,1,7])
    loss,_=kd(F.normalize(student,dim=1),indices)
    logits=2*F.linear(F.normalize(student,dim=1),weight.detach())/4
    expected_loss=F.kl_div(F.log_softmax(logits,dim=1),logt.detach()[indices].exp(),reduction='batchmean')*16
    assert torch.allclose(loss,expected_loss,atol=2e-6)
    loss.backward()
    assert student.grad is not None and student.grad.abs().sum()>0
    assert source.grad is None and weight.grad is None and not list(kd.parameters())
    learned_logits=torch.randn(3,5,requires_grad=True)
    loss,_=kd(F.normalize(student.detach(),dim=1),indices,student_logits=learned_logits)
    loss.backward();assert learned_logits.grad is not None and learned_logits.grad.abs().sum()>0
    print('PASS original NFC mutual aggregation; stable duplicate indices; T^2 class KL; student-only gradient')
    if args.integration:
        from config import cfg as defaults
        from datasets import make_dataloader
        from mldc_convstream_texture_cnl.model.make_model_mldc import add_mldc_config,make_model
        from mldc_convstream_texture_cnl.indexed_data import indexed_loader
        from mldc_convstream_texture_cnl.global_cnl import GlobalCNL
        from mldc_convstream_texture_cnl.loss.make_loss_mldc import make_loss
        cfg=add_mldc_config(defaults.clone());cfg.merge_from_file(args.config_file)
        cfg.DATALOADER.NUM_WORKERS=0
        loader,_,val,nq,nc,cameras,views=make_dataloader(cfg);loader,data=indexed_loader(loader)
        cp=torch.load(cfg.NFC.TEACHER_CHECKPOINT,map_location='cpu',mmap=True,weights_only=False)
        assert cp['training_dataset_fingerprint']==data.fingerprint
        assert (len(data),nq,len(val.dataset)-nq,nc)==(15514,1394,26111,500)
        if cfg.NFC.STUDENT_INIT=='ssl':
            assert Path(cfg.MODEL.PRETRAIN_PATH).is_file()
            model=make_model(cfg,nc,cameras,views,initialize=True)
            assert not torch.equal(model.classifier.weight.detach(),cp['state_dict']['classifier.weight'])
            assert torch.equal(model.mldc.layer_weights.detach(),torch.zeros_like(model.mldc.layer_weights))
        else:
            model=make_model(cfg,nc,cameras,views,initialize=False);model.load_baseline(cfg.NFC.TEACHER_CHECKPOINT)
        if cfg.NFC.FREEZE_ID_CLASSIFIER:model.classifier.requires_grad_(False)
        model.train();images=torch.randn(4,3,256,128);labels=torch.tensor([0,0,1,1]);cameras=torch.tensor([0,1,0,1])
        scores,feat,unit,pad=model(images,labels,return_cnl_features=True,return_distillation=True)
        teacher=F.log_softmax(torch.randn(4,500),dim=1)
        kd=module.ClassProbabilityKD(cp['state_dict']['classifier.weight'],teacher,32.,4.)
        loss,_=kd(unit,torch.arange(4),student_logits=scores if cfg.NFC.STUDENT_PROBABILITY=='student_classifier' else None)
        original,_=make_loss(cfg,nc)
        memory=GlobalCNL(labels.tolist(),cameras.tolist(),dim=768,gamma=.9,temperature=.05,momentum=.2,threshold_reference='all_nonself')
        cnl,_=memory(unit,torch.arange(4),labels,cameras)
        total=original(scores,feat,labels,cameras)+.1*cnl+loss
        assert torch.isfinite(total);total.backward()
        assert model.mldc.layer_weights.grad is not None and model.mldc.layer_weights.grad.abs().sum()>0
        assert model.base.patch_embed.proj.weight.grad is not None
        if cfg.NFC.FREEZE_ID_CLASSIFIER:assert model.classifier.weight.grad is None
        else:assert model.classifier.weight.grad is not None and model.classifier.weight.grad.abs().sum()>0
        model.eval()
        with torch.no_grad():assert model(images).shape==(4,768)
        print(json.dumps({'integration':'PASS real500ID model, baseline initialization, dataset ordering, ID+Triplet+CNL+KD backward, teacher-free inference',
            'student_init':cfg.NFC.STUDENT_INIT,'student_probability':cfg.NFC.STUDENT_PROBABILITY,
            'cuda_available':torch.cuda.is_available(),'architecture':model.architecture()}))


if __name__=='__main__':main()
