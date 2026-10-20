"""All three retrieval branches, with chunked distances and paired query outcomes."""
import csv,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .model import BRANCHES


def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def evaluate_arrays(fusion,last,pids,cameras,nq,paths=None,chunk=32,concat=None):
    """Evaluate actual descriptors; raw/independent-BN concat is not mean cosine."""
    from utils.metrics import eval_func
    pid=np.asarray(pids);cam=np.asarray(cameras);n=len(pid);ng=n-nq
    if not (0<nq<n) or fusion.shape!=last.shape or fusion.shape!=(n,768):
        raise ValueError('Expected aligned query-first arrays [N,768] for both branches')
    if concat is None or concat.shape!=(n,1536):raise ValueError('Supply the actual 1536D concat descriptor')
    if chunk<1:raise ValueError('Query chunk must be positive')
    paths=paths or [str(i) for i in range(n)]
    totals={name:dict(cmc=np.zeros(min(50,ng)),ap=0.,valid=0) for name in BRANCHES}
    rows={name:[] for name in BRANCHES}
    for name,array in [('fusion',fusion),('last',last),('concat',concat)]:
        gallery=F.normalize(torch.from_numpy(np.array(array[nq:],copy=True)),dim=1)
        if not torch.isfinite(gallery).all() or float(gallery.norm(dim=1).min())<.999:raise ValueError('Invalid gallery descriptor')
        for begin in range(0,nq,chunk):
            end=min(begin+chunk,nq)
            query=F.normalize(torch.from_numpy(np.array(array[begin:end],copy=True)),dim=1)
            if not torch.isfinite(query).all() or float(query.norm(dim=1).min())<.999:raise ValueError('Invalid query descriptor')
            distance=(query.square().sum(1,keepdim=True)+gallery.square().sum(1)[None]-2*query@gallery.T).numpy()
            batch=[]
            for j in range(end-begin):
                i=begin+j;order=np.argsort(distance[j])
                order=order[~((pid[nq:][order]==pid[i])&(cam[nq:][order]==cam[i]))]
                match=pid[nq:][order]==pid[i];pos=np.flatnonzero(match)
                if not len(pos):continue
                batch.append(dict(query_index=i,pid=int(pid[i]),camera=int(cam[i]),query_path=paths[i],
                    AP=float(np.mean(np.arange(1,len(pos)+1)/(pos+1))),rank1=int(match[0]),
                    rank5=int(match[:5].any()),rank10=int(match[:10].any()),first_positive_rank=int(pos[0]+1),
                    top1_gallery_index=int(order[0]),top1_gallery_path=paths[nq+int(order[0])]))
            if batch:
                cmc,ap=eval_func(distance,pid[begin:end],pid[nq:],cam[begin:end],cam[nq:])
                assert abs(ap-np.mean([r['AP'] for r in batch]))<1e-8
                assert abs(float(cmc[0])-np.mean([r['rank1'] for r in batch]))<1e-6
                if len(cmc)<len(totals[name]['cmc']):
                    cmc=np.pad(cmc,(0,len(totals[name]['cmc'])-len(cmc)),constant_values=float(cmc[-1]))
                totals[name]['cmc']+=cmc*len(batch);totals[name]['ap']+=ap*len(batch);totals[name]['valid']+=len(batch)
                rows[name].extend(batch)
        del gallery,query,distance
    metrics={}
    for name,total in totals.items():
        if not total['valid']:raise RuntimeError('No valid evaluation queries')
        cmc=total['cmc']/total['valid']
        metrics[name]=dict(mAP=total['ap']/total['valid'],rank1=float(cmc[0]),rank5=float(cmc[min(4,len(cmc)-1)]),
            rank10=float(cmc[min(9,len(cmc)-1)]),valid_queries=total['valid'],correct=sum(r['rank1'] for r in rows[name]))
    valid=[r['query_index'] for r in rows['concat']]
    assert all([r['query_index'] for r in rows[name]]==valid for name in BRANCHES)
    ok={name:np.asarray([r['rank1'] for r in rows[name]],dtype=bool) for name in BRANCHES}
    union=ok['fusion']|ok['last']
    comparisons={}
    for name in ('fusion','last'):
        comparisons['concat_vs_'+name]=dict(corrected=int((ok['concat']&~ok[name]).sum()),
            broken=int((~ok['concat']&ok[name]).sum()),net=int(ok['concat'].sum()-ok[name].sum()))
    comparisons.update(single_branch_union=int(union.sum()),either_branch_correct_concat_wrong=int((union&~ok['concat']).sum()),
        both_branches_wrong_concat_correct=int((~union&ok['concat']).sum()),
        fusion_only_correct=int((ok['fusion']&~ok['last']).sum()),last_only_correct=int((ok['last']&~ok['fusion']).sum()))
    return metrics,comparisons,rows


@torch.inference_mode()
def evaluate(cfg,model,loader,nq,device,output):
    model.eval();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    n=len(loader.dataset);cache_path=output/'branch_features_server_only.npy'
    slices={'fusion':slice(0,768),'last':slice(768,1536),'concat':slice(1536,3072)}
    features=np.lib.format.open_memmap(cache_path,mode='w+',dtype=np.float32,shape=(n,3072))
    pids=[];cameras=[];paths=[];offset=0
    shares=[]
    for images,pid,cam,camera,view,path in loader:
        data=model(images.to(device),cam_label=camera.to(device),view_label=view.to(device),diagnostics=True)
        size=len(images)
        for name in BRANCHES:
            value=data['retrieval'][name].float().cpu().numpy()
            if not np.isfinite(value).all():raise FloatingPointError('Non-finite retrieval features')
            features[offset:offset+size,slices[name]]=value
        # Per-image energy fraction, not a supervised or learned scalar weight.
        z=data['retrieval']['concat'].float()
        shares.extend((z[:,:768].square().sum(1)/z.square().sum(1).clamp_min(1e-12)).cpu().tolist())
        offset+=size;pids.extend(map(int,pid));cameras.extend(map(int,cam));paths.extend(path)
    if offset!=n:raise RuntimeError('Evaluation loader/image count mismatch')
    features.flush()
    metrics,comparisons,rows=evaluate_arrays(features[:,slices['fusion']],features[:,slices['last']],pids,cameras,nq,paths,
        cfg.TEST.DUAL_QUERY_CHUNK,concat=features[:,slices['concat']])
    for name in BRANCHES:write_csv(output/f'queries_{name}.csv',rows[name])
    (output/'image_order_server_only.json').write_text(json.dumps(dict(pids=pids,cameras=cameras,paths=paths,num_query=nq)))
    result=dict(metrics=metrics,comparisons=comparisons,images=n,num_query=nq,num_gallery=n-nq,
        neck_feat=cfg.TEST.NECK_FEAT,
        concat_fusion_energy_share=dict(mean=float(np.mean(shares)),min=float(np.min(shares)),max=float(np.max(shares))),
        protocol=f'BN-{cfg.TEST.NECK_FEAT}; actual 1536D descriptor with global L2; no branch-wise L2; no reranking; same-PID same-camera exclusion')
    (output/'results.json').write_text(json.dumps(result,indent=2));del features
    return metrics,comparisons
