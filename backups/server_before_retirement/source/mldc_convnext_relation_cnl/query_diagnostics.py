"""Per-query retrieval records with the original evaluation filter."""
import json
from pathlib import Path
import torch


def query_statistics(distance,pids,camids,nq):
    """Per-query AP/Rank-1 using the repository's same-ID/same-camera filter."""
    import numpy as np
    pids=np.asarray(pids);camids=np.asarray(camids)
    gallery_pids=pids[nq:];gallery_cams=camids[nq:]
    rows=[]
    for i,distances in enumerate(distance):
        order=np.argsort(distances)
        keep=~((gallery_pids[order]==pids[i])&(gallery_cams[order]==camids[i]))
        order=order[keep];matches=gallery_pids[order]==pids[i]
        valid=bool(matches.any())
        ap=float(((matches.cumsum()/np.arange(1,len(matches)+1))*matches).sum()/matches.sum()) if valid else None
        rows.append(dict(index=i,pid=int(pids[i]),camera=int(camids[i]),valid=valid,
            rank1=bool(matches[0]) if valid else None,ap=ap,
            top_gallery_index=int(order[0]) if len(order) else None))
    return rows

