"""Preserve original global supervision; train the actual fused retrieval path."""

def supervised_losses(data,targets,cameras,loss_fn,cfg):
    global_loss=loss_fn(data['global_scores'],data['global_raw'],targets,cameras)
    terms={'global_id':float(loss_fn.last_terms['identity']),
           'global_triplet':float(loss_fn.last_terms['triplet'])}
    fused_loss=global_loss*0.
    terms.update(fused_id=0.,fused_triplet=0.)
    if 'fused_scores' in data:
        fused_loss=loss_fn(data['fused_scores'],data['fused_unit'],targets,cameras)
        terms.update(fused_id=float(loss_fn.last_terms['identity']),fused_triplet=float(loss_fn.last_terms['triplet']))
    return global_loss+cfg.SOLVER.DETAIL_LOSS_WEIGHT*fused_loss,terms
