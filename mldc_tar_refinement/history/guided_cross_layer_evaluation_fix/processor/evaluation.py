"""Original ReID evaluator, applied once to the main retrieval descriptor."""
import csv
from pathlib import Path

import torch
from utils.metrics import R1_mAP_eval


@torch.no_grad()
def evaluate(cfg, model, loader, num_query, device):
    model.eval()
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                            reranking=cfg.TEST.RE_RANKING)
    evaluator.reset()
    for images, pids, camids, camera, view, _ in loader:
        feature = model(images.to(device), cam_label=camera.to(device), view_label=view.to(device))
        evaluator.update((feature, pids, camids))
    cmc, mean_ap, *_ = evaluator.compute()
    return {'mAP': float(mean_ap), 'rank1': float(cmc[0]),
            'rank5': float(cmc[4]), 'rank10': float(cmc[9])}


def append_csv(path, rows):
    if not rows:
        return
    with Path(path).open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)


def record_evaluation(output, epoch, metrics):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    append_csv(output/'evaluation.csv', [{'epoch': epoch, **metrics}])
