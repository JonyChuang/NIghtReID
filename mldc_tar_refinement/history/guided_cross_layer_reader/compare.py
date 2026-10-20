"""Compare independent main-retrieval endpoints."""
import argparse
import csv
import json
from pathlib import Path


def read_csv(path):
    with Path(path).open(encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def compare(baseline, lpam, output):
    baseline, lpam, output = Path(baseline), Path(lpam), Path(output)
    metadata = json.loads((lpam/'run_metadata.json').read_text(encoding='utf-8'))
    if metadata.get('variant') != 'mldc_lpam_pyramid_adaptive_v3':
        raise ValueError('This comparison expects the new free-query run subdirectory')
    rows = read_csv(lpam/'evaluation.csv')
    baseline_rows = read_csv(baseline/'evaluation.csv')
    baseline_epoch = max(int(r['epoch']) for r in baseline_rows)
    baseline_fixed = next((r for r in baseline_rows if int(r['epoch']) == baseline_epoch), None)
    if baseline_fixed is None:
        raise ValueError('Missing baseline reference epoch')
    endpoint = int(metadata['primary_endpoint'])
    final = next((r for r in rows if int(r['epoch']) == endpoint), None)
    diagnostic_rows = []  # Routine evaluation records only the main retrieval feature.
    summary = {'baseline_checkpoint_epoch': baseline_epoch, 'baseline_fixed': baseline_fixed,
               'training_epochs': endpoint, 'completed': final is not None,
               'primary_endpoint': final,
               'note': 'Independent joint training from generic backbone initialization; no trained MLDC checkpoint is loaded. Compare training budgets before interpreting baseline deltas.'}
    if final:
        summary['delta_vs_baseline_pp'] = {k: 100*(float(final[k])-float(baseline_fixed[k])) for k in ('mAP', 'rank1')}
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, metric in zip(axes, ('mAP', 'rank1')):
        ax.plot([int(r['epoch']) for r in rows], [100*float(r[metric]) for r in rows], 'o-', label='Joint 1792D')
        ax.axhline(100*float(baseline_fixed[metric]), linestyle='--', color='gray', label='Independent MLDC epoch {}'.format(baseline_epoch))
        ax.set(xlabel='Training epoch', ylabel=metric+' (%)'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(output/'comparison.png', dpi=150); plt.close(fig)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='/root/log/transreid/night600/vit_base_ics_a3pfn_mldc')
    parser.add_argument('--lpam', default='/root/log/transreid/night600/vit_base_ics_a3pfn_mldc_lpam/free_queries_v3_pyramid_adaptive_120')
    parser.add_argument('--output', default='/root/log/transreid/night600/vit_base_ics_a3pfn_mldc_lpam/free_queries_v3_pyramid_adaptive_120/comparison')
    args = parser.parse_args()
    compare(args.baseline, args.lpam, args.output)
