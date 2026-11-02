"""Compare independently trained MLDC and matched last-CLS experiments."""
import argparse
import csv
import json
from pathlib import Path


def read_run(path, expected_readout):
    import yaml
    path = Path(path)
    metadata = json.loads((path / 'run_metadata.json').read_text(encoding='utf-8'))
    if metadata['readout'] != expected_readout:
        raise ValueError(f'{path} has readout {metadata["readout"]}, expected {expected_readout}')
    config = yaml.safe_load(metadata['config'])
    config['MODEL'].pop('MLDC_READOUT')
    config.pop('OUTPUT_DIR')
    with (path / 'evaluation.csv').open(encoding='utf-8') as stream:
        rows = [{key: int(value) if key == 'epoch' else float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]
    if not rows:
        raise ValueError(f'No completed evaluation in {path}')
    return metadata, config, sorted(rows, key=lambda row: row['epoch'])


def compare(mldc_output, control_output, output):
    metadata, config, mldc = read_run(mldc_output, 'mldc')
    control_metadata, control_config, control = read_run(control_output, 'last_cls')
    if config != control_config or metadata['seed'] != control_metadata['seed']:
        raise ValueError('Experiment settings differ beyond readout and output path')
    by_epoch = {row['epoch']: row for row in control}
    common = []
    for row in mldc:
        if row['epoch'] not in by_epoch:
            continue
        other = by_epoch[row['epoch']]
        common.append({'epoch': row['epoch'], **{
            f'{method}_{metric}': values[metric]
            for method, values in (('mldc', row), ('last_cls', other))
            for metric in ('mAP', 'rank1', 'rank5', 'rank10')},
            **{f'delta_{metric}_pp': 100*(row[metric]-other[metric])
               for metric in ('mAP', 'rank1', 'rank5', 'rank10')}})
    if not common:
        raise ValueError('No shared evaluation epochs')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(common[0]))
        writer.writeheader()
        writer.writerows(common)
    summary = {'seed': metadata['seed'], 'common_epoch_count': len(common),
               'last_common_epoch': common[-1],
               'mldc_best_mAP_epoch': max(mldc, key=lambda row: row['mAP']),
               'last_cls_best_mAP_epoch': max(control, key=lambda row: row['mAP']),
               'note': 'Metrics are fractions; deltas are percentage points. Same seed, independently '
                       'trained matched control; one seed does not establish statistical significance.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metric in zip(axes, ('mAP', 'rank1')):
        for method, label in (('mldc', 'MLDC'), ('last_cls', 'Matched last CLS')):
            ax.plot([r['epoch'] for r in common], [100*r[f'{method}_{metric}'] for r in common], 'o-', label=label)
        ax.set(xlabel='Epoch', ylabel=f'{metric} (%)', title='Independent training | same seed and settings')
        ax.legend()
        ax.grid(alpha=.2)
    fig.savefig(output / 'comparison.png', dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print('Saved comparison to', output)
    return summary


def main():
    parser = argparse.ArgumentParser(description='MLDC versus independent matched last-CLS control')
    parser.add_argument('--mldc_output', default='../../log/transreid/night600/vit_base_ics_a3pfn_mldc')
    parser.add_argument('--control_output', default='../../log/transreid/night600/vit_base_ics_a3pfn_last_cls')
    parser.add_argument('--output', default='../../log/transreid/night600/vit_base_ics_a3pfn_comparison')
    args = parser.parse_args()
    compare(args.mldc_output, args.control_output, args.output)


if __name__ == '__main__':
    main()
