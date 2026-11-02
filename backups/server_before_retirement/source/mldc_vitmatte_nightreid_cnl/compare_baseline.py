"""Compare complete-gallery retrieval to the original fixed NightReID run."""
import argparse
import json
from pathlib import Path


def paired(old, new):
    assert len(old) == len(new) == 1394
    valid = []
    for a, b in zip(old, new):
        assert all(a[k] == b[k] for k in ('index', 'pid', 'camera', 'valid'))
        if a['valid']:
            valid.append((a, b))
    gain = sum(max(b['ap'] - a['ap'], 0.) for a, b in valid) / len(valid) * 100
    loss = sum(max(a['ap'] - b['ap'], 0.) for a, b in valid) / len(valid) * 100
    corrected = sum(not a['rank1'] and b['rank1'] for a, b in valid)
    broken = sum(a['rank1'] and not b['rank1'] for a, b in valid)
    return dict(queries=len(valid), ap_gain_pp=gain, ap_loss_pp=loss, mAP_delta_pp=gain-loss,
        corrected=corrected, broken=broken, net_correct=corrected-broken,
        rank1_delta_pp=(corrected-broken)/len(valid)*100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    reference = json.loads((Path(__file__).parent / 'baseline_reference.json').read_text())
    best = json.loads((output / 'best_metrics.json').read_text())
    records = [json.loads(line) for line in (output / 'evaluation_diagnostics.jsonl').read_text().splitlines() if line.strip()]
    result = dict(protocol='NightReID full gallery, distractors included, SIE off, seed1234; exploratory single-seed comparison',
        baseline_output=reference['output'], experiment_output=str(output), target=dict(mAP=53.2,rank1=68.1))
    for name, epoch in [('best', best['epoch']), ('final', 120)]:
        row = next(row for row in records if row['epoch'] == epoch)
        old = reference[name]
        values = row['readouts']['fused']
        result[name] = dict(epoch=epoch, baseline_epoch=old['epoch'],
            metrics_percent={k:100*v for k,v in values.items()},
            versus_independent_baseline=paired(old['queries'], row['queries']['fused']),
            versus_same_checkpoint_global=paired(row['queries']['global'], row['queries']['fused']))
    path = output / 'comparison_to_original.json'
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
