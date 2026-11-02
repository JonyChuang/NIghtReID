"""Run the CNL-off control independently of the SSH log viewer."""
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def last_row(output, name):
    path = output / name
    if not path.exists():
        return None
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    return rows[-1] if rows else None


def save_status(phase, **values):
    path = HERE / 'status.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(phase=phase, monitor_pid=os.getpid(),
                 time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values)
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(state, indent=2))
    pending.replace(path)


def main():
    plan = json.loads((HERE / 'plan.json').read_text())
    output, console = Path(plan['output']), Path(plan['console'])
    with (HERE / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not output.exists(), 'Control output already exists'
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1',
                           PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        with console.open('x') as log:
            log.write('NightReID | learned MLDC | CNL OFF (weight 0)\n')
            log.write('Matched schedule | 120 epochs | batch 64 | seed 1234\n')
            log.write('Only CNL weight and output directory differ from the reference.\n\n')
            log.flush()
            process = subprocess.Popen(
                [sys.executable, '-u', plan['entrypoint'], '--config_file', plan['config']],
                cwd=plan['project'], env=environment, stdout=log,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        launch = dict(pid=process.pid, monitor_pid=os.getpid(), output=str(output),
                      console=str(console), config=plan['config'], runtime=str(HERE),
                      epochs=120, batch=64, seed=1234, cnl_weight=0.0,
                      training_started=True)
        (HERE / 'launch.json').write_text(json.dumps(launch, indent=2))
        save_status('training_starting', **{k: v for k, v in launch.items() if k != 'monitor_pid'})
        try:
            while process.poll() is None:
                train = last_row(output, 'training.csv')
                save_status('training_running' if train else 'training_starting',
                            latest_training=train, latest_evaluation=last_row(output, 'evaluation.csv'))
                time.sleep(10)
            train, final = last_row(output, 'training.csv'), last_row(output, 'evaluation.csv')
            assert process.returncode == 0, 'Training exited with code %d' % process.returncode
            assert train and final and int(train['epoch']) == 120 and int(final['epoch']) == 120
            assert (output / 'transformer_120.pth').exists()
            best = json.loads((output / 'best_metrics.json').read_text())
            save_status('completed', training_completed=True, final_metrics=final,
                        best_metrics=best, exit_code=0)
            with console.open('a') as log:
                log.write('\nCNL-off control completed. Final: mAP %.4f%%, Rank-1 %.4f%%\n' %
                          (100 * float(final['mAP']), 100 * float(final['rank1'])))
                log.write('Results: ' + str(output) + '\n')
        except BaseException as error:
            save_status('failed', error=str(error), exit_code=process.poll())
            with console.open('a') as log:
                log.write('\nControl monitor: ' + str(error) + '\n')
            raise


if __name__ == '__main__':
    main()
