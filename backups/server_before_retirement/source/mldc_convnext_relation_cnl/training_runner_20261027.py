"""Detached supervisor for the authorized 120-epoch student experiment."""
import csv
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    run = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    root = Path(__file__).resolve().parent.parent
    status_path = run/'status.json'
    state = dict(stage='starting', runner_pid=os.getpid(), child_pid=None,
        output=str(output), seed=1234, epochs=120, teacher_frozen=True)

    def write():
        state['updated_at'] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
        temporary = run/'status.tmp'
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(status_path)

    write()
    try:
        environment = os.environ.copy()
        environment.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='0')
        with (run/'student.console.log').open('x', buffering=1) as log:
            child = subprocess.Popen([sys.executable, '-u', '-m', 'mldc_convnext_relation_cnl.train'],
                cwd=str(root), env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
            state.update(stage='training', child_pid=child.pid)
            write()
            while child.poll() is None:
                progress = output/'training.csv'
                if progress.is_file():
                    with progress.open() as handle:
                        rows = list(csv.DictReader(handle))
                    if rows:
                        state['last_completed_epoch'] = int(rows[-1]['epoch'])
                        state['last_training'] = rows[-1]
                best = output/'best_metrics.json'
                if best.is_file():
                    state['best_metrics'] = json.loads(best.read_text())
                write()
                time.sleep(10)
            code = child.wait()
        state.update(stage='complete' if code == 0 else 'failed', exit_code=code)
        if code == 0:
            with (output/'training.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            state['last_completed_epoch'] = int(rows[-1]['epoch'])
            if state['last_completed_epoch'] != 120 or not (output/'transformer_120.pth').is_file():
                raise RuntimeError('Training exited without the required 120-epoch checkpoint')
        write()
    except Exception as error:
        state.update(stage='failed', error=repr(error))
        write()
        raise


if __name__ == '__main__':
    main()
