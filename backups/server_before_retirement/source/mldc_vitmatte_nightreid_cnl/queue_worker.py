"""Wait for the existing training to finish, then run the requested first ablation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
OUT = Path('/root/log/transreid/nightreid/mldc_vitmatte_nightreid_cnl/dcm_rgb_gamma085_alpha01_seed1234_schedule120')
STATUS = PACKAGE / 'queue_status.json'


def active_training():
    active = []
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = [a.decode(errors='replace') for a in proc.read_bytes().split(b'\0') if a]
            if args and 'python' in args[0] and any(a.endswith('.train') or a.endswith('/train.py') or a == 'train.py' for a in args):
                active.append(int(proc.parent.name))
        except OSError:
            pass
    return active


def status(state, **values):
    record = dict(state=state, worker_pid=os.getpid(), time=time.strftime('%Y-%m-%d %H:%M:%S'), output=str(OUT), **values)
    STATUS.write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


def main():
    check = json.loads((PACKAGE / 'integration_cpu.json').read_text())
    assert check['all_passed'] and check['train_ids'] == 500 and not check['SIE_CAMERA']
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError('Refuse to replace experiment output')
    previous = None
    while True:
        blockers = active_training()
        # Training can briefly be between processes; require two consecutive
        # idle checks before acquiring the GPU for the new experiment.
        state = bool(blockers)
        if state != previous:
            status('waiting_for_current_training' if state else 'confirming_gpu_idle', blocking_pids=blockers)
            previous = state
        if not blockers:
            time.sleep(15)
            if not active_training():
                break
        time.sleep(30)
    config = 'mldc_vitmatte_nightreid_cnl/configs/dcm_rgb.yml'
    command = [sys.executable, '-u', '-m', 'mldc_vitmatte_nightreid_cnl.train', '--config_file', config]
    status('training', config=config)
    result = subprocess.run(command, cwd=str(ROOT))
    if result.returncode:
        status('failed', exit_code=result.returncode)
        return result.returncode
    analysis = subprocess.run([sys.executable, '-u', '-m', 'mldc_vitmatte_nightreid_cnl.compare_baseline', '--output', str(OUT)], cwd=str(ROOT))
    status('complete' if analysis.returncode == 0 else 'analysis_failed', exit_code=analysis.returncode)
    return analysis.returncode


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        status('failed', error=str(error))
        raise
