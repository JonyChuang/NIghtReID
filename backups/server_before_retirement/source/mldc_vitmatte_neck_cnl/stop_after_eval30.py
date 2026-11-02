"""Stop the already-running Night600 job after its complete epoch30 evaluation."""
import csv
import json
import os
from pathlib import Path
import signal
import time

PACKAGE = Path(__file__).resolve().parent
INSPECTION = json.loads((PACKAGE / 'switch_inspection_20261025.json').read_text())
OUTPUT = Path(json.loads(INSPECTION['metadata']['config'].replace('\n', '\\n'))) if False else Path(
    __import__('yaml').safe_load(INSPECTION['metadata']['config'])['OUTPUT_DIR'])
PID = INSPECTION['training_main_processes'][0]['pid']
STATUS = PACKAGE / 'stop_epoch30_status.json'


def rows(name):
    path = OUTPUT / name
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def alive(pid):
    try:
        return (Path('/proc') / str(pid) / 'stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def record(state, **kwargs):
    data = dict(state=state, monitor_pid=os.getpid(), training_pid=PID,
        requested_stop_after_epoch=30, time=time.strftime('%Y-%m-%d %H:%M:%S'), output=str(OUTPUT), **kwargs)
    pending = STATUS.with_suffix('.pending')
    pending.write_text(json.dumps(data, indent=2)); pending.replace(STATUS)
    print(json.dumps(data), flush=True)


def main():
    command = [s.decode(errors='replace') for s in (Path('/proc') / str(PID) / 'cmdline').read_bytes().split(b'\0') if s]
    assert 'mldc_vitmatte_neck_cnl.train' in command
    assert 'mldc_vitmatte_neck_cnl/configs/neck_all.yml' in command
    record('waiting_for_epoch30_evaluation')
    while alive(PID):
        evaluations = rows('evaluation.csv')
        target = next((r for r in evaluations if int(r['epoch']) == 30), None)
        if target:
            try:
                # The processor writes branch diagnostics before logging and
                # checkpointing. Wait for the full JSON row AND the best save.
                lines = (OUTPUT / 'evaluation_diagnostics.jsonl').read_text().splitlines()
                detail = next((json.loads(line) for line in reversed(lines) if json.loads(line)['epoch'] == 30), None)
                expected_best = max(evaluations, key=lambda r: float(r['mAP']))
                best = json.loads((OUTPUT / 'best_metrics.json').read_text())
                ready = detail is not None and int(best['epoch']) == int(expected_best['epoch'])
            except (OSError, ValueError, KeyError):
                ready = False
            if ready:
                workers = []
                for proc in Path('/proc').glob('[0-9]*/status'):
                    try:
                        if any(line == 'PPid:\t' + str(PID) for line in proc.read_text().splitlines()):
                            args = (proc.parent / 'cmdline').read_bytes().split(b'\0')
                            if b'mldc_vitmatte_neck_cnl.train' in args:
                                workers.append(int(proc.parent.name))
                    except OSError:
                        pass
                os.kill(PID, signal.SIGINT)
                for _ in range(150):
                    if not alive(PID):
                        break
                    time.sleep(.1)
                if alive(PID):
                    os.kill(PID, signal.SIGTERM)
                for worker in workers:
                    if alive(worker):
                        try:
                            os.kill(worker, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                record('stopped_after_epoch30_evaluation', evaluation=target, best_metrics=best,
                    checkpoint30_available=int(best['epoch']) == 30, latest_training=rows('training.csv')[-1],
                    training_process_alive=alive(PID), nightreid_started=False,
                    note='Existing process was stopped externally after evaluation and checkpoint writes completed.')
                return
        time.sleep(.05)
    record('process_ended_before_monitor_stop', latest_training=rows('training.csv')[-1] if rows('training.csv') else None)


if __name__ == '__main__':
    main()
