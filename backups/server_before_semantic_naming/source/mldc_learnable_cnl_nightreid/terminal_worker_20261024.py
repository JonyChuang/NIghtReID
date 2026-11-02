"""Keep the server training alive independently of its live-log terminal."""
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
STATUS=HERE/'queue_status.json'


def row(output,name):
    path=output/name
    if not path.exists():return None
    with path.open() as stream:rows=list(csv.DictReader(stream))
    return rows[-1] if rows else None


def status(phase,**values):
    data=json.loads(STATUS.read_text()) if STATUS.exists() else {}
    data.update(phase=phase,worker_pid=os.getpid(),time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**values)
    pending=STATUS.with_suffix('.pending');pending.write_text(json.dumps(data,indent=2));pending.replace(STATUS)


def start_next_if_pending(environment):
    """Preserve the user's confirmed NightReID -> FeatEnHancer order."""
    import yaml
    package=ROOT/'mldc_featenhancer_learnable_cnl'
    config=yaml.safe_load((package/'configs/night600.yml').read_text())
    output=Path(config['OUTPUT_DIR'])
    if output.exists():return dict(phase='existing_output_preserved',output=str(output))
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=[s.decode(errors='replace') for s in path.read_bytes().split(b'\0') if s]
            if any(s=='mldc_featenhancer_learnable_cnl.run_when_free' or s.endswith('/mldc_featenhancer_learnable_cnl/train.py') for s in args):
                return dict(phase='already_running_or_queued',pid=int(path.parent.name))
        except OSError:pass
    with (package/'queue_restart_20261024.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u','-m','mldc_featenhancer_learnable_cnl.run_when_free'],
            cwd=str(ROOT),env=environment,stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    return dict(phase='queued_after_nightreid',worker_pid=process.pid,output=str(output))


def main():
    plan=json.loads((HERE/'terminal_plan_20261024.json').read_text())
    output=Path(plan['output']);console=Path(plan['console'])
    with (HERE/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not output.exists(),'Experiment output already exists; inspect before rerunning'
        environment=os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        console.parent.mkdir(parents=True,exist_ok=True)
        with console.open('x') as log:
            log.write('NightReID | learned MLDC + CNL | aligned learning-rate schedule\n')
            log.write('120 epochs | batch 64 | seed 1234 | live output\n\n');log.flush()
            process=subprocess.Popen([sys.executable,'-u',str(HERE/'train.py'),
                '--config_file',str(HERE/'configs/nightreid.yml')],cwd=str(ROOT),env=environment,
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        launch=dict(pid=process.pid,monitor_pid=os.getpid(),output=str(output),console=str(console),
            config=str(HERE/'configs/nightreid.yml'),epochs=120,batch=64,seed=1234,
            training_started=True,live_terminal=True)
        (HERE/'launch.json').write_text(json.dumps(launch,indent=2))
        status('training_starting',**launch)
        try:
            while process.poll() is None:
                train=row(output,'training.csv');evaluation=row(output,'evaluation.csv')
                status('training_running' if train else 'training_starting',
                    latest_training=train,latest_evaluation=evaluation)
                time.sleep(10)
            train=row(output,'training.csv');final=row(output,'evaluation.csv')
            assert process.returncode==0,'Training exited with code %d; see console log'%process.returncode
            assert train and final and int(train['epoch'])==120 and int(final['epoch'])==120
            assert (output/'transformer_120.pth').exists()
            best=json.loads((output/'best_metrics.json').read_text())
            status('completed',training_completed=True,final_metrics=final,best_metrics=best,exit_code=0)
            with console.open('a') as log:
                log.write('\nNightReID training completed: epoch 120, mAP %.4f%%, Rank-1 %.4f%%\n'%
                    (100*float(final['mAP']),100*float(final['rank1'])))
                log.write('Results: '+str(output)+'\n')
            if plan.get('start_previously_queued_featenhancer_afterwards'):
                next_run=start_next_if_pending(environment)
                status('completed',next_experiment=next_run)
        except BaseException as error:
            status('failed',error=str(error),exit_code=process.poll())
            with console.open('a') as log:log.write('\nExperiment monitor: '+str(error)+'\n')
            raise


if __name__=='__main__':main()
