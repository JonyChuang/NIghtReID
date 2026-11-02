"""Run NightReID once the specified TransReID+CNL run completes and GPU 0 is free."""
import csv
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
STATUS=HERE/'queue_status.json'


def process_matches(pid,start_ticks):
    try:
        fields=Path('/proc',str(pid),'stat').read_text().split(') ',1)[1].split()
        return fields[19]==str(start_ticks) and fields[0]!='Z'
    except FileNotFoundError:return False


def status(phase,**values):
    previous=json.loads(STATUS.read_text()) if STATUS.exists() else {}
    previous.update(phase=phase,worker_pid=os.getpid(),time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**values)
    pending=STATUS.with_suffix('.pending')
    pending.write_text(json.dumps(previous,indent=2));pending.replace(STATUS)


def last_row(directory,name):
    path=Path(directory)/name
    if not path.exists():return None
    with path.open() as stream:rows=list(csv.DictReader(stream))
    return rows[-1] if rows else None


def completed(directory,epochs):
    train=last_row(directory,'training.csv');evaluation=last_row(directory,'evaluation.csv')
    return bool(train and evaluation and int(train['epoch'])==epochs and int(evaluation['epoch'])==epochs
                and (Path(directory)/('transformer_%d.pth'%epochs)).exists())


def gpu_state():
    free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True).strip().splitlines()[0])
    text=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid',
        '--format=csv,noheader,nounits'],text=True)
    return dict(free_mib=free,compute_pids=[int(v.strip()) for v in text.splitlines() if v.strip().isdigit()])


def release_older_queue(plan):
    older=plan.get('deferred_queue')
    if older and process_matches(older['pid'],older['start_ticks']):
        os.kill(older['pid'],signal.SIGCONT)
        status(json.loads(STATUS.read_text())['phase'],older_queue_resumed=True)


def main():
    plan=json.loads((HERE/'queue_plan.json').read_text())
    assert json.loads((HERE/'verification_cpu.json').read_text())['all_passed']
    # Held for the full experiment so this package cannot launch duplicate jobs.
    with (HERE/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            predecessor=plan['predecessor'];output=Path(plan['output'])
            assert not output.exists(),'NightReID output already exists'
            status('waiting_for_predecessor',training_started=False,output=str(output),
                   predecessor=predecessor,deferred_queue=plan.get('deferred_queue'))
            while process_matches(predecessor['pid'],predecessor['start_ticks']):
                status('waiting_for_predecessor',predecessor_training=last_row(predecessor['output'],'training.csv'))
                time.sleep(30)
            assert completed(predecessor['output'],predecessor['epochs']), 'TransReID+CNL did not complete its final epoch, evaluation and checkpoint'
            status('waiting_for_gpu',predecessor_completed=True)
            while True:
                gpu=gpu_state()
                status('waiting_for_gpu',gpu=gpu)
                if not gpu['compute_pids'] and gpu['free_mib']>=14*1024:break
                time.sleep(30)
            environment=os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            status('checking_gpu_capacity',training_started=False)
            report=HERE/'verification_gpu.json'
            with (HERE/'capacity_check.log').open('w') as log:
                check=subprocess.run([sys.executable,'-u','-m',HERE.name+'.verify','--cuda-batch','64',
                    '--report',str(report)],cwd=str(ROOT),env=environment,stdout=log,stderr=subprocess.STDOUT)
            assert check.returncode==0,'GPU capacity check failed; see capacity_check.log'
            assert json.loads(report.read_text())['capacity_with_margin']
            # A second user job may have started during verification; wait for it.
            while True:
                gpu=gpu_state()
                if not gpu['compute_pids'] and gpu['free_mib']>=14*1024:break
                status('waiting_for_gpu',gpu=gpu)
                time.sleep(30)
            assert not output.exists(),'NightReID output appeared while queued'
            output.parent.mkdir(parents=True,exist_ok=True)
            console=output.with_suffix('.console.log')
            with console.open('x') as log:
                process=subprocess.Popen([sys.executable,'-u',str(HERE/'train.py'),
                    '--config_file',str(HERE/'configs/nightreid.yml')],cwd=str(ROOT),env=environment,
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            launch=dict(pid=process.pid,output=str(output),console=str(console),
                config=str(HERE/'configs/nightreid.yml'),training_started=True,epochs=120,batch=64,seed=1234)
            (HERE/'launch.json').write_text(json.dumps(launch,indent=2))
            status('training_starting',**launch)
            while process.poll() is None:
                training=last_row(output,'training.csv');evaluation=last_row(output,'evaluation.csv')
                status('training_running' if training else 'training_starting',
                    latest_training=training,latest_evaluation=evaluation)
                time.sleep(30)
            assert process.returncode==0 and completed(output,120),'NightReID training did not finish successfully; see console log'
            final=last_row(output,'evaluation.csv')
            best=json.loads((output/'best_metrics.json').read_text())
            status('completed',final_metrics=final,best_metrics=best,training_completed=True)
        except BaseException as error:
            status('failed',error=str(error))
            raise
        finally:release_older_queue(plan)


if __name__=='__main__':
    def terminate(signum,frame):raise SystemExit('Queue terminated by signal %d'%signum)
    signal.signal(signal.SIGTERM,terminate)
    main()
