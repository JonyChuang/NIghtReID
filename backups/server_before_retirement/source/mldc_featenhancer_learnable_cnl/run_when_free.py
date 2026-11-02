"""One-shot queue: wait for GPU 0, verify capacity, then start this experiment."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PYTHON=sys.executable
NAME=HERE.name
STATUS=HERE/'queue_status.json'


def status(phase,**values):
    value=dict(phase=phase,worker_pid=os.getpid(),package=str(HERE),
               time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**values)
    pending=STATUS.with_suffix('.pending')
    pending.write_text(json.dumps(value,indent=2));pending.replace(STATUS)
    return value


def gpu_state():
    command=['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits']
    free=int(subprocess.check_output(command,text=True).strip().splitlines()[0])
    raw=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid',
                                 '--format=csv,noheader,nounits'],text=True)
    pids=[int(line.strip()) for line in raw.splitlines() if line.strip().isdigit()]
    return dict(free_mib=free,compute_pids=pids)


def main():
    cpu=json.loads((HERE/'verification_cpu.json').read_text())
    assert cpu['all_passed']
    import yaml
    cfg=yaml.safe_load((HERE/'configs/night600.yml').read_text())
    output=Path(cfg['OUTPUT_DIR'])
    assert not output.exists(),'Experiment output already exists'
    environment=os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    started=time.monotonic()
    while True:
        state=gpu_state()
        free_disk=shutil.disk_usage(ROOT).free
        if not state['compute_pids'] and state['free_mib']>=14*1024 and free_disk>=2*1024**3:
            break
        status('waiting_for_gpu',gpu=state,disk_free_bytes=free_disk,
               output=str(output),training_started=False)
        if time.monotonic()-started>24*3600:
            raise RuntimeError('GPU queue timed out after 24 hours; training was not started')
        time.sleep(30)
    status('checking_gpu_capacity',gpu=state,output=str(output),training_started=False)
    capacity_report=HERE/'verification_gpu.json'
    with (HERE/'capacity_check.log').open('w') as log:
        result=subprocess.run([PYTHON,'-u','-m',NAME+'.verify','--cuda-batch','64',
                               '--report',str(capacity_report)],cwd=str(ROOT),env=environment,
                              stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0,'GPU verification failed; see capacity_check.log'
    capacity=json.loads(capacity_report.read_text())
    assert capacity['all_passed'] and capacity['capacity_with_margin'],capacity
    # The capacity process has exited and released its allocations.
    state=gpu_state()
    assert not state['compute_pids'],'Another GPU task started during capacity validation; inspect before launching'
    assert not output.exists(),'Experiment output appeared while queued'
    output.parent.mkdir(parents=True,exist_ok=True)
    console=output.with_suffix('.console.log')
    with console.open('x') as log:
        process=subprocess.Popen([PYTHON,'-u',str(HERE/'train.py')],cwd=str(ROOT),env=environment,
                                 stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                 start_new_session=True)
    launch=dict(pid=process.pid,output=str(output),console=str(console),
                config=str(HERE/'configs/night600.yml'),training_started=True,
                epochs=120,batch=64,seed=1234)
    (HERE/'launch.json').write_text(json.dumps(launch,indent=2))
    status('training_starting',**launch)
    for _ in range(300):
        if process.poll() is not None:
            raise RuntimeError('Training exited during startup: '+console.read_text(errors='replace')[-3500:])
        if (output/'startup_gradients.json').exists() or (output/'training.csv').exists():
            status('training_running',**launch,successful_optimizer_update_observed=True)
            return
        time.sleep(2)
    status('training_starting',**launch,successful_optimizer_update_observed=False)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        previous=json.loads(STATUS.read_text()) if STATUS.exists() else {}
        status('failed',error=str(error),training_started=previous.get('training_started',False),
               training_pid=previous.get('pid'),output=previous.get('output'))
        raise
