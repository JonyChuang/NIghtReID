"""Supervise one NightReID run, retaining a file log and existing console output."""
from pathlib import Path
import csv
import datetime
import json
import os
import subprocess
import sys

HERE=Path(__file__).resolve().parent
STATE=HERE/'nightreid_state.json'


def status(value,**extra):
    report=dict(status=value,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
    temporary=STATE.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2));temporary.replace(STATE)
    print(json.dumps(report),flush=True)


def main():
    descriptor=os.open(str(HERE/'nightreid_run.lock'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'w') as handle:handle.write(str(os.getpid()))
    settings=json.loads((HERE/'nightreid_launch_settings.json').read_text())
    output=Path(settings['output_dir'])
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    busy=subprocess.run(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                        capture_output=True,text=True,check=True).stdout.strip()
    if busy:raise RuntimeError('GPU is occupied; no concurrent run started')
    environment=os.environ.copy();environment['CUDA_VISIBLE_DEVICES']='0'
    command=[settings['python'],'-u','-m','mldc_nfc_batchformer.train_batchformer',
             '--config_file',str(HERE/'configs/neighbor_nightreid.yml')]
    console=HERE/'training_nightreid_console.log'
    with console.open('w') as log:
        child=subprocess.Popen(command,cwd=settings['project_root'],env=environment,
                               stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        status('training',pid=child.pid,dataset='nightreid',epochs=120,output=str(output))
        terminal=settings.get('terminal')
        if terminal and Path(terminal).exists():
            try:
                descriptor=os.open(terminal,os.O_WRONLY|os.O_NOCTTY)
                with os.fdopen(descriptor,'wb',buffering=0) as display:
                    display.write(b'\n=== NightReID cached-neighbor BatchFormer: live training log ===\n')
                    subprocess.Popen(['tail','--pid='+str(child.pid),'-n','30','-f',str(console)],
                                     stdin=subprocess.DEVNULL,stdout=display,stderr=display,start_new_session=True)
            except OSError as error:
                print('Console unavailable; file logging continues:',error,flush=True)
        code=child.wait()
    if code:status('failed_training',exit_code=code,output=str(output));return
    best=json.loads((output/'best_metrics.json').read_text())
    with (output/'evaluation.csv').open() as handle:last=list(csv.DictReader(handle))[-1]
    assert int(last['epoch'])==120,'Final evaluation is incomplete'
    status('completed',dataset='nightreid',output=str(output),best=best,final=last)


if __name__=='__main__':
    try:main()
    except FileExistsError:raise
    except Exception as error:
        status('failed_launcher',error=repr(error));raise
