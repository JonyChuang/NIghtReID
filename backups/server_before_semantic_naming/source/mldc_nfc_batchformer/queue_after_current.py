"""Run the isolated experiment after the recorded predecessor completes.

No process is stopped or signaled. Only a successful full predecessor and an
unoccupied GPU allow the GPU check and the new training to start.
"""
from pathlib import Path
import csv
import datetime
import json
import os
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
STATE=HERE/'queue_state.json'


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_state(status,**extra):
    value=dict(status=status,updated_at=now(),**extra)
    temporary=STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temporary.replace(STATE)
    print(json.dumps(value),flush=True)


def alive(pid,start_ticks):
    try:
        value=Path('/proc')/str(pid)/'stat'
        fields=value.read_text().rsplit(')',1)[1].split()
        return fields[0]!='Z' and fields[19]==str(start_ticks)
    except (FileNotFoundError,ProcessLookupError):
        return False


def rows(path):
    with path.open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def predecessor_complete(settings):
    output=Path(settings['predecessor_output'])
    try:
        training=rows(output/'training.csv');evaluation=rows(output/'evaluation.csv')
        epoch=settings['predecessor_epochs']
        checkpoint=output/f'transformer_{epoch}.pth'
        return (int(training[-1]['epoch'])==epoch and int(evaluation[-1]['epoch'])==epoch
                and checkpoint.stat().st_size>0
                and checkpoint.stat().st_mtime>=settings['predecessor_started_unix'])
    except (FileNotFoundError,IndexError,ValueError,KeyError):
        return False


def gpu_busy():
    process=subprocess.run(['nvidia-smi','-i','0','--query-compute-apps=pid',
                            '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return bool(process.stdout.strip())


def compare(settings,output):
    predecessor=Path(settings['predecessor_output'])
    old=json.loads((predecessor/'best_metrics.json').read_text())
    new=json.loads((output/'best_metrics.json').read_text())
    old_final=rows(predecessor/'evaluation.csv')[-1]
    new_final=rows(output/'evaluation.csv')[-1]
    result=dict(predecessor_best=old,neighbor_best=new,
                best_map_delta_points=100*(new['mAP']-old['mAP']),
                rank1_at_best_map_delta_points=100*(new['rank1']-old['rank1']),
                predecessor_final=old_final,neighbor_final=new_final,
                final_map_delta_points=100*(float(new_final['mAP'])-float(old_final['mAP'])),
                final_rank1_delta_points=100*(float(new_final['rank1'])-float(old_final['rank1'])),
                evaluation='independent single-image retrieval; no neighbor lookup at test time')
    (output/'comparison_to_predecessor.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    settings=json.loads((HERE/'queue_settings.json').read_text())
    lock=HERE/'queue.lock'
    descriptor=os.open(str(lock),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'w') as stream:stream.write(str(os.getpid()))
    write_state('waiting_for_predecessor',predecessor_pid=settings['predecessor_pid'],
                predecessor_output=settings['predecessor_output'])
    while alive(settings['predecessor_pid'],settings['predecessor_start_ticks']):
        time.sleep(30)
    if not predecessor_complete(settings):
        write_state('blocked_predecessor_incomplete',reason='Predecessor exited without complete final evaluation/checkpoint')
        return
    if gpu_busy():
        write_state('waiting_for_gpu',reason='Predecessor completed; GPU has another compute process')
        while gpu_busy():time.sleep(30)
    python=settings['python'];root=settings['project_root']
    environment=os.environ.copy();environment['CUDA_VISIBLE_DEVICES']='0'
    write_state('gpu_check_running')
    with (HERE/'gpu_check.log').open('w') as log:
        check=subprocess.run([python,'-u','-m','mldc_nfc_batchformer.smoke_gpu'],cwd=root,
                             env=environment,stdout=log,stderr=subprocess.STDOUT)
    if check.returncode:
        write_state('failed_gpu_check',exit_code=check.returncode,log=str(HERE/'gpu_check.log'))
        return
    output=Path(settings['output_dir'])
    if output.exists() and any(output.iterdir()):
        write_state('blocked_output_not_empty',output=str(output));return
    write_state('training',output=str(output),epochs=120)
    command=[python,'-u','-m','mldc_nfc_batchformer.train_batchformer',
             '--config_file',str(HERE/'configs/neighbor_night600.yml')]
    with (HERE/'training_console.log').open('w') as log:
        child=subprocess.Popen(command,cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT)
        write_state('training',pid=child.pid,output=str(output),epochs=120)
        code=child.wait()
    if code:
        write_state('failed_training',exit_code=code,output=str(output));return
    result=compare(settings,output)
    write_state('completed',output=str(output),comparison=result)


if __name__=='__main__':
    try:main()
    except Exception as error:
        # A second launcher must never overwrite the first launcher's state.
        if isinstance(error,FileExistsError):raise
        write_state('failed_queue',error=repr(error))
        raise
