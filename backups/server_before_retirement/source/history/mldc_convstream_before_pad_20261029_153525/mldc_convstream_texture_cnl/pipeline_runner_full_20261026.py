"""Durable fixed-budget teacher120 -> CUDA preflight -> student120 sequence."""
import datetime
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/root/log/transreid/nightreid/mldc_convstream_texture_cnl/pipeline_texture_xkd_teacher120_seed1234_20261026')
PYTHON='/root/miniconda3/envs/QREID/bin/python'


def now(): return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def status(stage,**kwargs):
    path=RUN/'status.json'; old=json.loads(path.read_text()) if path.exists() else {}
    old.update(stage=stage,updated_at=now(),**kwargs)
    temp=RUN/'status.tmp'; temp.write_text(json.dumps(old,indent=2),encoding='utf-8'); temp.replace(path)
    print(json.dumps(old),flush=True)


def execute(command,stage,logname):
    with (RUN/logname).open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=str(ROOT),env=os.environ.copy(),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT)
        status(stage,child_pid=process.pid,command=command,console=str(RUN/logname))
        code=process.wait()
    if code: raise RuntimeError(stage+' exited with code '+str(code)+'; see '+str(RUN/logname))


def main():
    status('starting',pipeline_pid=os.getpid(),seed=1234,teacher_epochs=120,student_epochs=120,
        started_at=now(),student_objective='final-feature identity-aware cross-model InfoNCE',
        teacher_output='/root/log/transreid/nightreid/mldc_convstream_texture_cnl/teachers/texture_full120_seed1234',
        student_output='/root/log/transreid/nightreid/mldc_convstream_texture_cnl/students/texture_xkd_teacher120_seed1234_matched120')
    execute([PYTHON,'-u','-m','mldc_convstream_texture_cnl.train_teacher','--config_file',
        'mldc_convstream_texture_cnl/configs/texture_teacher120_nightreid.yml'],'teacher_training','teacher.console.log')
    status('teacher_complete',teacher_completed_at=now())
    execute([PYTHON,'-u','-m','mldc_convstream_texture_cnl.verify_xkd_gpu'],
        'student_gpu_preflight','student_preflight.console.log')
    execute([PYTHON,'-u','-m','mldc_convstream_texture_cnl.train','--config_file',
        'mldc_convstream_texture_cnl/configs/texture_xkd_nightreid.yml'],'student_training','student.console.log')
    status('complete',completed_at=now(),child_pid=None)


if __name__=='__main__':
    try: main()
    except BaseException as error:
        status('failed',error=repr(error),child_pid=None)
        raise
