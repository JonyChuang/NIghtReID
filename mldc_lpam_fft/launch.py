"""Launch an authorized experiment when a suitable GPU is already present."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',action='store_true')
    args=parser.parse_args()
    package=Path(__file__).resolve().parent
    project=package.parent
    sys.path.insert(0,str(project))
    import torch
    if not torch.cuda.is_available():
        print(json.dumps(dict(started=False,reason='No visible CUDA GPU',package=str(package))))
        return
    free,total=torch.cuda.mem_get_info(0)
    if free<12*1024**3:
        print(json.dumps(dict(started=False,reason='Less than 12 GiB GPU memory free for batch 64',free_gpu_gib=free/1024**3)))
        return
    # Check cgroup limits instead of host-wide available RAM.
    limit_path=Path('/sys/fs/cgroup/memory.max');current_path=Path('/sys/fs/cgroup/memory.current')
    if limit_path.exists() and limit_path.read_text().strip()!='max':
        limit=int(limit_path.read_text())
        if limit<8*1024**3:
            print(json.dumps(dict(started=False,reason='Container RAM limit below 8 GiB for full training/evaluation',limit_gib=limit/1024**3)))
            return
    from config import cfg as base_cfg
    from mldc_lpam_fft.model import add_config
    cfg=add_config(base_cfg.clone());cfg.merge_from_file(str(package/'variants/fixed_global_lpam_fft_concat/config.yml'))
    if args.baseline:
        cfg.MODEL.LPAM_ENABLED=False
        cfg.OUTPUT_DIR=str(Path(cfg.OUTPUT_DIR).parent/'deep4_fixed_equal_baseline_120')
    output=Path(cfg.OUTPUT_DIR)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Existing experiment output: '+str(output))
    launch_file=package/('baseline_launch.json' if args.baseline else 'launch.json')
    if launch_file.exists():
        previous=json.loads(launch_file.read_text())
        pid=previous.get('pid')
        if pid and Path(f'/proc/{pid}').exists():
            raise RuntimeError('Previously launched process is still present; no duplicate launch')
    output.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u',str(package/'train.py'),'--config_file',str(package/'variants/fixed_global_lpam_fft_concat/config.yml')]
    if args.baseline:
        command+=['MODEL.LPAM_ENABLED','False','OUTPUT_DIR',str(output)]
    env=os.environ.copy();env['PYTHONUNBUFFERED']='1';env['PYTHONDONTWRITEBYTECODE']='1'
    env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='4';env['MKL_NUM_THREADS']='4'
    console=output.with_suffix('.console.log')
    with console.open('xb') as stream:
        child=subprocess.Popen(command,cwd=str(project),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    report=dict(started=True,pid=child.pid,command=command,output=str(output),console=str(console),baseline=args.baseline)
    launch_file.write_text(json.dumps(report,indent=2))
    time.sleep(2)
    if child.poll() is not None:
        report.update(started=False,exit_code=child.returncode,reason='Training process exited during startup; inspect console')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
