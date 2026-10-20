"""Run the full MLDC/control pair sequentially, then produce comparison plots."""
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    for variant in ('all_cls_random_initialization', 'last_raw_cls_control'):
        subprocess.run([sys.executable, 'mldc_cls_readout/train.py', '--config_file',
                        f'mldc_cls_readout/variants/{variant}/config.yml'], cwd=root, check=True)
    subprocess.run([sys.executable, 'mldc_cls_readout/compare.py'], cwd=root, check=True)


if __name__ == '__main__':
    main()
