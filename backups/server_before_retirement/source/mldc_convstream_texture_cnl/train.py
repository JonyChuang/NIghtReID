"""Default entry for the original MLDC+CNL baseline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mldc_convstream_texture_cnl.train_batchformer import main

if __name__ == '__main__':
    main(default_config='mldc_cnl_baseline_nightreid.yml')
