from pathlib import Path
COMP_NAME = "g2net-gravitational-wave-detection"

OUTPUT_PATH = Path('/home/isamu/workdata/kaggle/G2net/test_output/models')
INPUT_PATH = Path('/home/isamu/workdata/kaggle/G2net/input/g2net-gravitational-wave-detection/') # tfrecord data
DATA_PATH = Path('/home/isamu/workdata/kaggle/G2net/final/dataset/whiten/') # data used for spectrul whitening
MODEL_PATH =Path('/home/isamu/workdata/kaggle/G2net/final/dataset/models/') # MODEL weights for inference
#MODEL_PATH =  Path('/home/isamu/workdata/kaggle/G2net/test_output/models')
CONFIG_PATH=Path(f"/home/isamu/workdata/kaggle/G2net/final/G2Net-GoGoGo-isamu/isamu/hyperparams.yml")
SR = 2048