"""Execute Saito's original public 2021 smoothing notebook after path adaptation."""
from pathlib import Path
import argparse
import os
import sys
import warnings
import numpy as np

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
sys.path.insert(0,str(root/'reproductions/saito_smoothing'))
# Original notebook predates NumPy 1.24; this alias has the identical dtype.
np.object=object
warnings.filterwarnings('ignore',category=FutureWarning)
p=root/'reproductions/saito_smoothing/area_prediction.py'
p.write_text(p.read_text().replace("Path('../input/google-smartphone-decimeter-challenge')",repr(root/'data/competition').replace('PosixPath','Path')))
src=(root/'reproductions/saito_smoothing/notebook_cells.py.txt').read_text().split('# Original cell 7')[0]
src=src.replace("INPUT_PATH = '../input/google-smartphone-decimeter-challenge'",f'INPUT_PATH = {str(root/"data/competition")!r}')
src=src.replace('processes = multiprocessing.cpu_count()',f'processes = {args.workers}')
src=src.replace('from tqdm.notebook import tqdm','from tqdm import tqdm')
out=root/'artifacts/saito_smoothing';out.mkdir(parents=True,exist_ok=True);os.chdir(out)
exec(compile(src,'saito_original_notebook_cell6','exec'),globals())
main()
