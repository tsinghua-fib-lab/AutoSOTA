from .gru_nvp import GRU_NVP
from .gru_maf import GRU_MAF
from .timegrad import TimeGrad
from .trans_maf import Trans_MAF
from .csdi import CSDI
from .tsdiff import TSDiffCond

# ------- add lag_llama to sys.path ---------
try:
    import os, sys
    current_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
    lag_llama_path = os.path.join(project_root, 'submodules', 'lag_llama')
    moirai_path = os.path.join(project_root, 'submodules', 'uni2ts', 'src')

    if lag_llama_path not in sys.path:
        sys.path.append(lag_llama_path)

    if moirai_path not in sys.path:
        sys.path.append(moirai_path)

except Exception as e:
    print(f"Warning: Unable to add lag_llama to sys.path. {e}")
# -------------------------------------------

import importlib

modules = [
    ('moirai', 'Moirai'),
    ('chronos', 'Chronos'),
    ('lag_llama', 'LagLlama'),
]

for module, class_name in modules:
    try:
        mod = importlib.import_module(f".{module}", package=__package__)
        globals()[class_name] = getattr(mod, class_name)
    except ImportError:
        # print(f"Warning: {class_name} is not available due to missing dependencies.")
        pass