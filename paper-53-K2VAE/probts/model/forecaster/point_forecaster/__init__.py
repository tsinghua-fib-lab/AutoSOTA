from .mean import MeanForecaster
from .linear import LinearForecaster
from .patchtst import PatchTST
from .transformer import TransformerForecaster
from .gru import GRUForecaster
from .dlinear import DLinear
from .nlinear import NLinear
from .nhits import NHiTS
from .timesnet import TimesNet
from .itransformer import iTransformer
from .autoformer import Autoformer
from .tsmixer import TSMixer
from .elastst import ElasTST

# ------- add timesfm to sys.path ----------
try:
    import os, sys
    current_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
    timesfm_path = os.path.join(project_root, 'submodules', 'timesfm', 'src')

    if timesfm_path not in sys.path:
        sys.path.append(timesfm_path)
except Exception as e:
    print(f"Warning: Unable to add timesfm to sys.path. {e}")
# ------------------------------------------

import importlib

modules = [
    ('timesfm', 'TimesFM'),
    ('timer', 'Timer'),
    ('units', 'UniTS'),
    ('forecastpfn', 'ForecastPFN'),
    ('tinytimemixer', 'TinyTimeMixer')
]

for module, class_name in modules:
    try:
        mod = importlib.import_module(f".{module}", package=__package__)
        globals()[class_name] = getattr(mod, class_name)
    except ImportError:
        # print(f"Warning: {class_name} is not available due to missing dependencies.")
        pass