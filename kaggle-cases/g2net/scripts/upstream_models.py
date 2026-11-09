"""Load contestants' actual implementations; do not replace their architectures."""
import copy
import importlib.util
import sys
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVENTHORIZON = ROOT / "sources/eventhorizon_2nd"
TEAM3 = ROOT / "sources/team_3rd/1D_Model"


def eventhorizon_config(name):
    if str(EVENTHORIZON) not in sys.path:
        sys.path.insert(0, str(EVENTHORIZON))
    import configs
    config = getattr(configs, name)
    if config.pseudo_labels is not None:
        raise ValueError("Pseudo-label configs require predictions from the same fold; first run their dependency.")
    return config


def make_eventhorizon(name="Seq03aug3", pretrained=True):
    config = eventhorizon_config(name)
    parameters = copy.deepcopy(config.model_params)
    if "pretrained" in parameters:
        parameters["pretrained"] = pretrained
    return config.model(**parameters), config


def make_team3(psd_path, width=16, survival=1.0, architecture='V2SD', spectral_dropout=0, stable_gem=False):
    module_path = TEAM3 / "src/models_1d.py"
    spec = importlib.util.spec_from_file_location("g2net_team3_models_1d", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if stable_gem:
        from team3_stable_training import patch_gem
        patch_gem(module)
    parameters = dict(n=width, use_raw_wave=True, avr_w0_path=str(psd_path), sdrop=spectral_dropout)
    if architecture == 'V2SD':
        return module.V2StochasticDepth(proba_final_layer=survival, **parameters)
    if architecture == 'V2S':
        return module.ModelIafossV2S(**parameters)
    raise ValueError(f'Unsupported source architecture {architecture}')


def model_for_args(args, pretrained=True):
    if args.family == "eventhorizon":
        return make_eventhorizon(args.config, pretrained)
    return make_team3(args.psd, args.width, args.survival), None


def apply_eventhorizon_transform(waves, transform):
    # Packed data is scaled by 1e20. Undo in float64 for the original transform,
    # which normalizes by ~1e-20 before filtering and converting to float32.
    values = []
    for wave in waves:
        value = transform(wave.numpy().astype("float64") / 1e20)
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(np.ascontiguousarray(value)).float()
        values.append(value)
    return torch.stack(values)
