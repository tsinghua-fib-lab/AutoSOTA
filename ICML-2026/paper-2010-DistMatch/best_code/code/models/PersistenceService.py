from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, List, Dict, Union, Tuple
import json
import dataclasses
from datetime import datetime
import time

FILE_NAME = "model_register"


class PersistenceModel:

    @abstractmethod
    def save(self, path: Path, model_name: str):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path):
        pass


@dataclasses.dataclass
class RegisterEntry:
    alpha: Optional[List[str]]
    ts_id: Optional[List[str]]
    file_names: List[str]
    class_name: str


class ModelPersistenceService:
    def __init__(self, model_dict: dict, base_directory: str, all_symbol: str = "0", save_new_reg_bak=False) -> None:
        super().__init__()
        self._model_dict = model_dict
        self._base_dir = base_directory
        self._all_symbol = all_symbol
        self._save_new_reg_bak = save_new_reg_bak
        self._register: Dict[str, List[RegisterEntry]] = self._load_model_register()

    def model_existing(self, fingerprint: str):
        return fingerprint in self._register

    def model_supported(self, model_class_name: str):
        return model_class_name in self._model_dict

    def load_models(self, fingerprint: str, init_model_func=None) -> Optional:
        if not self.model_existing(fingerprint):
            return None, None
        access_dict = defaultdict(dict)
        model_list = []
        for entry in self._register[fingerprint]:
            models = self._load_models(class_name=entry.class_name, file_names=entry.file_names,
                                       init_model_func=init_model_func)
            model_list.extend(models)
            if len(models) < 2:
                models = models[0]
            alpha = entry.alpha if entry.alpha is not None else self._all_symbol
            ts_id = entry.ts_id if entry.ts_id is not None else self._all_symbol
            access_dict[ts_id][alpha] = models
        return access_dict, model_list

    def save_models(self, access_dict, model_class_name, fingerprint: str,
                    own_per_alpha, own_per_data, only_state=False):
        base_name = f"{model_class_name}_{datetime.now().strftime('%d%m%Y_%H%M%S')}"
        entries = self._generate_entries(access_dict, own_per_alpha=own_per_alpha, own_per_data=own_per_data,
                                         base_name=base_name)
        for entry, models in entries:
            for idx, file_name in enumerate(entry.file_names):
                if only_state:
                    models[idx].save_state(path=Path(self._base_dir), model_name=f"{file_name}")
                else:
                    models[idx].save(path=Path(self._base_dir), model_name=f"{file_name}")
        self._register[fingerprint] = [entry for entry, _ in entries]
        self._update_register(base_name)

    def _generate_entries(self, access_dict, own_per_alpha, own_per_data, base_name) -> List[Tuple[RegisterEntry, List[PersistenceModel]]]:
        entries = []
        for ts_id, _dict in access_dict.items() if own_per_data else [(None, access_dict[self._all_symbol])]:
            for alpha, models in _dict.items() if own_per_alpha else [(None, _dict[self._all_symbol])]:
                if not isinstance(models, List):
                    models = [models]
                file_names = [f"{base_name}_{ts_id if ts_id else ''}_{alpha if alpha else ''}_{idx}.p"
                              for idx, _ in enumerate(models)]
                entries.append((RegisterEntry(alpha=alpha, ts_id=ts_id, class_name=models[0].__class__.__name__,
                                              file_names=file_names), models))
        return entries

    def _load_models(self, class_name: str, file_names: List[str], init_model_func=None) -> List[PersistenceModel]:
        if init_model_func is not None:
            # Only Load the state dict
            models = []
            for file in file_names:
                new_model = init_model_func()
                new_model.load_state(f"{self._base_dir}/{file}")
                models.append(new_model)
            return models
        else:
            # Init model completely
            return [self._model_dict[class_name].load(f"{self._base_dir}/{file}") for file in file_names]

    def _load_model_register(self) -> Dict[str, List[RegisterEntry]]:
        register_raw: Dict[str, Any] = json.load(open(f"{self._base_dir}/{FILE_NAME}.json", mode="r"))
        register = dict()
        for fingerprint, entrylist in register_raw.items():
            register[fingerprint] = [RegisterEntry(**entry) for entry in entrylist]
        return register

    def _update_register(self, updated_name):
        json.dump(self._register, open(f"{self._base_dir}/{FILE_NAME}.json", mode="w"), cls=EnhancedJSONEncoder)
        if self._save_new_reg_bak:
            json.dump(self._register, open(f"{self._base_dir}/{FILE_NAME}_bak_{updated_name}_{round(time.time() * 1000)}.json", mode="w"), cls=EnhancedJSONEncoder)

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)
