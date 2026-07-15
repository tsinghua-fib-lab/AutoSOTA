from typing import Dict
from appfl.algorithm.trainer import VanillaTrainer


class ViTFineTuningTrainer(VanillaTrainer):
    def get_parameters(self) -> Dict:
        return {k: v.cpu() for k, v in self.model.heads.state_dict().items()}

    def load_parameters(self, params: Dict):
        self.model.heads.load_state_dict(params, strict=False)
