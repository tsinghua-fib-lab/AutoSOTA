from mmseg.registry import HOOKS
from mmengine.hooks import Hook

@HOOKS.register_module()
class SaveBackboneHook(Hook):

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if "VFM" in k:
                continue
            if "backbone" in k:
                new_key = k.replace('backbone.', '')
                new_state_dict[new_key] = v
        checkpoint["state_dict"] = new_state_dict