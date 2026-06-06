from mmseg.registry import HOOKS
from mmengine.hooks import Hook

@HOOKS.register_module()
class SaveVFMHook(Hook):

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        for k in list(checkpoint["state_dict"].keys()):
            if "VFM" in k:
                checkpoint["state_dict"].pop(k)