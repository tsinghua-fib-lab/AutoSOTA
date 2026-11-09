from typing import Any


class CallbackTemplate:
    '''
    Callback is called before or after each epoch.
    '''

    def __init__(self):
        pass

    def before_epoch(self, env):
        pass

    def after_epoch(self, env):
        pass

    def save_snapshot(self, trainer, path):
        pass

    def load_snapshot(self, trainer, path, device):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, checkpoint):
        pass

    def __repr__(self):
        return self.__class__.__name__
