from .base import CallbackTemplate
from pprint import pformat


class SaveEveryEpoch(CallbackTemplate):
    '''
    Save snapshot every epoch
    '''

    def __init__(self, patience=5, target='valid_metric', maximize=False, skip_epoch=0):
        super().__init__()
        
    def after_epoch(self, env):
        env.checkpoint = True
    

class EarlyStopping(CallbackTemplate):
    '''
    Early stops the training if validation loss doesn't improve after a given patience.
    patience: int       = 
    target: str         = 
    maximize: bool      = 
    skip_epoch: int     =
    '''

    def __init__(self, patience=5, target='valid_metric', maximize=False, skip_epoch=0):
        super().__init__()
        self.state = {
            'patience': patience,
            'target': target,
            'maximize': maximize,
            'skip_epoch': skip_epoch,
            'counter': 0,
            'best_score': None,
            'best_epoch': None
        }

    def after_epoch(self, env):
        score = env.state[self.state['target']]
        epoch = env.state['epoch'] # local epoch
        if epoch < self.state['skip_epoch'] or epoch == 0:
            self.state['best_score'] = score
            self.state['best_epoch'] = env.global_epoch
            env.checkpoint = True
            env.state['best_score'] = self.state['best_score']
            env.state['best_epoch'] = self.state['best_epoch']
        else:
            if (self.state['maximize'] and score > self.state['best_score']) or \
                    (not self.state['maximize'] and score < self.state['best_score']):
                self.state['best_score'] = score
                self.state['best_epoch'] = env.global_epoch
                self.state['counter'] = 0
                env.checkpoint = True
                env.state['best_score'] = self.state['best_score']
                env.state['best_epoch'] = self.state['best_epoch']
            else:
                self.state['counter'] += 1

            env.state['patience'] = self.state['counter']
            if self.state['counter'] >= self.state['patience']:
                env.stop_train = True

    def state_dict(self):
        return self.state

    def load_state_dict(self, checkpoint):
        self.state = checkpoint

    def __repr__(self):
        return f'EarlyStopping(\n{pformat(self.state, compact=True, indent=2)})'
