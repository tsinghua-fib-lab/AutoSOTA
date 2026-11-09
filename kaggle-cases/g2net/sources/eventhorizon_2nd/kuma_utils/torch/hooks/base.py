
class HookTemplate:
    '''
    Hook is called in each mini-batch during traing / inference 
    and after processed all mini-batches, 
    in order to define the training process and evaluate the results of each epoch.
    '''

    def __init__(self):
        pass

    def forward_train(self, trainer, inputs):
        # return loss, approx
        pass

    forward_valid = forward_train

    def forward_test(self, trainer, inputs, approx):
        # return approx
        pass

    def evaluate_batch(self, trainer, inputs):
        # return None
        pass

    def evaluate_epoch(self, trainer):
        # return metric_total, monitor_metrics_total
        pass
