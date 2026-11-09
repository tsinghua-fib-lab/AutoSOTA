from .base import HookTemplate


class TrainHook(HookTemplate):

    def __init__(self, evaluate_in_batch=False):
        super().__init__()
        self.evaluate_in_batch = evaluate_in_batch

    def _evaluate(self, trainer, approx, target):
        if trainer.eval_metric is None:
            metric_score = None
        else:
            metric_score = trainer.eval_metric(approx, target)
        monitor_score = []
        for monitor_metric in trainer.monitor_metrics:
            monitor_score.append(
                monitor_metric(approx, target))
        return metric_score, monitor_score

    def forward_train(self, trainer, inputs):
        target = inputs[-1]
        approx = trainer.model(*inputs[:-1])
        loss = trainer.criterion(approx, target)
        return loss, approx.detach()

    forward_valid = forward_train

    def forward_test(self, trainer, inputs):
        approx = trainer.model(*inputs[:-1])
        return approx

    def evaluate_batch(self, trainer, inputs, approx):
        target = inputs[-1]
        storage = trainer.epoch_storage
        if self.evaluate_in_batch:
            # Add scores to storage
            metric_score, monitor_score = self._evaluate(trainer, approx, target)
            storage['batch_metric'].append(metric_score)
            storage['batch_monitor'].append(monitor_score)
        else:
            # Add prediction and target to storage
            storage['approx'].append(approx)
            storage['target'].append(target)

    def evaluate_epoch(self, trainer):
        storage = trainer.epoch_storage
        if self.evaluate_in_batch:
            # Calculate mean metrics from all batches
            metric_total = storage['batch_metric'].mean(0)
            monitor_total = storage['batch_monitor'].mean(0).tolist()

        else: 
            # Calculate scores
            metric_total, monitor_total = self._evaluate(
                trainer, storage['approx'], storage['target'])
        return metric_total, monitor_total


SimpleHook = TrainHook # Compatibility
