import aim
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only

import sys
sys.path.extend(['./', '../', '../../'])
from models.FlowMatching.TSFlow.seml_util.media_logger import MediaLogger


class SemlLogger(Logger):
    def __init__(self, experiment, logdir, media_logger: MediaLogger):
        super().__init__()
        self.experiment = experiment
        self.logdir = logdir
        self.media_logger = media_logger

    @property
    def name(self):
        return "SemlLogger"

    @property
    def version(self):
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        for key, val in metrics.items():
            if isinstance(val, aim.Image):
                self.media_logger.save_image(f"{key}_{step}", val)
            else:
                if key not in self.experiment.current_run.info:
                    self.experiment.current_run.info[key] = []
                self.experiment.current_run.info[key].append(val)
