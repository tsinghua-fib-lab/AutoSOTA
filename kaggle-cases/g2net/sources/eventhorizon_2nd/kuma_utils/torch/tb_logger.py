from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir=log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scaler(tag, value, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        for tag, value in tag_value_pairs:
            self.writer.add_scaler(tag, value, step)


class DummyTensorBoardLogger:
    def __init__(self, log_dir=''):
        pass

    def scalar_summary(self, tag, value, step):
        pass

    def list_of_scalars_summary(self, tag_value_pairs, step):
        pass
