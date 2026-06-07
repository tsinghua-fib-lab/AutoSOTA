from torch.optim import Adam


class BaseTask:
    def __init__(self, args, client_id, data, data_dir, device):
        self.client_id = client_id
        self.data_dir = data_dir
        self.args = args
        self.device = device

        if data is not None:
            self.data = data
            if hasattr(self.data, "_data_list"):
                self.data._data_list = None
            self.data = self.data.to(device)
            self.load_train_val_test_split()
            self.model = self.default_model.to(device)
            self.optim = Adam(self.model.parameters(
            ), lr=self.args.lr, weight_decay=self.args.weight_decay)

        self.override_evaluate = None
        self.step_preprocess = None

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    @property
    def num_samples(self):
        raise NotImplementedError

    @property
    def default_model(self):
        raise NotImplementedError

    @property
    def default_optim(self):
        raise NotImplementedError

    @property
    def default_loss_fn(self):
        raise NotImplementedError

    @property
    def train_val_test_path(self):
        raise NotImplementedError

    @property
    def default_train_val_test_split(self):
        raise NotImplementedError

    def load_train_val_test_split(self):
        raise NotImplementedError

    def load_custom_model(self, custom_model):
        self.model = custom_model.to(self.device)
        self.optim = self.optim = Adam(self.model.parameters(
        ), lr=self.args.lr, weight_decay=self.args.weight_decay)
