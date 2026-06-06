from utils import *
from models import TimeXer, SOFTS, CycleNet, IFT, iTransformer, TimeMixer, FreTS, SegRNN, PatchTST, Crossformer, TimesNet, DLinear, FEDformer, Autoformer, Informer


MODEL = {
    'TimeXer': TimeXer,
    'SOFTS': SOFTS,
    'CycleNet': CycleNet,
    'IFT': IFT,
    'iTransformer': iTransformer,
    'TimeMixer': TimeMixer,
    'FreTS': FreTS,
    'SegRNN': SegRNN,
    'PatchTST': PatchTST,
    'Crossformer': Crossformer,
    'TimesNet': TimesNet,
    'DLinear': DLinear,
    'FEDformer': FEDformer,
    'Autoformer': Autoformer,
    'Informer': Informer
}


LOSS = {
    'MSE': MSELoss,
    'MAE': MAELoss
}


class Main:
    def __init__(self, CFG):
        self.CFG = CFG
        self.model = MODEL[self.CFG.model].Model(self.CFG)

        self.criterion = LOSS[self.CFG.loss](self.CFG)
        self.optimizer = Adam(self.model.parameters(), lr=self.CFG.lr)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion.to(self.device)

        self.metric = Metric()
        self.factory = TSFactory(self.CFG)
        self.lr_scheduler = LRScheduler(self.CFG)
        self.attn_visualizer = AttnVisualizer(self.CFG)
        self.test_visualizer = TestVisualizer(self.CFG)

    def __str__(self):
        return self.CFG.description

    def __save__(self):
        path = os.path.join(self.CFG.root_path, self.CFG.file_path, self.CFG.model)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), f"{path}/{mark(self.CFG)}.pt")

    def __load__(self):
        path = os.path.join(self.CFG.root_path, self.CFG.file_path, self.CFG.model)
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.load_state_dict(torch.load(f"{path}/{mark(self.CFG)}.pt", map_location=self.device))

    def forecast(self, x, x_mark, y, y_mark, dataset=None):
        x_y = torch.cat((y[:, :self.CFG.label_len, :], torch.zeros_like(y[:, -self.CFG.pred_len:, :])), dim=1)
        pred, true = self.model(x, x_mark, x_y, y_mark), y[:, -self.CFG.pred_len:, :]
        rest = pred[:, self.CFG.pred_len:, :]
        pred = pred[:, :self.CFG.pred_len, :]
        loss = self.criterion(pred, true)
        pred, true = pred.detach().cpu().numpy(), true.detach().cpu().numpy()
        if dataset and dataset.scale and self.CFG.inverse_transform:
            shape = pred.shape
            pred = dataset.inverse_transform(pred.reshape(shape[0] * shape[1], -1)).reshape(shape)
            true = dataset.inverse_transform(true.reshape(shape[0] * shape[1], -1)).reshape(shape)
        i = -1 if self.CFG.mode == 'MS' else 0
        pred, true = pred[:, :, i:], true[:, :, i:]
        return pred, true, loss

    def train(self):
        self.model.train()
        _, train_loader = self.factory('train')

        min_loss = math.inf
        patience = self.CFG.patience
        for epoch in range(self.CFG.epochs):
            epoch_loss = []
            epoch_metric1, epoch_metric2 = [], []
            progress_bar = ProgressBar(self.CFG, epoch, len(train_loader))
            with progress_bar.progress:
                for x, x_mark, y, y_mark in train_loader:
                    x, x_mark, y, y_mark = x.to(self.device), x_mark.to(self.device), y.to(self.device), y_mark.to(self.device)
                    if ('PEMS' in self.CFG.data_path or 'Solar' in self.CFG.data_path) and self.CFG.model != 'CycleNet':
                        x_mark, y_mark = None, None
                    self.optimizer.zero_grad()
                    pred, true, loss = self.forecast(x, x_mark, y, y_mark)
                    loss.backward()
                    self.optimizer.step()
                    memory = torch.cuda.max_memory_allocated()
                    epoch_loss.append(loss.item())
                    epoch_metric = self.metric(pred, true)
                    epoch_metric1.append(epoch_metric[self.CFG.metric1])
                    epoch_metric2.append(epoch_metric[self.CFG.metric2])
                    progress_bar(memory, epoch_loss, epoch_metric1, epoch_metric2)
            epoch_loss = np.mean(epoch_loss)
            epoch_metric1 = np.mean(epoch_metric1)
            epoch_metric2 = np.mean(epoch_metric2)
            print(f"Epoch: {epoch + 1:>3}  -  Loss: {epoch_loss:>6.3f},  {self.CFG.metric1}: {epoch_metric1:>6.3f},  {self.CFG.metric2}: {epoch_metric2:>6.3f}")

            vali_loss = self.vali()
            self.model.train()
            if vali_loss < min_loss:
                min_loss = vali_loss
                self.__save__()
            else:
                patience -= 1
            if not patience:
                break
            self.lr_scheduler(self.optimizer, epoch + 1)
        self.__load__()

    def vali(self):
        self.model.eval()
        _, vali_loader = self.factory('vali')

        with torch.no_grad():
            vali_loss = []
            vali_metric1, vali_metric2 = [], []
            for x, x_mark, y, y_mark in vali_loader:
                x, x_mark, y, y_mark = x.to(self.device), x_mark.to(self.device), y.to(self.device), y_mark.to(self.device)
                if ('PEMS' in self.CFG.data_path or 'Solar' in self.CFG.data_path) and self.CFG.model != 'CycleNet':
                    x_mark, y_mark = None, None
                pred, true, loss = self.forecast(x, x_mark, y, y_mark)
                vali_loss.append(loss.item())
                vali_metric = self.metric(pred, true)
                vali_metric1.append(vali_metric[self.CFG.metric1])
                vali_metric2.append(vali_metric[self.CFG.metric2])
            vali_loss = np.mean(vali_loss)
            vali_metric1 = np.mean(vali_metric1)
            vali_metric2 = np.mean(vali_metric2)
            print(f"Validation  -  Loss: {vali_loss:>6.3f},  {self.CFG.metric1}: {vali_metric1:>6.3f},  {self.CFG.metric2}: {vali_metric2:>6.3f}")
            return vali_loss

    def test(self):
        n_mc = 10  # MC Dropout samples
        self.model.train()  # Enable dropout for MC Dropout ensemble
        if self.CFG.phase == 1:
            self.__load__()
        test_set, test_loader = self.factory('test')

        with torch.no_grad():
            test_loss = []
            test_metric1, test_metric2 = [], []
            plot = 0
            plots = set([int(i * len(test_loader) / self.CFG.num_plots) for i in range(self.CFG.num_plots)]) if len(test_loader) >= self.CFG.num_plots else set(range(len(test_loader)))
            for i, (x, x_mark, y, y_mark) in enumerate(test_loader):
                x, x_mark, y, y_mark = x.to(self.device), x_mark.to(self.device), y.to(self.device), y_mark.to(self.device)
                if ('PEMS' in self.CFG.data_path or 'Solar' in self.CFG.data_path) and self.CFG.model != 'CycleNet':
                    x_mark, y_mark = None, None
                mc_preds = []
                for _ in range(n_mc):
                    pred_mc, true, loss = self.forecast(x, x_mark, y, y_mark, dataset=test_set)
                    mc_preds.append(pred_mc)
                pred = np.mean(mc_preds, axis=0)
                test_loss.append(loss.item())
                test_metric = self.metric(pred, true)
                test_metric1.append(test_metric[self.CFG.metric1])
                test_metric2.append(test_metric[self.CFG.metric2])
                if self.CFG.test_visualization and i in plots:
                    x = x.detach().cpu().numpy()
                    if test_set.scale and self.CFG.inverse_transform:
                        shape = x.shape
                        x = test_set.inverse_transform(x.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    pred = np.concatenate((x[0, :, -1], pred[0, :, -1]))
                    true = np.concatenate((x[0, :, -1], true[0, :, -1]))
                    self.test_visualizer(pred, true, plot)
                    plot += 1
            test_loss = np.mean(test_loss)
            test_metric1 = np.mean(test_metric1)
            test_metric2 = np.mean(test_metric2)
            print(f"Test Score  -  Loss: {test_loss:>6.3f},  {self.CFG.metric1}: {test_metric1:>6.3f},  {self.CFG.metric2}: {test_metric2:>6.3f}")
            update_result(self.CFG, result=f"{self.CFG.metric1}: {test_metric1:>6.3f},  {self.CFG.metric2}: {test_metric2:>6.3f}")
