import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size

        # 测试时使用更大的batch size以加速评估
        # 测试不需要反向传播，可以使用更大的batch
        self.test_batch_size = getattr(args, 'test_batch_size', None)
        if self.test_batch_size is None:
            # 根据数据集自动设置test_batch_size (训练batch_size的4-8倍)
            # CIFAR-10/100等小图像: 4倍
            # ImageNet等大图像: 2倍
            small_image_datasets = ['Mnist', 'MNIST', 'Cifar10', 'Cifar100', 'kvasir', 'Kvasir', 'Digit5']
            if self.dataset in small_image_datasets:
                # self.test_batch_size = self.batch_size * 8
                self.test_batch_size = 1024
            else:
                # self.test_batch_size = self.batch_size * 2
                self.test_batch_size = 256

        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

        # 数据缓存策略：可配置的全局缓存开关
        # 默认启用缓存以避免重复文件 I/O，提升训练效率
        # 可通过 args.use_cache 参数禁用（适用于极大数据集或内存受限场景）
        self.use_cache = getattr(args, 'use_cache', True)

        if self.use_cache:
            # 初始化时加载并缓存数据（避免重复文件 I/O）
            self._train_data_cache = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
            self._test_data_cache = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        else:
            # 大数据集不缓存，每次按需加载
            self._train_data_cache = None
            self._test_data_cache = None


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        if self.use_cache:
            # 使用缓存数据（零文件 I/O）
            train_data = self._train_data_cache
        else:
            # 大数据集按需加载
            train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)

        return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.test_batch_size

        if self.use_cache:
            # 使用缓存数据（零文件 I/O）
            test_data = self._test_data_cache
        else:
            # 大数据集按需加载
            test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)

        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        # 使用更大的batch_size加速测试
        testloaderfull = self.load_test_data(batch_size=self.test_batch_size)
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data(batch_size=self.test_batch_size)
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
