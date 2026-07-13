import os
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter
from models import DMANet
from utils.tools import ensure_path


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'DMANet':DMANet,
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        self.writer = None

        self.epoch = 0
        self.step = 0

        self.output_pred = args.output_pred
        self.output_vis = args.output_vis
    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def _create_writer(self, log_dir):
        item_list = os.listdir(log_dir)
        item_path_list = [os.path.join(log_dir, item) for item in item_list]
        item_path_list = [item_path for item_path in item_path_list if os.path.isfile(item_path)]
        if len(item_path_list) > 0:
            pre_log_dir = os.path.join(log_dir, "pre_logs")
            ensure_path(pre_log_dir)

            item_list = [os.path.basename(item_path) for item_path in item_path_list]
            for item, item_path in zip(item_list, item_path_list):
                shutil.move(item_path, os.path.join(pre_log_dir, item))

        return SummaryWriter(log_dir)
