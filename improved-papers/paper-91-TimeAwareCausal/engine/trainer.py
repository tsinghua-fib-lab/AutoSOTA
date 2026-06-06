# -*- coding:utf-8 -*-
import os
import json
import logging
import numpy as np
from pathlib import Path
from PIL import ImageFile

import torch
import torch.nn as nn

import engine.inits as inits
from engine import trainval_tool as tv
from engine.utils import Checkpointer, mean_var, format_time, MyDataParallel, apply_dataparallel
import sys
from engine.inits import colorstr
from engine.utils.general import increment_path, Pool

sys.dont_write_bytecode = True
ImageFile.LOAD_TRUNCATED_IMAGES = True
# from torch.utils.tensorboard import SummaryWriter


class Trainer(object):
    def __init__(self, cfg):
        print('Constructing components...')

        # basic settings
        self.cfg = cfg
        self.gpu_ids = str(cfg.gpu_ids)
        self.epochs = cfg.epochs
        self.stage = cfg.mode
        self.cur_time = format_time()
        self.output_path = Path(cfg.save_path + '/' + self.cur_time)
        os.makedirs(cfg.save_path, exist_ok=True)
        self.cfg.data_size = json.loads(cfg.data_size) if isinstance(cfg.data_size, str) else cfg.data_size

        # seed and stage
        inits.set_seed(cfg.seed)

        # To cuda
        print('GPUs id:{}'.format(self.gpu_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_ids
        self.use_cuda = torch.cuda.is_available()

        # components
        self.algorithm = inits.get_algorithm(cfg)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # multi gpu
        if len(self.gpu_ids.split(',')) > 1:
            apply_dataparallel(self.algorithm)
            print('GPUs:', torch.cuda.device_count())
            print('Using CUDA...')
        if self.use_cuda:
            self.algorithm.cuda()

        self.train_loader, self.val_loader, self.test_loader = inits.get_minibatches_iterators(cfg)

        self.test_epoch = cfg.test_epoch
        self.start_epoch = 0

        # log and checkpoint
        self.checkpointer = Checkpointer(self.output_path, self.algorithm, cfg.seed)

        self.writer = None
        self.logger = inits.get_logger(cfg, self.output_path)
        self.logger.setLevel(logging.INFO)
        self.set_training_stage(self.stage)

        self.logger.info(colorstr("Setting and Parameters: ") + ", ".join(f"{k}={v}" for k, v in vars(self.cfg).items()))
        self.logger.info(colorstr("learning rate: ") + "{}".format(self.algorithm.hparams["lr"]))

        self.best_val_acc = 0.

    def set_training_stage(self, stage):
        stage = stage.strip().lower()
        if stage == 'train':
            self.stage = 2

        elif stage == 'val' or stage == 'test':
            self.stage = 1
            self.checkpointer.load_model(self._get_load_name(self.test_epoch))

        elif stage == 'continue':
            self.stage = 2
            # name needs to be specialized
            start_model = self._get_load_name(-2)
            self.start_epoch = self.checkpointer.load_model(start_model)

    @staticmethod
    def _get_load_name(self, epoch=-1):
        if epoch == -1:
            model_name = 'best'
        elif epoch == -2:
            model_name = None
        else:
            model_name = str(epoch)
        return model_name

    def _train_net(self, epoch):
        losses, acc1 = tv.train(args=self.cfg,
                                epoch=epoch,
                                algorithm=self.algorithm,
                                minibatches_loader=self.train_loader,
                                use_cuda=self.use_cuda,
                                writer=self.writer)
        return losses, acc1

    def _val_net(self, epoch, dataloader):
        loss, top1, accuracies = tv.val(algorithm=self.algorithm,
                                        dataloaders=dataloader,
                                        use_cuda=self.use_cuda,
                                        progress_bar=False,
                                        writer=self.writer)
        return loss, top1, accuracies

    def train(self):
        if self.stage >= 2:
            for epoch_item in range(self.start_epoch, self.epochs):
                self.logger.info('==================================== Epoch %d ===================================='
                                 % epoch_item)
                train_loss, train_top1 = self._train_net(epoch_item)
                self.logger.info('Epoch:{} || Train {}, Train {}'.format(epoch_item, train_top1, train_loss))

                if hasattr(self.algorithm, "prepare_for_inference"):
                    self.algorithm.prepare_for_inference()

                val_loss, val_top1, _ = self._val_net(epoch_item, self.val_loader)
                test_loss, test_top1, _ = self._val_net(epoch_item, self.test_loader)

                cur_val_top1 = np.mean(np.array(val_top1))
                if cur_val_top1 > self.best_val_acc:
                    self.best_val_acc = cur_val_top1
                    self.checkpointer.save_model('best', epoch_item)

                self.logger.info(
                    'Epoch:{} || Val  Acc@1: {} || Cur Val Avg@1: {:5.2f}, Best Val Avg@1: {:5.2f}\nVal Loss: {}'.format(
                    epoch_item, [round(val, 2) for val in val_top1], cur_val_top1, self.best_val_acc, val_loss))
                    
                self.logger.info(
                    'Epoch:{} || Test  Acc@1: {} || Test Loss: {}'.format(
                        epoch_item, [round(test, 2) for test in test_top1], test_loss))

                # Saving model params
                self.checkpointer.save_model('last', epoch_item)

            # self.writer.close()

        elif self.stage == 1:
            self.logger.info('==================================== Final Test ====================================')
            val_loss, val_top1, val_accs = self._val_net('val', self.val_loader)
            test_loss, test_top1, test_accs = self._val_net('val&test', self.test_loader)
            self.logger.info('Val  Acc@1: {}, Val Loss  {}'.format(val_top1, val_loss))
            self.logger.info('Test  Acc@1: {}, Test Loss {}'.format(test_top1, test_loss))

            self.logger.info('-----------------------------------  Report  --------------------------------------')
            # Record mean and var for each domain
            val_acc_vector, test_acc_vector = [], []
            for cur_acc in val_accs:
                cur_accuracy, h = mean_var(cur_acc)
                val_acc_vector.append((cur_accuracy, h))

            for cur_acc in test_accs:
                cur_accuracy, h = mean_var(cur_acc)
                test_acc_vector.append((cur_accuracy, h))
            self.logger.info('Val  Mean&Var: {}'.format(val_acc_vector))
            self.logger.info('Test  Mean&Var: {}'.format(test_acc_vector))
        else:
            raise ValueError('Stage is wrong!')
