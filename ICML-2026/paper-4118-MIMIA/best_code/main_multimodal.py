import os  # 导入os模块，用于处理文件和目录路径
from utils.args import parser_args  # 从自定义的utils.args模块导入参数解析函数（此处代码未提供，但功能是解析命令行参数）
import patch_fast_loading; patch_fast_loading.patch()  # Monkey-patch for fast precomputed data loading
from utils.datasets import *  # 从自定义的utils.datasets模块导入所有内容，可能包含数据加载和预处理函数
import copy  # 导入copy模块，用于深度复制对象（如模型权重）
import random  # 导入random模块，用于生成随机数
from tqdm import tqdm  # 导入tqdm库，用于显示进度条
import numpy as np  # 导入numpy库，用于进行科学计算，特别是数组操作
import math  # 导入math模块，提供数学运算函数
from scipy import spatial  # 从scipy库导入spatial模块，可能用于计算空间距离或相似度
import torch  # 导入PyTorch主库
from torch.utils.data import DataLoader  # 从PyTorch导入DataLoader，用于创建数据加载器
import torch.multiprocessing as mp  # 导入多进程模块，但在此代码中未使用
import time  # 导入time模块，用于处理时间相关的任务
import torch.optim as optim  # 导入PyTorch的优化器模块（如SGD, Adam）
import json  # 导入json模块，用于处理JSON格式的数据
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的神经网络函数库（如激活函数、损失函数）
import models as models  # 导入自定义的models模块，其中应包含模型架构的定义

from opacus import PrivacyEngine  # 导入Opacus库，用于实现差分隐私训练
from experiments.base import Experiment  # 从自定义的experiments.base模块导入Experiment基类
from experiments.trainer_private import TrainerPrivate, TesterPrivate  # 从自定义模块导入私有化训练和测试器
from experiments.utils import quant  # 从自定义模块导入量化函数
from argparse import Namespace  # 导入Namespace，用于创建一个简单的对象来存放属性
from models.basic_model import AVClassifier  # 从自定义模块导入AVClassifier模型
from opacus.validators import ModuleValidator

class FederatedLearning(Experiment):  # 定义一个名为FederatedLearning的类，继承自Experiment基类
    """
    Perform federated learning
    执行联邦学习
    """

    def __init__(self, args):  # 类的初始化方法
        super().__init__(args)  # 调用父类（Experiment）的初始化方法，从args中定义许多self属性
        self.watch_train_client_id = 0  # 设置要重点观察的训练客户端ID为0
        self.watch_val_client_id = 1  # 设置要重点观察的验证客户端ID为1
        self.balancedmethod = args.balancedmethod
        self.criterion = torch.nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失
        self.in_channels = 3  # 设置模型输入的通道数（例如RGB图像为3）
        self.optim = args.optim  # 从参数中获取优化器类型
        self.dp = args.dp  # 从参数中获取是否启用差分隐私（DP）
        self.defense = args.defense  # 从参数中获取防御方法的名称
        self.sigma = args.sigma  # 从参数中获取用于差分隐私的噪声标准差
        self.cosine_attack = args.cosine_attack  # 从参数中获取是否执行余弦相似度攻击
        self.sigma_sgd = args.sigma_sgd  # 从参数中获取SGD中的噪声乘数
        self.grad_norm = args.grad_norm  # 从参数中获取梯度裁剪的范数阈值
        self.save_dir = args.save_dir  # 从参数中获取保存结果的目录
        if not os.path.exists(self.save_dir):  # 如果保存目录不存在
            os.makedirs(self.save_dir)  # 则创建该目录
        self.data_root = args.data_root  # 从参数中获取数据集的根目录

        print('==> Preparing data...')  # 打印提示信息，表示正在准备数据
        # 调用get_data函数加载和划分数据集
        self.train_set, self.test_set, self.train_set_mia, self.test_set_mia, self.dict_users, self.train_idxs, self.val_idxs = get_data(
            dataset=self.dataset,  # 数据集名称
            data_root=self.data_root,  # 数据集根目录
            iid=self.iid,  # 是否独立同分布（IID）划分
            num_users=self.num_users,  # 客户端数量
            data_aug=self.args.data_augment,  # 是否进行数据增强
            noniid_beta=self.args.beta  # non-IID划分时使用的Dirichlet分布参数
            )

        print(len(self.train_set), len(self.test_set))  # 打印训练集和测试集的总大小
        print(len(self.train_idxs[0]), len(self.train_idxs[1]))  # 打印客户端0和客户端1的训练数据索引数量
        if  'cremad' in self.args.dataset:
            self.num_classes = 6
        elif 'AVE' in self.args.dataset:
            self.num_classes = 28
        elif self.args.dataset == 'balance':
            self.num_classes = 30
        self.MIA_trainset_dir = []  # 初始化用于成员推理攻击（MIA）的训练集目录列表
        self.MIA_valset_dir = []  # 初始化用于MIA的验证集目录列表
        self.MIA_trainset_dir_cos = []  # 初始化用于余弦攻击的MIA训练集目录列表
        self.MIA_valset_dir_cos = []  # 初始化用于余弦攻击的MIA验证集目录列表
        self.train_idxs_cos = []  # 初始化用于余弦攻击的训练索引列表
        self.testset_idx = (50000 + np.arange(10000)).astype(int)  # 定义测试集的索引（假设训练集有50000个样本）

        print('==> Preparing model...')  # 打印提示信息，表示正在准备模型

        # 初始化一个字典用于记录训练过程中的各种日志
        self.logs = {
            # 训练集指标
            'train_loss': [],
            'train_loss_a': [],
            'train_loss_v': [],
            'train_acc': [],
            'train_acc_a': [],
            'train_acc_v': [],
            'train_sign_acc': [],

            # 验证集指标
            'val_loss': [],
            'val_loss_a': [],
            'val_loss_v': [],
            'val_acc': [],
            'val_acc_a': [],
            'val_acc_v': [],

            # 测试集指标
            'test_loss': [],
            'test_loss_a': [],  # 新增：测试集音频loss
            'test_loss_v': [],  # 新增：测试集视觉loss
            'test_acc': [],
            'test_acc_a': [],  # 新增：测试集音频acc
            'test_acc_v': [],  # 新增：测试集视觉acc

            # 其他
            'keys': [],
            'local_loss': [],
            'best_test_acc': -np.inf,
            'best_model': [],
        }

        self.construct_model()  # 调用方法构建模型

        self.w_t = copy.deepcopy(self.model.state_dict())  # 深度复制初始模型权重，作为全局模型

        # 实例化一个私有训练器
        self.trainer = TrainerPrivate(self.model, self.train_set, self.device, self.dp, self.sigma, self.num_classes,self.balancedmethod,
                                      self.defense, args.klam, args.up_bound, args.mix_alpha)
        # 实例化一个私有测试器
        self.tester = TesterPrivate(self.model, self.device)

    def construct_model(self):  # 定义构建模型的方法
        model = AVClassifier(self.args)  # 实例化AVClassifier模型
        # model = torch.nn.DataParallel(model) # 注释掉了数据并行（用于多GPU）
        if not ModuleValidator.is_valid(model):
            print("发现不兼容的层，正在自动修复模型...")
            model = ModuleValidator.fix(model)
            print("模型修复完成。")

        self.model = model.to(self.device)  # 将模型移动到指定的设备（如GPU）

        torch.backends.cudnn.benchmark = True  # 启用cuDNN的自动调优功能，可以加速计算
        print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))  # 计算并打印模型的总参数量

    def train(self):  # 定义训练方法
        # 为整个训练集和测试集创建数据加载器，主要用于评估全局模型的性能
        train_ldr = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=8)
        val_ldr = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=8)
        test_ldr = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=8)

        local_train_ldrs = []  # 初始化一个列表，用于存放每个客户端的本地数据加载器
        if args.iid:  # 如果是IID数据划分
            for i in range(self.num_users):  # 遍历所有客户端
                if args.defense == 'instahide':  # 如果使用instahide防御
                    self.batch_size = len(self.dict_users[i])  # batch_size设置为该客户端的全部数据量
                # 创建一个只包含该客户端数据的DataLoader
                local_train_ldr = DataLoader(DatasetSplit(self.train_set, self.dict_users[i]),
                                             batch_size=self.batch_size,
                                             shuffle=True, num_workers=8)
                local_train_ldrs.append(local_train_ldr)  # 将创建的DataLoader添加到列表中
        else:  # 如果是Non-IID数据划分
            for i in range(self.num_users):  # 遍历所有客户端
                # self.dict_users[i]此时已经是Subset对象
                local_train_ldr = DataLoader(DatasetSplit(self.train_set, self.dict_users[i]),
                                             batch_size=self.batch_size,
                                             shuffle=True, num_workers=8)
                local_train_ldrs.append(local_train_ldr)  # 将创建的DataLoader添加到列表中

        total_time = 0  # 初始化总训练时间
        # 构建日志文件名
        file_name = "_".join(
            ['a', args.model_name, args.dataset, args.modality, str(args.num_users), str(args.optim), str(args.lr_up),
             str(args.batch_size), str(time.strftime("%Y_%m_%d_%H%M%S", time.localtime()))])

        b = os.path.join(os.getcwd(), self.save_dir)  # 获取保存目录的完整路径
        if not os.path.exists(b):  # 如果目录不存在
            os.makedirs(b)  # 创建目录
        fn = os.path.join(b, file_name + '.log')  # 拼接成完整的日志文件路径
        print("training log saved in:", fn)  # 打印日志文件保存路径

        lr_0 = self.lr  # 保存初始学习率

        for epoch in range(self.epochs):  # 开始主训练循环，遍历每一轮（epoch）
            global_state_dict = copy.deepcopy(self.model.state_dict())  # 深度复制当前全局模型的状态字典

            if self.sampling_type == 'uniform':  # 如果客户端采样策略是均匀采样
                self.m = max(int(self.frac * self.num_users), 1)  # 计算本轮要选择的客户端数量
                idxs_users = np.random.choice(range(self.num_users), self.m, replace=False)  # 无放回地随机选择客户端

            local_ws, local_losses, = [], []  # 初始化列表，用于存储本轮各客户端的本地模型权重和损失

            start = time.time()  # 记录本轮开始时间
            for idx in tqdm(idxs_users, desc='Epoch:%d, lr:%f' % (epoch, self.lr)):  # 遍历被选中的客户端，并使用tqdm显示进度条
                self.model.load_state_dict(global_state_dict)  # 将当前客户端的模型重置为全局模型
                client_train_indices = []
                for step, (index, spec, image, label) in enumerate(local_train_ldrs[idx]):
                    client_train_indices.extend(index.cpu().numpy().tolist())
                val_indices = []
                for step, (index, spec, image, label) in enumerate(val_ldr):
                    val_indices.extend(index.cpu().numpy().tolist())
                # 调用训练器的本地更新方法，进行本地训练
                local_w, local_loss = self.trainer._local_update_noback(local_train_ldrs[idx], self.local_ep, self.lr,
                                                                        self.optim, args.sampling_proportion)


                test_loss, test_acc, loss_train_a, acc_train_a, loss_train_v, acc_train_v = self.trainer.test(val_ldr)

                local_ws.append(copy.deepcopy(local_w))  # 将该客户端的本地模型权重添加到列表中
                local_losses.append(local_loss)  # 将该客户端的本地损失添加到列表中

                # 如果MIA模式开启，并且在特定的epoch，则保存用于MIA分析的数据
                if args.MIA_mode == 1 and epoch>45  and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch in args.schedule_milestone or epoch - 1 in args.schedule_milestone or epoch - 2 in args.schedule_milestone) == 1:
                    save_dict = {}  # 初始化一个字典用于保存数据
                    save_dict['test_acc'] = test_acc  # 保存测试准确率
                    save_dict['test_loss'] = test_loss  # 保存测试损失
                    crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')  # 定义一个不进行均值化的交叉熵损失函数，以获取每个样本的损失
                    device = torch.device("cuda")  # 定义设备

                    # 计算并保存测试集上每个样本的损失、logits和标签
                    test_ldr_mia = DataLoader(self.test_set_mia, batch_size=self.batch_size, shuffle=False,
                                              num_workers=0)
                    test_res = get_all_losses(test_ldr_mia, self.model, crossentropy_noreduce, device)
                    save_dict['test_index'] = self.testset_idx
                    save_dict['test_res'] = test_res
                    save_dict['client_train_indices'] = client_train_indices
                    save_dict['val_indices'] = val_indices
                    # 计算并保存目标客户端（成员）的每个样本的损失、logits和标签
                    train_res = get_all_losses_from_indexes(self.train_set_mia,
                                                            self.train_idxs[self.watch_train_client_id], self.model)
                    save_dict['train_index'] = self.train_idxs[self.watch_train_client_id]
                    save_dict['train_res'] = train_res

                    # 计算并保存另一个客户端（非成员）的每个样本的损失、logits和标签
                    val_res = get_all_losses_from_indexes(self.train_set_mia, self.train_idxs[self.watch_val_client_id],
                                                          self.model)
                    save_dict['val_index'] = self.train_idxs[self.watch_val_client_id]
                    save_dict['val_res'] = val_res

                    # 准备混合数据集（来自其他客户端的数据）用于MIA分析
                    mixed_indexs = []
                    needed_test_indexs = []
                    if self.args.dataset == 'cifar100':
                        data_num = int(10000 / self.num_users)  # 每个客户端抽样数量
                        needed_test_indexs = random.sample(list(range(0, 10000)), data_num)  # 从测试集中抽样
                        save_dict['needed_test_index'] = needed_test_indexs
                    elif self.args.dataset == 'dermnet':
                        data_num = 300
                        needed_test_indexs = None
                    for c_id in range(1, self.num_users):  # 从除了目标客户端以外的其他客户端抽样
                        # mixed_indexs.extend(random.sample(list(self.train_idxs[c_id]), 1200))
                        mixed_indexs.extend(list(self.train_idxs[c_id]))
                    # 计算并保存混合数据集上每个样本的损失、logits和标签
                    mix_res = get_all_losses_from_indexes(self.train_set_mia, mixed_indexs, self.model)
                    save_dict['mix_index'] = mixed_indexs
                    save_dict['mix_res'] = mix_res

                    if self.cosine_attack == True:  # 如果启用了余弦相似度攻击

                        ## 计算模型更新量（梯度）
                        model_grads_list = []
                        model_grads_a_list = []
                        model_grads_v_list = []

                        # 2. 遍历模型的所有可训练参数
                        for name, local_param in self.model.named_parameters():
                            if local_param.requires_grad == True:
                                # 3. 计算全局模型和本地模型更新后的参数差异
                                para_diff = global_state_dict[name] - local_w[name]

                                # 将参数差异展平，并从计算图中分离
                                flattened_diff = para_diff.detach().cpu().flatten()

                                # 4. 将展平后的梯度向量添加到全模型的列表中
                                model_grads_list.append(flattened_diff)

                                # 5. 【新增】根据参数名称判断其所属模态，并添加到相应的列表中
                                if 'audio' in name:
                                    model_grads_a_list.append(flattened_diff)
                                if 'visual' in name:
                                    model_grads_v_list.append(flattened_diff)

                        # 6. 将每个列表中的所有梯度向量拼接成一个单一的、长的一维张量
                        model_grads = torch.cat(model_grads_list, -1)  # 全模型的梯度向量
                        model_grads_a = torch.cat(model_grads_a_list, -1)  # 纯音频模态的梯度向量
                        model_grads_v = torch.cat(model_grads_v_list, -1)  # 纯视觉模态的梯度向量
                        ## 计算余弦分数和梯度差异分数



                        # 2. 创建一个临时的、新的大字典，专门用来存放所有模态的余弦相似度结果
                        #    我们不再直接写入旧的 save_dict，避免信息混乱
                        all_modalities_to_test = ['full', 'audio', 'visual']

                        # 1. 【关键】创建一个新的、临时的空字典来收集结果
                        # The save_dict will now be a nested dictionary
                        # Example: save_dict['full']['tarin_cos'], save_dict['audio']['tarin_cos']
                        for modality_to_run in all_modalities_to_test:
                            print(
                                f"\n===== [Epoch {epoch + 1}, Client {idx}] Running Cosine Attack for Modality: '{modality_to_run}' =====")

                            # 4. 在循环内部创建cos_model实例
                            #    确保每次分析都使用一个干净的、未被Opacus污染的模型副本
                            cos_model = AVClassifier(self.args)
                            if not ModuleValidator.is_valid(cos_model):
                                cos_model = ModuleValidator.fix(cos_model)
                            cos_model = cos_model.to(self.device)
                            cos_model.load_state_dict(global_state_dict)

                            # 5. 调用get_all_cos，并将当前循环的模态(modality_to_run)作为参数传入
                            train_cos, train_diffs, train_norm, val_cos, val_diffs, val_norm, test_cos, test_diffs, test_norm, mix_cos, mix_diffs, mix_norm,index_dict = get_all_cos(
                                cos_model, val_ldr, test_ldr_mia, self.test_set_mia, self.train_set_mia,
                                self.train_idxs[self.watch_train_client_id],
                                self.train_idxs[self.watch_val_client_id],
                                mixed_indexs,
                                needed_test_indexs,
                                model_grads,
                                model_grads_a,
                                model_grads_v,
                                self.lr, self.optim,
                                modality_to_train=modality_to_run  # <-- 关键：传入当前循环的模态
                            )
                            save_dict['index']=index_dict
                            save_dict[modality_to_run] = {}
                        # 4. Save the results into the sub-dictionary for the current modality
                            save_dict[modality_to_run]['tarin_cos'] = train_cos
                            save_dict[modality_to_run]['val_cos'] = val_cos
                            save_dict[modality_to_run]['test_cos'] = test_cos
                            save_dict[modality_to_run]['mix_cos'] = mix_cos

                            save_dict[modality_to_run]['tarin_diffs'] = train_diffs
                            save_dict[modality_to_run]['val_diffs'] = val_diffs
                            save_dict[modality_to_run]['test_diffs'] = test_diffs
                            save_dict[modality_to_run]['mix_diffs'] = mix_diffs

                            save_dict[modality_to_run]['tarin_grad_norm'] = train_norm
                            save_dict[modality_to_run]['val_grad_norm'] = val_norm
                            save_dict[modality_to_run]['test_grad_norm'] = test_norm
                            save_dict[modality_to_run]['mix_grad_norm'] = mix_norm
                        del cos_model
                        del train_cos, train_diffs, train_norm, val_cos, val_diffs, val_norm, test_cos, test_diffs, test_norm, mix_cos, mix_diffs, mix_norm

                        # 2. 强制清空PyTorch的CUDA缓存
                        torch.cuda.empty_cache()

                    if not os.path.exists(os.path.join(os.getcwd(), self.save_dir)):  # 确保保存目录存在
                        os.makedirs(os.path.join(os.getcwd(), self.save_dir))
                        print('MIA Score Saved in:', os.path.join(os.getcwd(), self.save_dir))
                    # 将包含所有MIA和余弦攻击数据的字典保存到文件中
                    torch.save(save_dict,
                               os.path.join(os.getcwd(), self.save_dir, f'client_{idx}_losses_epoch{epoch + 1}.pkl'))

            # 更新学习率
            if self.optim == "sgd":  # 如果优化器是SGD
                if self.args.lr_up == 'common':  # 普通衰减策略
                    self.lr = self.lr * 0.99
                elif self.args.lr_up == 'milestone':  # 里程碑式衰减策略
                    if epoch in self.args.schedule_milestone:
                        self.lr *= 0.1
                else:  # 余弦退火策略
                    self.lr = lr_0 * (1 + math.cos(math.pi * epoch / self.args.epochs)) / 2
            else:  # 其他优化器（如Adam）不改变学习率
                pass

            # 计算每个客户端的权重（根据其数据量占总数据量的比例）
            client_weights = []
            for i in range(self.num_users):
                client_weight = len(DatasetSplit(self.train_set, self.dict_users[i])) / len(self.train_set)
                client_weights.append(client_weight)

            # 执行联邦平均
            self._fed_avg(local_ws, client_weights, 1)
            self.model.load_state_dict(self.w_t)  # 将聚合后的全局权重加载到模型中
            end = time.time()  # 记录本轮结束时间
            interval_time = end - start  # 计算本轮耗时
            total_time += interval_time  # 累加总耗时

            if (epoch + 1) == self.epochs or (epoch + 1) % 1 == 0:  # 在每轮结束或最后一轮时评估模型
                loss_train_mean, acc_train_mean, loss_train_a, acc_train_a, loss_train_v, acc_train_v = self.trainer.test(
                    train_ldr)
                loss_val_mean, acc_val_mean, loss_val_a, acc_val_a, loss_val_v, acc_val_v = self.trainer.test(val_ldr)
                loss_test_mean, acc_test_mean = loss_val_mean, acc_val_mean  # 此处将测试集性能等同于验证集

                # 记录日志
                self.logs['train_acc'].append(acc_train_mean)
                self.logs['train_acc_a'].append(acc_train_a)  # 训练集音频准确率
                self.logs['train_acc_v'].append(acc_train_v)  # 训练集视觉准确率
                self.logs['train_loss'].append(loss_train_mean)
                self.logs['train_loss_a'].append(loss_train_a)
                self.logs['train_loss_v'].append(loss_train_v)
                self.logs['val_acc'].append(acc_val_mean)
                self.logs['val_acc_a'].append(acc_val_a)  # 验证集音频准确率
                self.logs['val_acc_v'].append(acc_val_v)  # 验证集视觉准确率
                self.logs['val_loss'].append(loss_val_mean)
                self.logs['val_loss_a'].append(loss_val_a)
                self.logs['val_loss_v'].append(loss_val_v)
                self.logs['local_loss'].append(np.mean(local_losses))

                # 如果当前验证准确率更高，则更新最佳准确率并保存模型
                if self.logs['best_test_acc'] < acc_val_mean:
                    self.logs['best_test_acc'] = acc_val_mean
                    self.logs['best_test_loss'] = loss_val_mean
                    self.logs['best_model'] = copy.deepcopy(self.model.state_dict())

                # 打印本轮训练信息
                print('Epoch {}/{}  --time {:.1f}'.format(
                    epoch, self.epochs,
                    interval_time
                ))
                print(
                    "Train Loss {:.4f} --- Val Loss {:.4f}"
                    .format(loss_train_mean, loss_val_mean))
                print("Train acc {:.4f} --- Val acc {:.4f} --Best acc {:.4f}".format(acc_train_mean, acc_val_mean,
                                                                                     self.logs[
                                                                                         'best_test_acc']
                                                                                     ))
                # 将本轮的简要日志写入文件
                s = 'epoch:{}, lr:{:.5f}, val_acc:{:.4f}, val_loss:{:.4f}, tarin_acc:{:.4f}, train_loss:{:.4f},time:{:.4f}, total_time:{:.4f}'.format(
                    epoch, self.lr, acc_val_mean, loss_val_mean, acc_train_mean, loss_train_mean, interval_time,
                    total_time)
                # with open(fn, "a") as f:
                #     json.dump({"epoch": epoch, "lr": round(self.lr, 5), "train_acc": round(acc_train_mean, 4),
                #                "test_acc": round(acc_val_mean, 4), "time": round(total_time, 2)}, f)
                #     f.write('\n')
                log_data = {
                    "epoch": epoch,
                    "lr": round(self.lr, 5),
                    "time": round(total_time, 2),

                    # 训练集指标 (Full, Audio, Visual)
                    "train_loss_full": round(loss_train_mean, 4),
                    "train_acc_full": round(acc_train_mean, 4),
                    "train_loss_audio": round(loss_train_a, 4),
                    "train_acc_audio": round(acc_train_a, 4),
                    "train_loss_visual": round(loss_train_v, 4),
                    "train_acc_visual": round(acc_train_v, 4),

                    # 验证集(Test)指标 (Full, Audio, Visual)
                    "test_loss_full": round(loss_val_mean, 4),
                    "test_acc_full": round(acc_val_mean, 4),
                    "test_loss_audio": round(loss_val_a, 4),
                    "test_acc_audio": round(acc_val_a, 4),
                    "test_loss_visual": round(loss_val_v, 4),
                    "test_acc_visual": round(acc_val_v, 4)
                }

                # 将字典写入 JSON 文件
                with open(fn, "a") as f:
                    json.dump(log_data, f)
                    f.write('\n')
        print('------------------------------------------------------------------------')
        # 训练结束后打印最佳测试结果
        print('Test loss: {:.4f} --- Test acc: {:.4f}  '.format(self.logs['best_test_loss'],
                                                                self.logs['best_test_acc']
                                                                ))

        return self.logs, interval_time, self.logs['best_test_acc'], acc_test_mean  # 返回日志、时间、最佳准确率和最后一次的准确率

    def _fed_avg(self, local_ws, client_weights, lr_outer):  # 定义联邦平均方法
        w_avg = copy.deepcopy(local_ws[0])  # 用第一个客户端的权重作为基础
        for k in w_avg.keys():  # 遍历模型每一层的参数
            w_avg[k] = w_avg[k] * client_weights[0]  # 乘以该客户端的权重

            for i in range(1, len(local_ws)):  # 遍历剩下的客户端
                w_avg[k] += local_ws[i][k] * client_weights[i]  # 累加其他客户端的加权参数

            self.w_t[k] = w_avg[k]  # 更新全局模型（self.w_t）的参数


# 以下是辅助函数

def get_loss_distributions(idx, MIA_trainset_dir, MIA_testloader, MIA_valset_dir, model):
    """ 获取成员和非成员损失分布 """
    crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')  # 不做reduce的损失函数
    device = torch.device("cuda")  # 设备
    train_res = get_all_losses(MIA_trainset_dir[idx], model, crossentropy_noreduce, device)  # 获取训练集（成员）的损失
    test_res = get_all_losses(MIA_testloader, model, crossentropy_noreduce, device)  # 获取测试集（非成员）的损失
    val_res = get_all_losses(MIA_valset_dir[idx], model, crossentropy_noreduce, device)  # 获取验证集（非成员）的损失
    return train_res, test_res, val_res  # 返回结果


def get_all_losses(dataloader, model, criterion, device, req_logits=False):
    """ 计算数据加载器中每个样本的损失 """
    model.eval()  # 将模型设置为评估模式
    losses = []  # 损失列表
    logits = []  # logits列表
    labels = []  # 标签列表
    losses_a = []  # 损失列表
    logits_a = []  # logits列表
    losses_v = []  # 损失列表
    logits_v = []  # logits列表
    all_index=[]
    with torch.no_grad():  # 不计算梯度，以节省计算资源
        for step, (index,spec, image, label) in enumerate(dataloader):
        # for batch_idx, (inputs, targets) in enumerate(dataloader):  # 遍历数据
        #     inputs, targets = inputs.to(device), targets.to(device)  # 数据移动到GPU
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)


            a, v, out = model(spec.unsqueeze(1).float(), image.float())

            weight_size = model.fusion_module.fc_out.weight.size(1)
            out_v = (torch.mm(v, torch.transpose(
                model.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                     + model.fusion_module.fc_out.bias / 2)

            out_a = (torch.mm(a, torch.transpose(
                model.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                     + model.fusion_module.fc_out.bias / 2)
            loss = criterion(out, label)
            loss_v = criterion(out_v, label)
            loss_a = criterion(out_a, label)
            losses.append(loss.cpu().numpy())  # 保存损失
            losses_a.append(loss_a.cpu().numpy())  # 保存损失
            losses_v.append(loss_v.cpu().numpy())
            all_index.append(index.cpu().numpy())
            logits.append(out.cpu())  # 保存logits
            logits_v.append(out_v.cpu())
            logits_a.append(out_a.cpu())
            labels.append(label.cpu())  # 保存标签

    losses = np.concatenate(losses)  # 将列表拼接成一个numpy数组
    losses_a = np.concatenate(losses_a)
    losses_v = np.concatenate(losses_v)
    logits = torch.cat(logits)  # 将列表拼接成一个tensor
    logits_v = torch.cat(logits_v)
    logits_a = torch.cat(logits_a)
    labels = torch.cat(labels)  # 将列表拼接成一个tensor
    all_index = np.concatenate(all_index)
    return {"loss": losses, "loss_a":losses_a,"loss_v":losses_v,"logit": logits,"logit_a":logits_a ,"logit_v":logits_v ,"labels": labels,"index":all_index}  # 返回包含损失、logits和标签的字典


def get_all_losses_from_indexes(dataset, indexes, model):
    """ 从给定的数据索引中计算每个样本的损失 """
    criterion = nn.CrossEntropyLoss(reduction='none')
    device = torch.device("cuda")
    # 根据索引创建一个临时的DataLoader
    dataloader = DataLoader(DatasetSplit(dataset, indexes), batch_size=200, shuffle=False, num_workers=0)
    model.eval()
    losses = []  # 损失列表
    logits = []  # logits列表
    labels = []  # 标签列表
    losses_a = []  # 损失列表
    logits_a = []  # logits列表
    losses_v = []  # 损失列表
    logits_v = []  # logits列表
    all_index = []
    with torch.no_grad():
        for step, (index,spec, image, label) in enumerate(dataloader):
            # for batch_idx, (inputs, targets) in enumerate(dataloader):  # 遍历数据
            # inputs, targets = inputs.to(device), targets.to(device)  # 数据移动到GPU
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            a, v, out = model(spec.unsqueeze(1).float(), image.float())

            weight_size = model.fusion_module.fc_out.weight.size(1)
            out_v = (torch.mm(v, torch.transpose(
                model.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                     + model.fusion_module.fc_out.bias / 2)

            out_a = (torch.mm(a, torch.transpose(
                model.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                     + model.fusion_module.fc_out.bias / 2)
            loss = criterion(out, label)
            loss_v = criterion(out_v, label)
            loss_a = criterion(out_a, label)
            losses.append(loss.cpu().numpy())  # 保存损失
            losses_a.append(loss_a.cpu().numpy())  # 保存损失
            losses_v.append(loss_v.cpu().numpy())
            logits.append(out.cpu())  # 保存logits
            logits_v.append(out_v.cpu())
            logits_a.append(out_a.cpu())
            labels.append(label.cpu())  # 保存标签
            all_index.append(index.cpu().numpy())

    losses = np.concatenate(losses)  # 将列表拼接成一个numpy数组
    losses_a = np.concatenate(losses_a)
    losses_v = np.concatenate(losses_v)
    logits = torch.cat(logits)  # 将列表拼接成一个tensor
    logits_v = torch.cat(logits_v)
    logits_a = torch.cat(logits_a)
    labels = torch.cat(labels)  # 将列表拼接成一个tensor
    all_index = np.concatenate(all_index)
    return {"loss": losses, "loss_a":losses_a,"loss_v":losses_v,"logit": logits,"logit_a":logits_a ,"logit_v":logits_v ,"labels": labels,"index":all_index}  # 返回包含损失、logits和标签的字典


def get_all_cos(cos_model, initial_loader, test_dataloader, test_set, train_set, train_idxs, val_idxs, mix_idxs,
                needed_test_indexs, model_grads,model_grads_a,model_grads_v, lr, optim_choice,modality_to_train='full'):
    """ 计算所有数据集上的余弦相似度分数 """
    device = torch.device("cuda")
    # 根据选择设置优化器
    if optim_choice == "sgd":
        optimizer = optim.SGD(cos_model.parameters(), lr, momentum=0.9, weight_decay=0.0005)
    else:
        optimizer = optim.AdamW(cos_model.parameters(), lr, weight_decay=0.0005)

    privacy_engine = PrivacyEngine()  # 实例化Opacus的隐私引擎
    # 使用make_private包装模型和优化器，这是为了能够方便地获取每个样本的梯度（per-sample gradients）
    # 即使noise_multiplier=0，这个包装也是必要的
    cos_model, optimizer, samples_loader = privacy_engine.make_private(
        module=cos_model,
        optimizer=optimizer,
        data_loader=initial_loader,
        noise_multiplier=0,  # 噪声乘数设为0，因为这里只用它来计算per-sample梯度，不添加噪声
        max_grad_norm=1e10,  # 梯度裁剪阈值设得很大，相当于不裁剪
    )

    def extract_indices_from_loader(dataloader):
        all_indices = []
        # 因为 shuffle=False，这里的遍历顺序就是后续计算指标时的顺序
        for step, (index, spec, image, label) in enumerate(dataloader):
            # index 是一个 Tensor (batch_size,)
            # 将其转为 numpy 或 list 并存起来
            all_indices.append(index.cpu().numpy())

        # 将所有 batch 的索引拼接成一个长数组
        return np.concatenate(all_indices)

    # 为不同数据集创建DataLoader
    tarin_dataloader = DataLoader(DatasetSplit(train_set, train_idxs), batch_size=10, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_set, batch_size=10, shuffle=False,
                                 num_workers=0)
    mix_dataloader = DataLoader(DatasetSplit(train_set, mix_idxs), batch_size=10, shuffle=False, num_workers=0)
    indices_dict = {}

    # 依次提取并存入字典
    indices_dict['train_indices'] = extract_indices_from_loader(tarin_dataloader)
    indices_dict['test_indices'] = extract_indices_from_loader(test_dataloader)
    indices_dict['mix_indices'] = extract_indices_from_loader(mix_dataloader)

    # 分别计算训练集、测试集、混合集上的余弦分数
    train_cos, train_diffs, train_norm = get_cos_score(tarin_dataloader, optimizer, cos_model, device, model_grads,model_grads_a,model_grads_v,modality_to_train=modality_to_train)
    test_cos, test_diffs, test_norm = get_cos_score(test_dataloader, optimizer, cos_model, device, model_grads,model_grads_a,model_grads_v,modality_to_train=modality_to_train)
    mix_cos, mix_diffs, mix_norm = get_cos_score(mix_dataloader, optimizer, cos_model, device, model_grads,model_grads_a,model_grads_v,modality_to_train=modality_to_train)
    val_cos, val_diffs, val_norm = None, None, None  # 验证集在此未计算

    return train_cos, train_diffs, train_norm, val_cos, val_diffs, val_norm, test_cos, test_diffs, test_norm, mix_cos, mix_diffs, mix_norm, indices_dict


def get_cos_score(samples_ldr, optimizer, cos_model, device, model_grads, model_grads_a, model_grads_v,
                  modality_to_train='full'):
    model_grads = model_grads.to(torch.device("cuda"))
    model_grads_a = model_grads_a.to(torch.device("cuda"))
    model_grads_v = model_grads_v.to(torch.device("cuda"))
    cos_model.train()

    # 1. 初始化所有列表
    cos_scores, grad_diffs, sample_grads = [], [], []
    cos_scores_a, grad_diffs_a, sample_grads_a = [], [], []
    cos_scores_v, grad_diffs_v, sample_grads_v = [], [], []

    model_diff_norm = torch.norm(model_grads, p=2, dim=0) ** 2
    model_diff_norm_a = torch.norm(model_grads_a, p=2, dim=0) ** 2
    model_diff_norm_v = torch.norm(model_grads_v, p=2, dim=0) ** 2

    total_samples_processed = 0

    for step, (index,spec, image, label) in enumerate(samples_ldr):
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        batch_size = label.size(0)
        total_samples_processed += batch_size

        # 2. 前向传播和选择性反向传播
        a, v, out = cos_model(spec.unsqueeze(1).float(), image.float(),modality_to_train=modality_to_train)
        weight_size = cos_model.fusion_module.fc_out.weight.size(1)
        out_v = (torch.mm(v, torch.transpose(
            cos_model.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                 + cos_model.fusion_module.fc_out.bias / 2)
        out_a = (torch.mm(a, torch.transpose(
            cos_model.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                 + cos_model.fusion_module.fc_out.bias / 2)

        loss = F.cross_entropy(out, label)
        loss_v = F.cross_entropy(out_v, label)
        loss_a = F.cross_entropy(out_a, label)

        if modality_to_train == 'full':
            loss.backward()
        elif modality_to_train == 'audio':
            loss_a.backward()
        elif modality_to_train == 'visual':
            loss_v.backward()

        # 3. 梯度收集 (只收集存在的梯度)
        sample_batch_grads_list, sample_batch_grads_a_list, sample_batch_grads_v_list = [], [], []
        for name, param in cos_model.named_parameters():
            if param.requires_grad and param.grad_sample is not None:
                actual_size = param.grad_sample.shape[0]
                if actual_size == batch_size:
                    # 音频层、融合层，正常处理
                    flattened_grad = param.grad_sample.flatten(start_dim=1)
                elif actual_size == batch_size * 3:
                    # 视觉层，Opacus把B*T=30当成batch了，折叠回B
                    param_shape = param.grad_sample.shape[1:]
                    reshaped = param.grad_sample.reshape(batch_size, 3, *param_shape)
                    summed = reshaped.sum(dim=1)  # (B, *param_shape) ✅
                    flattened_grad = summed.flatten(start_dim=1)  # (B, param.numel()) ✅
                else:
                    continue
                sample_batch_grads_list.append(flattened_grad)
                if 'audio' in name: sample_batch_grads_a_list.append(flattened_grad)
                if 'visual' in name: sample_batch_grads_v_list.append(flattened_grad)

        # 4. 安全地拼接Tensor
        sample_batch_grads_cat = torch.cat(sample_batch_grads_list, 1) if sample_batch_grads_list else torch.empty(
            batch_size, 0, device=device)
        sample_batch_grads_a_cat = torch.cat(sample_batch_grads_a_list,
                                             1) if sample_batch_grads_a_list else torch.empty(batch_size, 0,
                                                                                              device=device)
        sample_batch_grads_v_cat = torch.cat(sample_batch_grads_v_list,
                                             1) if sample_batch_grads_v_list else torch.empty(batch_size, 0,
                                                                                              device=device)

        # 5. --- 核心修改：严格根据 modality_to_train 参数来决定计算内容 ---
        if modality_to_train == 'full':
            for sample_grad in sample_batch_grads_cat:
                cos_scores.append(F.cosine_similarity(sample_grad, model_grads, dim=0))
                grad_diffs.append(model_diff_norm - torch.norm(model_grads - sample_grad, p=2, dim=0) ** 2)
                sample_grads.append(torch.norm(sample_grad, p=2, dim=0) ** 2)
            for sample_grad_a in sample_batch_grads_a_cat:
                cos_scores_a.append(F.cosine_similarity(sample_grad_a, model_grads_a, dim=0))
                grad_diffs_a.append(model_diff_norm_a - torch.norm(model_grads_a - sample_grad_a, p=2, dim=0) ** 2)
                sample_grads_a.append(torch.norm(sample_grad_a, p=2, dim=0) ** 2)
            for sample_grad_v in sample_batch_grads_v_cat:
                cos_scores_v.append(F.cosine_similarity(sample_grad_v, model_grads_v, dim=0))
                grad_diffs_v.append(model_diff_norm_v - torch.norm(model_grads_v - sample_grad_v, p=2, dim=0) ** 2)
                sample_grads_v.append(torch.norm(sample_grad_v, p=2, dim=0) ** 2)

        elif modality_to_train == 'audio':
            for sample_grad_a in sample_batch_grads_a_cat:
                cos_scores_a.append(F.cosine_similarity(sample_grad_a, model_grads_a, dim=0))
                grad_diffs_a.append(model_diff_norm_a - torch.norm(model_grads_a - sample_grad_a, p=2, dim=0) ** 2)
                sample_grads_a.append(torch.norm(sample_grad_a, p=2, dim=0) ** 2)

        elif modality_to_train == 'visual':
            for sample_grad_v in sample_batch_grads_v_cat:
                cos_scores_v.append(F.cosine_similarity(sample_grad_v, model_grads_v, dim=0))
                grad_diffs_v.append(model_diff_norm_v - torch.norm(model_grads_v - sample_grad_v, p=2, dim=0) ** 2)
                sample_grads_v.append(torch.norm(sample_grad_v, p=2, dim=0) ** 2)

    # 6. 在末尾，对未计算的指标列表创建全零Tensor
    zeros_tensor = torch.zeros(total_samples_processed)

    cos_full = torch.tensor([t.item() for t in cos_scores]).cpu() if cos_scores else zeros_tensor
    diff_full = torch.tensor([t.item() for t in grad_diffs]).cpu() if grad_diffs else zeros_tensor
    norm_full = torch.tensor([t.item() for t in sample_grads]).cpu() if sample_grads else zeros_tensor

    cos_a = torch.tensor([t.item() for t in cos_scores_a]).cpu() if cos_scores_a else zeros_tensor
    diff_a = torch.tensor([t.item() for t in grad_diffs_a]).cpu() if grad_diffs_a else zeros_tensor
    norm_a = torch.tensor([t.item() for t in sample_grads_a]).cpu() if sample_grads_a else zeros_tensor

    cos_v = torch.tensor([t.item() for t in cos_scores_v]).cpu() if cos_scores_v else zeros_tensor
    diff_v = torch.tensor([t.item() for t in grad_diffs_v]).cpu() if grad_diffs_v else zeros_tensor
    norm_v = torch.tensor([t.item() for t in sample_grads_v]).cpu() if sample_grads_v else zeros_tensor

    # 7. 返回部分 ()
    cos_metrics = {'full': cos_full, 'audio': cos_a, 'visual': cos_v}
    diff_metrics = {'full': diff_full, 'audio': diff_a, 'visual': diff_v}
    norm_metrics = {'full': norm_full, 'audio': norm_a, 'visual': norm_v}

    return cos_metrics, diff_metrics, norm_metrics


def main(args):  # 主函数
    logs = {'net_info': None,  # 初始化日志字典
            'arguments': {  # 存储实验参数
                'frac': args.frac,
                'local_ep': args.local_ep,
                'local_bs': args.batch_size,
                'lr_outer': args.lr_outer,
                'lr_inner': args.lr,
                'iid': args.iid,
                'wd': args.wd,
                'optim': args.optim,
                'model_name': args.model_name,
                'dataset': args.dataset,
                'log_interval': args.log_interval,
                'num_classes': args.num_classes,
                'epochs': args.epochs,
                'num_users': args.num_users
            }
            }
    save_dir = args.save_dir  # 保存目录
    fl = FederatedLearning(args)  # 实例化联邦学习实验

    logg, time, best_test_acc, test_acc = fl.train()  # 开始训练并获取结果

    logs['net_info'] = logg  # 将训练过程的详细日志保存到主日志字典中
    logs['test_acc'] = test_acc  # 保存最终的测试准确率
    logs['bp_local'] = True if args.bp_interval == 0 else False  # 记录一个标志位

    # 创建保存最终结果的目录
    if not os.path.exists(save_dir + args.model_name + '/' + args.dataset):
        os.makedirs(save_dir + args.model_name + '/' + args.dataset)
    # 将包含所有日志和参数的字典保存为.pkl文件
    torch.save(logs,
               save_dir + args.model_name + '/' + args.dataset + '/epoch_{}_E_{}_u_{}_{:.4f}_{:.4f}.pkl'.format(
                   args.epochs, args.local_ep, args.num_users, time, test_acc
               ))
    return


def setup_seed(seed):  # 设置随机种子以保证实验可复现
    torch.manual_seed(seed)  # 设置PyTorch的CPU随机种子
    torch.cuda.manual_seed_all(seed)  # 设置PyTorch的所有GPU的随机种子
    np.random.seed(seed)  # 设置Numpy的随机种子
    random.seed(seed)  # 设置Python内置的random模块的随机种子


if __name__ == '__main__':  # Python脚本的入口点
    # 使用字典定义默认参数，替代argparse
    default_args = {
        # federated learning parameters
        'num_users': 6,
        'save_dir': 'saved_mia_models',
        'log_folder_name': '/training_log_correct_iid/',
        'samples_per_user': 1100,
        'defense': 'none',
        'd_scale': 0.0,
        'klam': 3,
        'up_bound': 0.65,
        'mix_alpha': 0.01,
        'seed': 42,
        'frac': 1.0,
        'local_ep': 3,
        'batch_size': 256,
        'lr_outer': 1.0,
        'lr': 0.01,
        'lr_up': 'common',
        'schedule_milestone': [],
        'gamma': 0.99,
        'iid': 1,
        'MIA_mode': 1,
        'beta': 1.0,
        'wd': 1e-5,
        'optim': 'sgd',
        'epochs': 20,
        'sampling_type': 'uniform',
        'data_augment': 0,
        'sampling_proportion': 1.0,
        'lira_attack': True,
        'cosine_attack': True,

        # Model arguments
        'model_name': 'AVClassifer',
        'dataset': 'cremad',
        'data_root': '/datasets/CREMA-D',

        # Other parameters
        'gpu': '0',
        'num_classes': 6,
        'bp_interval': 30,
        'log_interval': 1,
        'exp-id': 1,
        'sigma_sgd': 0.0,
        'grad_norm': 1e4,

        # DP
        'dp': False,
        'sigma': 0.1,
        'modality': 'multimodal',
        'balancedmethod' :'OGM',
        "modality_balance" : 'OGM'
    }

    args = Namespace(**default_args)  # 将字典转换为Namespace对象，可以通过 . 访问属性
    print(args)  # 打印所有参数
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 设置CUDA设备顺序，使其与nvidia-smi显示的顺序一致
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 设置程序可见的GPU设备
    setup_seed(args.seed)  # 设置随机种子
    config_dict = vars(args)
    # 根据实验参数动态构建详细的保存目录路径
    args.save_dir = os.path.join(
        args.save_dir,
        f"{args.dataset}_K{args.num_users}_N{args.samples_per_user}_{args.model_name}_def{args.defense}_iid${args.iid}_${args.beta}_${args.optim}_local{args.local_ep}_s{args.seed}_mod${args.modality}_random"
    )
    print("scores saved in:", os.path.join(os.getcwd(), args.save_dir))  # 打印最终保存MIA分数的目录
    args.log_folder_name = args.save_dir  # 将日志文件夹名称设置为保存目录

    main(args)  # 调用主函数，开始实验