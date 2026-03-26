import os
import time
from pathlib import Path

import numpy as np
import torch

from utils import test_eval
from model import *

from dgl.nn import EdgeWeightNorm, GraphConv
from sklearn.metrics import roc_auc_score


def load_out_t(out_t_dir, name):
    return torch.from_numpy(np.load(out_t_dir.joinpath(name))["arr_0"])


def max_message(feature, adj_matrix):
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    sim_matrix = torch.mm(feature, feature.T)

    sim_matrix = torch.squeeze(sim_matrix) * adj_matrix
    sim_matrix[torch.isinf(sim_matrix)] = 0
    sim_matrix[torch.isnan(sim_matrix)] = 0

    row_sum = torch.sum(adj_matrix, 0)
    r_inv = torch.pow(row_sum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.

    message = torch.sum(sim_matrix, 1)

    message = message * r_inv

    return - torch.sum(message), message


# def max_message(feature, adj_matrix):
#     # 归一化特征
#     feature = feature / torch.norm(feature, dim=-1, keepdim=True)
#
#     # 计算相似度矩阵，保持稀疏性
#     sim_matrix = torch.sparse.mm(feature, feature.T)  # 稀疏矩阵乘法
#
#     # 将相似度矩阵和邻接矩阵相乘，保持稀疏性
#     sim_matrix = sim_matrix.multiply(adj_matrix)  # 使用稀疏矩阵的乘法
#     sim_matrix = sim_matrix.coalesce()  # 确保它是压缩的稀疏矩阵
#
#     # 处理无穷大和NaN
#     sim_matrix.values()[torch.isinf(sim_matrix.values())] = 0
#     sim_matrix.values()[torch.isnan(sim_matrix.values())] = 0
#
#     # 计算每个节点的度数的倒数
#     row_sum = torch.sparse.sum(adj_matrix, dim=0).to_dense()  # 转换为密集矩阵进行度数计算
#     r_inv = torch.pow(row_sum, -1).flatten()
#     r_inv[torch.isinf(r_inv)] = 0.
#
#     # 聚合每个节点的相似度信息，保留稀疏矩阵
#     message = torch.sparse.sum(sim_matrix, dim=1).to_dense()  # 转换为密集矩阵以执行聚合
#
#     # 按照度数的倒数进行加权
#     message = message * r_inv
#
#     # 返回负的加权和和每个节点的亲和力
#     return - torch.sum(message), message


def normalize_score(ano_score):
    ano_score = ((ano_score - np.min(ano_score)) / (
            np.max(ano_score) - np.min(ano_score)))
    return ano_score


class my_GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(my_GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, 2 * h_feats)
        self.conv2 = GraphConv(2 * h_feats, h_feats)

        self.fc1 = nn.Linear(h_feats, h_feats, bias=False)
        self.fc2 = nn.Linear(h_feats, h_feats, bias=False)

        # self.param_init()
        # self.fc1 = nn.Linear(h_feats, 1, bias=False)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        return h

    def get_final_predict(self, h):
        return torch.sigmoid(self.fc1(h))

    def param_init(self):
        nn.init.xavier_normal_(self.conv1.weight, gain=1.414)
        nn.init.xavier_normal_(self.conv2.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)


class IA_GGAD_Detector:
    def __init__(self, train_config, model_config, data, args):
        self.model_config = model_config
        self.train_config = train_config
        self.data = data
        self.args = args

        self.device = train_config['device']
        self.model = GCN(args, **model_config).to(train_config['device'])

        embedding_dim = 128
        self.GCN_model = my_GCN(model_config['in_feats'], embedding_dim).to(train_config['device'])

    def train(self, args):
        start_time = time.time()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.model_config['lr'],
                                      weight_decay=self.model_config['weight_decay'])
        optimizer_GCN = torch.optim.AdamW(self.GCN_model.parameters(), lr=self.model_config['lr'],
                                          weight_decay=self.model_config['weight_decay'])

        for e in range(self.train_config['epochs']):
            for didx, train_data in enumerate(self.data['train']):

                self.GCN_model.train()
                self.model.train()
                train_graph = self.data['train'][didx].graph.to(self.device)
                residual_embed, loss_code, quantized, codebook = self.model(train_graph, train_data)

                loss = self.model.cross_attn.get_train_loss(residual_embed, quantized, codebook,
                                                            train_graph.ano_labels,
                                                            self.model_config['num_prompt'])

                train_graph = self.data['train'][didx].graph.to(self.device)
                dgl_cut_graph_gpu = train_graph.dgl_cut_graph.to(self.device)
                node_emb_cut = self.GCN_model(dgl_cut_graph_gpu, train_graph.x)
                cut_adj = dgl_cut_graph_gpu.adj().to_dense().to(self.device)
                loss_cut, message_sum_2 = max_message(node_emb_cut, cut_adj)

                loss += loss_code.squeeze()
                optimizer.zero_grad()
                optimizer_GCN.zero_grad()
                loss.backward()
                loss_cut.backward()
                optimizer.step()
                optimizer_GCN.step()

                if e % 20 == 0:
                    print(f"current epoch {e}")

                if e == self.train_config['epochs'] - 1:
                    output_dir = Path.cwd().joinpath(
                        "output",
                        f"{train_data.name}"
                    )

                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    codebook = codebook.detach().cpu().numpy()
                    quantized = quantized.detach().cpu().numpy()
                    np.savez(output_dir.joinpath("codebook"), codebook)
                    np.savez(output_dir.joinpath("node_emb_with_I_emebding"), quantized)
                    message_list = []
                    message_list.append(torch.unsqueeze(message_sum_2, 0))
                    message_list = torch.mean(torch.cat(message_list), 0)

                    message = np.array(message_list.cpu().detach())
                    final_message = 1 - normalize_score(message)
                    auc = roc_auc_score(train_graph.ano_labels.cpu().numpy(), final_message)
                    print(f"affinity score {train_data.name} AUROC : {auc}")

        print('Finish Training for {} epochs!'.format(self.train_config['epochs']))

        # Evaluation
        test_score_list = {}
        self.model.eval()
        self.GCN_model.eval()

        codebook_sum = None
        codebook_list = []
        for didx, train_data in enumerate(self.data['train']):
            output_dir = Path.cwd().joinpath(
                "output",
                f"{train_data.name}"
            )

            codebook_1 = load_out_t(output_dir, 'codebook.npz')
            if codebook_sum is None:
                codebook_sum = codebook_1.clone()
            else:
                codebook_sum += codebook_1  # 累加
            codebook_list.append(codebook_1)

        # codebook_sum = codebook_sum / 4
        # codebook_sum = codebook_sum.to(self.train_config['device'])

        final_codebook = torch.cat(codebook_list, dim=0).to(self.train_config['device'])
        end_time = time.time()
        print(f"训练总时间：{end_time - start_time:.2f} 秒")
        start_time = time.time()
        for didx, test_data in enumerate(self.data['test']):

            test_graph = test_data.graph.to(self.train_config['device'])
            labels = test_graph.ano_labels
            shot_mask = test_graph.shot_mask.bool()

            query_labels = labels[~shot_mask].to(self.train_config['device'])
            # residual_embed = self.model(test_graph)

            residual_embed, _, quantized, codebook = self.model(test_graph, test_data)
            output_dir = Path.cwd().joinpath(
                "output",
                f"{test_data.name}"
            )

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            query_scores = self.model.cross_attn.get_test_score(residual_embed, final_codebook,
                                                                test_graph.shot_mask,
                                                                test_graph.ano_labels)

            dgl_cut_graph_gpu = test_graph.dgl_cut_graph.to(self.device)
            node_emb_cut = self.GCN_model(dgl_cut_graph_gpu, test_graph.x)
            cut_adj = dgl_cut_graph_gpu.adj().to_dense().to(self.device)

            _, message_sum_2 = max_message(node_emb_cut, cut_adj)

            message_list = []
            message_list.append(torch.unsqueeze(message_sum_2, 0))

            message_list = torch.mean(torch.cat(message_list), 0)

            message = np.array(message_list.cpu().detach())
            final_message = 1 - normalize_score(message)
            auc = roc_auc_score(labels.cpu().numpy(), final_message)

            print(f"affinity score {test_data.name}  AUROC : {auc}")
            final_message = torch.FloatTensor(final_message).to(self.device)
            lamda = test_graph.l
            print(f"{test_data.name}:{lamda}")

            query_scores = (1 - lamda) * query_scores + lamda * final_message[~shot_mask]

            test_score = test_eval(query_labels, query_scores)

            test_data_name = self.train_config['testdsets'][didx]
            test_score_list[test_data_name] = {
                'AUROC': test_score['AUROC'],
                'AUPRC': test_score['AUPRC'],
            }
        end_time = time.time()
        print(f"测试总时间：{end_time - start_time:.2f} 秒")

        # 获取显存峰值（单位：MB）
        max_memory_MB = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        # peak_gpu_memory_by_torch = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024

        print(f"训练期间GPU最大显存占用：{max_memory_MB:.2f} GB")
        return test_score_list
