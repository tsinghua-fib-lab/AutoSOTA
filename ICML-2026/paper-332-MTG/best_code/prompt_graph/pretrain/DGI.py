import os
import copy
import time
import numpy as np
from ..defines import GRAPH_TASKS, NODE_TASKS
from .base import PreTrain

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric import loader
from ..data import load4node, load4graph, NodePretrain


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 0)
        c_x = c_x.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class DGI(PreTrain):
    def __init__(self, *args, **kwargs):  # hid_dim=16
        super().__init__(*args, **kwargs)

        self.disc = Discriminator(self.hid_dim).to(self.device)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.graph_data = self.load_data()
        self.initialize_gnn(self.input_dim, self.hid_dim)
        self.optimizer = Adam(self.gnn.parameters(), lr=0.001, weight_decay=0.0)

    # def load_data(self):
    #     if self.dataset_name in ['PubMed', 'CiteSeer', 'Cora','Computers', 'Photo', 'Reddit', 'WikiCS', 'Flickr', 'ogbn-arxiv']:
    #         data, input_dim, _ = load4node(self.dataset_name)
    #         self.input_dim = input_dim
    #     elif self.dataset_name in GRAPH_TASKS:
    #         input_dim, _, graph_list= load4graph(self.dataset_name,pretrained=True) # need graph list not dataset object, so the pretrained = True
    #         self.input_dim = input_dim
    #         graph_data_batch = Batch.from_data_list(graph_list)
    #         data= Data(x=graph_data_batch.x, edge_index=graph_data_batch.edge_index)

    #         if self.dataset_name in  ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY']:
    #             from torch_geometric import loader
    #             self.batch_dataloader = loader.DataLoader(graph_list,batch_size=512,shuffle=False)

    #     return data

    # def pretrain_one_epoch(self):
    #     self.gnn.train()
    #     self.optimizer.zero_grad()
    #     device = self.device

    #     graph_original = self.graph_data
    #     graph_corrupted = copy.deepcopy(graph_original)
    #     idx_perm = np.random.permutation(graph_original.x.size(0))
    #     graph_corrupted.x = graph_original.x[idx_perm].to(self.device)

    #     graph_original.to(device)
    #     graph_corrupted.to(device)

    #     pos_z = self.gnn(graph_original.x, graph_original.edge_index)
    #     neg_z = self.gnn(graph_corrupted.x, graph_corrupted.edge_index)

    #     s = torch.sigmoid(torch.mean(pos_z, dim=0)).to(device)

    #     logits = self.disc(s, pos_z, neg_z)

    #     lbl_1 = torch.ones((pos_z.shape[0], 1))
    #     lbl_2 = torch.zeros((neg_z.shape[0], 1))
    #     lbl = torch.cat((lbl_1, lbl_2), 1).to(device)

    #     loss = self.loss_fn(logits, lbl)
    #     loss.backward()
    #     self.optimizer.step()

    #     accum_loss = float(loss.detach().cpu().item())
    #     return accum_loss

    def load_data(self):
        if self.dataset_name in NODE_TASKS:
            data, input_dim, _ = load4node(self.dataset_name)
            self.input_dim = input_dim
        elif self.dataset_name in GRAPH_TASKS:
            input_dim, _, graph_list = load4graph(self.dataset_name,
                                                  pretrained=True)  # need graph list not dataset object, so the pretrained = True
            self.input_dim = input_dim

            self.batch_dataloader = loader.DataLoader(graph_list, batch_size=512, shuffle=False,
                                                      num_workers=self.num_workers)

            data = graph_list

        return data

    def pretrain_one_epoch(self):
        self.gnn.train()
        self.optimizer.zero_grad()
        device = self.device

        if self.dataset_name in NODE_TASKS:
            graph_original = self.graph_data
            graph_corrupted = copy.deepcopy(graph_original)
            idx_perm = np.random.permutation(graph_original.x.size(0))
            graph_corrupted.x = graph_original.x[idx_perm].to(self.device)

            graph_original.to(device)
            graph_corrupted.to(device)

            pos_z = self.gnn(graph_original.x, graph_original.edge_index)
            neg_z = self.gnn(graph_corrupted.x, graph_corrupted.edge_index)

            s = torch.sigmoid(torch.mean(pos_z, dim=0)).to(device)

            logits = self.disc(s, pos_z, neg_z)

            lbl_1 = torch.ones((pos_z.shape[0], 1))
            lbl_2 = torch.zeros((neg_z.shape[0], 1))
            lbl = torch.cat((lbl_1, lbl_2), 1).to(device)

            loss = self.loss_fn(logits, lbl)
            loss.backward()
            self.optimizer.step()

            accum_loss = float(loss.detach().cpu().item())
        elif self.dataset_name in GRAPH_TASKS:
            accum_loss = torch.tensor(0.0)
            for batch_id, batch_graph in enumerate(self.batch_dataloader):
                graph_original = batch_graph.to(device)
                graph_corrupted = copy.deepcopy(graph_original)
                idx_perm = np.random.permutation(graph_original.x.size(0))
                graph_corrupted.x = graph_original.x[idx_perm].to(self.device)

                graph_original.to(device)
                graph_corrupted.to(device)

                pos_z = self.gnn(graph_original.x, graph_original.edge_index)
                neg_z = self.gnn(graph_corrupted.x, graph_corrupted.edge_index)

                s = torch.sigmoid(torch.mean(pos_z, dim=0)).to(device)

                logits = self.disc(s, pos_z, neg_z)

                lbl_1 = torch.ones((pos_z.shape[0], 1))
                lbl_2 = torch.zeros((neg_z.shape[0], 1))
                lbl = torch.cat((lbl_1, lbl_2), 1).to(device)

                loss = self.loss_fn(logits, lbl)
                loss.backward()
                self.optimizer.step()

                accum_loss += float(loss.detach().cpu().item())

            accum_loss = accum_loss / (batch_id + 1)

        return accum_loss

    def pretrain(self):
        train_loss_min = 1000000
        patience = 20
        cnt_wait = 0

        for epoch in range(1, self.epochs + 1):
            st_time = time.time()
            train_loss = self.pretrain_one_epoch()

            print(f"DGI [Pretrain] Epoch {epoch}/{self.epochs} | Train Loss {train_loss:.5f} | "
                  f"Cost Time {time.time() - st_time:.3}s")

            if train_loss_min > train_loss:
                train_loss_min = train_loss
                cnt_wait = 0
            else:
                cnt_wait += 1
                if cnt_wait == patience:
                    print('-' * 100)
                    print('Early stopping at ' + str(epoch) + ' epoch!')
                    break

        folder_path = f"./Experiment/pre_trained_model/{self.dataset_name}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(self.gnn.state_dict(),
                   "{}/{}.{}.{}.pth".format(folder_path, 'DGI', self.gnn_type, str(self.hid_dim) + 'hidden_dim'))
        print("+++model saved ! {}/{}.{}.{}.pth".format(self.dataset_name, 'DGI', self.gnn_type,
                                                        str(self.hid_dim) + 'hidden_dim'))
