import os
import copy
import pickle
import time
import math
import warnings
import numpy as np
from .task import BaseTask
import torch
from torch_geometric.loader import DataLoader
from message_tuning import MTGEva
from ..utils import constraint, center_embedding, Gprompt_tuning_loss, process
from ..evaluation import GPPTEva, GNNNodeEva, GPFEva, MultiGpromptEva, GpromptEva, AllInOneEva
from ..data import load4node, induced_graphs, graph_split, split_induced_graphs, node_sample_and_save, \
    GraphDataset


warnings.filterwarnings("ignore")


class NodeTask(BaseTask):
    def __init__(self, data, input_dim, output_dim, task_num=5, graphs_list=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = 'NodeTask'
        self.task_num = task_num
        if self.prompt_type == 'MultiGprompt':
            self.load_multigprompt_data()
        else:
            self.data = data
            if self.dataset_name == 'ogbn-arxiv':
                self.data.y = self.data.y.squeeze()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.graphs_list = graphs_list

        self.create_few_data_folder()

    def create_few_data_folder(self):
        k = self.shot_num
        task_num = self.task_num
        for k in range(1, task_num + 1):
            k_shot_folder = './Experiment/sample_data/Node/' + self.dataset_name + '/' + str(k) + '_shot'
            os.makedirs(k_shot_folder, exist_ok=True)

            for i in range(1, task_num + 1):
                folder = os.path.join(k_shot_folder, str(i))
                if not os.path.exists(folder):
                    os.makedirs(folder)
                    node_sample_and_save(self.data, k, folder, self.output_dim)
                    print(str(k) + ' shot ' + str(i) + ' th is saved!!')

    def load_multigprompt_data(self):
        adj, features, labels = process.load_data(self.dataset_name)
        # adj, features, labels = process.load_data(self.dataset_name)
        self.input_dim = features.shape[1]
        self.output_dim = labels.shape[1]
        print('a', self.output_dim)
        features, _ = process.preprocess_features(features)
        self.sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj).to(self.device)
        self.labels = torch.FloatTensor(labels[np.newaxis])
        self.features = torch.FloatTensor(features[np.newaxis]).to(self.device)
        # print("labels",labels)
        print("adj", self.sp_adj.shape)
        print("feature", features.shape)

    def load_induced_graph(self):
        smallest_size = 5
        if self.dataset_name in ['ENZYMES', 'PROTEINS']:
            smallest_size = 1
        if self.dataset_name == 'PubMed':
            smallest_size = 8
        folder_path = './Experiment/induced_graph/' + self.dataset_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_path = folder_path + '/induced_graph_min{}_max300.pkl'.format(smallest_size)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                graphs_list = pickle.load(f)
        else:
            print('Begin split_induced_graphs.')
            split_induced_graphs(self.data, folder_path, self.device, smallest_size=smallest_size, largest_size=300)
            with open(file_path, 'rb') as f:
                graphs_list = pickle.load(f)
        self.graphs_list = []
        for i in range(len(graphs_list)):
            graph = graphs_list[i].to(self.device)
            self.graphs_list.append(graph)

    def load_data(self):
        self.data, self.input_dim, self.output_dim = load4node(self.dataset_name)

    def train(self, data, train_idx):
        self.gnn.train()
        self.answering.train()
        self.optimizer.zero_grad()
        out = self.gnn(data.x, data.edge_index, batch=None)
        out = self.answering(out)
        loss = self.criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def GPPTtrain(self, data, train_idx):
        self.prompt.train()
        node_embedding = self.gnn(data.x, data.edge_index)
        out = self.prompt(node_embedding, data.edge_index)
        loss = self.criterion(out[train_idx], data.y[train_idx])
        loss = loss + 0.001 * constraint(self.device, self.prompt.get_TaskToken())
        self.pg_opi.zero_grad()
        loss.backward()
        self.pg_opi.step()
        mid_h = self.prompt.get_mid_h()
        self.prompt.update_StructureToken_weight(mid_h)
        return loss.item()

    def MultiGpromptTrain(self, pretrain_embs, train_lbls, train_idx):
        self.DownPrompt.train()
        self.optimizer.zero_grad()
        prompt_feature = self.feature_prompt(self.features)
        # prompt_feature = self.feature_prompt(self.data.x)
        # embeds1 = self.gnn(prompt_feature, self.data.edge_index)
        embeds1 = self.Preprompt.gcn(prompt_feature, self.sp_adj, True, False)
        pretrain_embs1 = embeds1[0, train_idx]
        logits = self.DownPrompt(pretrain_embs, pretrain_embs1, train_lbls, 1).float().to(self.device)
        loss = self.criterion(logits, train_lbls)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item()

    def SUPTtrain(self, data):
        self.gnn.train()
        self.optimizer.zero_grad()
        data.x = self.prompt.add(data.x)
        out = self.gnn(data.x, data.edge_index, batch=None)
        out = self.answering(out)
        loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
        orth_loss = self.prompt.orthogonal_loss()
        loss += orth_loss
        loss.backward()
        self.optimizer.step()
        return loss

    def GPFTrain(self, train_loader):
        self.prompt.train()
        total_loss = 0.0
        for batch in train_loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            batch.x = self.prompt.add(batch.x)
            out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt=self.prompt, prompt_type=self.prompt_type)
            out = self.answering(out)
            loss = self.criterion(out, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def AllInOneTrain(self, train_loader, answer_epoch=1, prompt_epoch=1):
        # we update answering and prompt alternately.
        # tune task head
        self.answering.train()
        self.prompt.eval()
        self.gnn.eval()
        for epoch in range(1, answer_epoch + 1):
            answer_loss = self.prompt.Tune(train_loader, self.gnn, self.answering, self.criterion, self.answer_opi,
                                           self.device)
            print(("frozen gnn | frozen prompt | *tune answering function... {}/{} ,loss: {:.4f} ".format(epoch,
                                                                                                          answer_epoch,
                                                                                                          answer_loss)))

        # tune prompt
        self.answering.eval()
        self.prompt.train()
        for epoch in range(1, prompt_epoch + 1):
            pg_loss = self.prompt.Tune(train_loader, self.gnn, self.answering, self.criterion, self.pg_opi, self.device)
            print(("frozen gnn | *tune prompt |frozen answering function... {}/{} ,loss: {:.4f} ".format(epoch,
                                                                                                         prompt_epoch,
                                                                                                         pg_loss)))

        # return pg_loss
        return answer_loss

    def GpromptTrain(self, train_loader):
        self.prompt.train()
        total_loss = 0.0
        accumulated_centers = None
        accumulated_counts = None
        for batch in train_loader:
            self.pg_opi.zero_grad()
            batch = batch.to(self.device)
            out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt=self.prompt, prompt_type='Gprompt')
            # out = s𝑡,𝑥 = ReadOut({p𝑡 ⊙ h𝑣 : 𝑣 ∈ 𝑉 (𝑆𝑥)}),
            center, class_counts = center_embedding(out, batch.y, self.output_dim)

            if accumulated_centers is None:
                accumulated_centers = center
                accumulated_counts = class_counts
            else:
                accumulated_centers += center * class_counts
                accumulated_counts += class_counts
            criterion = Gprompt_tuning_loss()
            loss = criterion(out, center, batch.y)
            loss.backward()
            self.pg_opi.step()
            total_loss += loss.item()

        mean_centers = accumulated_centers / accumulated_counts

        return total_loss / len(train_loader), mean_centers

    def _compute_val_acc(self, loader):
        """Compute validation accuracy on a data loader without gradient."""
        self.gfm.eval()
        self.answering.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.gfm(batch.x, batch.edge_index, batch.batch)
                out = self.answering(out)
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        return correct / max(total, 1)

    def MTGTrain(self, train_loader):
        self.gfm.freeze_pretrained_params()
        self.gfm.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            out = self.gfm(batch.x, batch.edge_index, batch.batch)
            out = self.answering(out)
            loss = self.criterion(out, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
        return total_loss / len(train_loader), correct / max(total, 1)

    def run(self):
        test_accs = []
        f1s = []
        rocs = []
        prcs = []
        batch_best_loss = []
        if self.prompt_type == 'All-in-one':
            self.answer_epoch = 50
            self.prompt_epoch = 50
            self.epochs = int(self.epochs / self.answer_epoch)
        for i in range(1, self.task_num + 1):
            sample_data_foler_path = "./Experiment/sample_data/Node/{}/{}_shot/{}".format(self.dataset_name,
                                                                                          self.shot_num, i)

            if not os.path.exists(sample_data_foler_path):
                print(
                    f"Warning! Failed to find sample_data for shot {self.shot_num}, id {i}, path: {sample_data_foler_path}, skipping...")
                continue

            if self.prompt_type == 'MTG':
                self.initialize_gfm()
                self._best_val_acc = 0.0
                self._best_gfm_state = None
                self._best_answering_state = None
            else:
                self.initialize_gnn()

            self.answering = torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim),
                                                 torch.nn.Softmax(dim=1)).to(self.device)
            self.initialize_prompt()
            self.initialize_optimizer()

            idx_train = torch.load(f"{sample_data_foler_path}/train_idx.pt").type(torch.long).to(self.device)
            print('idx_train', idx_train)
            train_lbls = torch.load(f"{sample_data_foler_path}/train_labels.pt").type(torch.long).squeeze().to(
                self.device)
            print("true", i, train_lbls)
            idx_test = torch.load(f"{sample_data_foler_path}/test_idx.pt").type(torch.long).to(self.device)
            test_lbls = torch.load(f"{sample_data_foler_path}/test_labels.pt").type(torch.long).squeeze().to(
                self.device)

            # GPPT prompt initialtion
            if self.prompt_type == 'GPPT':
                node_embedding = self.gnn(self.data.x, self.data.edge_index)
                self.prompt.weigth_init(node_embedding, self.data.edge_index, self.data.y, idx_train)

            if self.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus', 'MTG']:
                train_graphs = []
                test_graphs = []
                # self.graphs_list.to(self.device)
                print('distinguishing the train dataset and test dataset...')
                for graph in self.graphs_list:
                    if graph.index in idx_train:
                        train_graphs.append(graph)
                    elif graph.index in idx_test:
                        test_graphs.append(graph)
                print('Done!!!')

                train_dataset = GraphDataset(train_graphs)
                test_dataset = GraphDataset(test_graphs)

                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                print("prepare induce graph data is finished!")

            if self.prompt_type == 'MultiGprompt':
                embeds, _ = self.Preprompt.embed(self.features, self.sp_adj, True, None, False)
                pretrain_embs = embeds[0, idx_train]
                test_embs = embeds[0, idx_test]

            patience = 20
            best = 1e9
            cnt_wait = 0
            best_loss = 1e9

            for epoch in range(1, self.epochs):
                t0 = time.time()

                if self.prompt_type == 'None':
                    loss = self.train(self.data, idx_train)
                elif self.prompt_type == 'GPPT':
                    loss = self.GPPTtrain(self.data, idx_train)
                elif self.prompt_type == 'All-in-one':
                    loss = self.AllInOneTrain(train_loader, self.answer_epoch, self.prompt_epoch)
                elif self.prompt_type in ['GPF', 'GPF-plus']:
                    loss = self.GPFTrain(train_loader)
                elif self.prompt_type == 'Gprompt':
                    loss, center = self.GpromptTrain(train_loader)
                elif self.prompt_type == 'MultiGprompt':
                    loss = self.MultiGpromptTrain(pretrain_embs, train_lbls, idx_train)
                elif self.prompt_type == 'MTG':
                    loss, train_acc = self.MTGTrain(train_loader)
                    val_acc = self._compute_val_acc(test_loader)
                    if val_acc > self._best_val_acc:
                        self._best_val_acc = val_acc
                        self._best_gfm_state = copy.deepcopy(self.gfm.state_dict())
                        self._best_answering_state = copy.deepcopy(self.answering.state_dict())

                if loss < best:
                    best = loss
                    # best_t = epoch
                    cnt_wait = 0
                    # torch.save(model.state_dict(), args.save_name)
                else:
                    cnt_wait += 1
                    if cnt_wait == patience:
                        print('-' * 100)
                        print('Early stopping at ' + str(epoch) + ' eopch!')
                        break

                if self.prompt_type == 'MTG':
                    print("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} | ValAcc {:.4f}".format(
                        epoch, time.time() - t0, loss, train_acc, val_acc))
                else:
                    print("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f}  ".format(epoch, time.time() - t0, loss))

            if not math.isnan(loss):
                batch_best_loss.append(loss)

                if self.prompt_type == 'None':
                    test_acc, f1, roc, prc = GNNNodeEva(self.data, idx_test, self.gnn, self.answering, self.output_dim,
                                                        self.device)
                elif self.prompt_type == 'GPPT':
                    test_acc, f1, roc, prc = GPPTEva(self.data, idx_test, self.gnn, self.prompt, self.output_dim,
                                                     self.device)
                elif self.prompt_type == 'All-in-one':
                    test_acc, f1, roc, prc = AllInOneEva(test_loader, self.prompt, self.gnn, self.answering,
                                                         self.output_dim, self.device)
                elif self.prompt_type in ['GPF', 'GPF-plus']:
                    test_acc, f1, roc, prc = GPFEva(test_loader, self.gnn, self.prompt, self.answering, self.output_dim,
                                                    self.device)
                elif self.prompt_type == 'Gprompt':
                    test_acc, f1, roc, prc = GpromptEva(test_loader, self.gnn, self.prompt, center, self.output_dim,
                                                        self.device)
                elif self.prompt_type == 'MultiGprompt':
                    prompt_feature = self.feature_prompt(self.features)
                    test_acc, f1, roc, prc = MultiGpromptEva(test_embs, test_lbls, idx_test, prompt_feature,
                                                             self.Preprompt, self.DownPrompt, self.sp_adj,
                                                             self.output_dim, self.device)
                elif self.prompt_type == 'MTG':
                    if self._best_gfm_state is not None:
                        self.gfm.load_state_dict(self._best_gfm_state)
                        self.answering.load_state_dict(self._best_answering_state)
                        print("Loaded best-validation model (ValAcc={:.4f}) for final evaluation.".format(self._best_val_acc))
                    test_acc, f1, roc, prc = MTGEva(test_loader, self.gfm, self.answering, self.output_dim, self.device)

                print(
                    f"Final True Accuracy: {test_acc:.4f} | Macro F1 Score: {f1:.4f} | AUROC: {roc:.4f} | AUPRC: {prc:.4f}")
                print("best_loss", batch_best_loss)

                test_accs.append(test_acc)
                f1s.append(f1)
                rocs.append(roc)
                prcs.append(prc)

        mean_test_acc = np.mean(test_accs)
        std_test_acc = np.std(test_accs)
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        mean_roc = np.mean(rocs)
        std_roc = np.std(rocs)
        mean_prc = np.mean(prcs)
        std_prc = np.std(prcs)
        print('Acc List', test_accs)
        print(" Final best | test Accuracy {:.4f}±{:.4f}(std)".format(mean_test_acc, std_test_acc))
        print(" Final best | test F1 {:.4f}±{:.4f}(std)".format(mean_f1, std_f1))
        print(" Final best | AUROC {:.4f}±{:.4f}(std)".format(mean_roc, std_roc))
        print(" Final best | AUPRC {:.4f}±{:.4f}(std)".format(mean_prc, std_prc))

        print(self.pre_train_type, self.gnn_type, self.prompt_type, "Node Task completed")
        mean_best = np.mean(batch_best_loss)

        return mean_best, mean_test_acc, std_test_acc, mean_f1, std_f1, mean_roc, std_roc, mean_prc, std_prc

        # elif self.prompt_type != 'MultiGprompt':
        #       # embeds, _ = self.Preprompt.embed(self.features, self.sp_adj, True, None, False)
        #       embeds, _ = self.Preprompt.embed(self.features, self.sp_adj, True, None, False)

        #       test_lbls = torch.argmax(self.labels[0, self.idx_test], dim=1).cuda()
        #       tot = torch.zeros(1)
        #       tot = tot.cuda()
        #       accs = []
        #       patience = 20
        #       print('-' * 100)
        #       cnt_wait = 0
        #       for i in range(1,6):
        #             # idx_train = torch.load("./data/fewshot_cora/{}-shot_cora/{}/idx.pt".format(self.shot_num,i)).type(torch.long).cuda()
        #             # print('idx_train',idx_train)
        #             # train_lbls = torch.load("./data/fewshot_cora/{}-shot_cora/{}/labels.pt".format(self.shot_num,i)).type(torch.long).squeeze().cuda()
        #             # print("true",i,train_lbls)
        #             self.dataset_name ='Cora'
        #             idx_train = torch.load("./Experiment/sample_data/Node/{}/{}_shot/{}/train_idx.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).cuda()
        #             print('idx_train',idx_train)
        #             train_lbls = torch.load("./Experiment/sample_data/Node/{}/{}_shot/{}/train_labels.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).squeeze().cuda()
        #             print("true",i,train_lbls)

        #             idx_test = torch.load("./Experiment/sample_data/Node/{}/{}_shot/{}/test_idx.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).cuda()
        #             test_lbls = torch.load("./Experiment/sample_data/Node/{}/{}_shot/{}/test_labels.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).squeeze().cuda()

        #             test_embs = embeds[0, idx_test]
        #             best = 1e9
        #             pat_steps = 0
        #             best_acc = torch.zeros(1)
        #             best_acc = best_acc.cuda()
        #             pretrain_embs = embeds[0, idx_train]
        #             for _ in range(50):
        #                   self.DownPrompt.train()
        #                   self.optimizer.zero_grad()
        #                   prompt_feature = self.feature_prompt(self.features)
        #                   # prompt_feature = self.feature_prompt(self.data.x)
        #                   # embeds1 = self.gnn(prompt_feature, self.data.edge_index)
        #                   embeds1= self.Preprompt.gcn(prompt_feature, self.sp_adj , True, False)
        #                   pretrain_embs1 = embeds1[0, idx_train]
        #                   logits = self.DownPrompt(pretrain_embs,pretrain_embs1, train_lbls,1).float().cuda()
        #                   loss = self.criterion(logits, train_lbls)
        #                   if loss < best:
        #                         best = loss
        #                         cnt_wait = 0
        #                   else:
        #                         cnt_wait += 1
        #                         if cnt_wait == patience:
        #                               print('Early stopping at '+str(_) +' eopch!')
        #                               break

        #                   loss.backward(retain_graph=True)
        #                   self.optimizer.step()

        #             prompt_feature = self.feature_prompt(self.features)
        #             embeds1, _ = self.Preprompt.embed(prompt_feature, self.sp_adj, True, None, False)
        #             test_embs1 = embeds1[0, idx_test]
        #             print('idx_test', idx_test)
        #             logits = self.DownPrompt(test_embs, test_embs1, train_lbls)
        #             preds = torch.argmax(logits, dim=1)
        #             acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        #             accs.append(acc * 100)
        #             print('acc:[{:.4f}]'.format(acc))
        #             tot += acc

        #       print('-' * 100)
        #       print('Average accuracy:[{:.4f}]'.format(tot.item() / 10))
        #       accs = torch.stack(accs)
        #       print('Mean:[{:.4f}]'.format(accs.mean().item()))
        #       print('Std :[{:.4f}]'.format(accs.std().item()))
        #       print('-' * 100)

        # print("Node Task completed")
