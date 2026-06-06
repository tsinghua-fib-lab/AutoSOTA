import os
import time
import copy
import h5py
import numpy as np
from flcore.clients.clientpFedMe import clientpFedMe
from flcore.servers.serverbase import Server
from threading import Thread


class pFedMe(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientpFedMe)

        self.beta = args.beta
        self.rs_train_acc_per = []
        self.rs_train_loss_per = []
        self.rs_test_acc_per = []

        # 保存每个客户端的详细指标（与 Ditto 一致）
        self.rs_client_test_acc = []
        self.rs_client_test_auc = []
        self.rs_client_train_loss = []
        self.rs_client_ids = []

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            # 评估条件：按 eval_gap 间隔评估，或者最后一轮强制评估
            should_evaluate = (i % self.eval_gap == 0) or (i == self.global_rounds)
            if should_evaluate:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized model")
                self.evaluate_personalized()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.previous_global_model = copy.deepcopy(list(self.global_model.parameters()))
            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()
            self.beta_aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc_per], top_cnt=self.top_cnt):
                break

        # print("\nBest global accuracy.")
        # # self.print_(max(self.rs_test_acc), max(
        # #     self.rs_train_acc), min(self.rs_train_loss))
        # print(max(self.rs_test_acc))

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc_per), max(
        #     self.rs_train_acc_per), min(self.rs_train_loss_per))
        print(max(self.rs_test_acc_per))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))


        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientpFedMe)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def beta_aggregate_parameters(self):
        # aggregate avergage model with previous model using parameter beta
        for pre_param, param in zip(self.previous_global_model, self.global_model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data

    def test_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics_personalized()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.clients:
            ct, cl, ns = c.train_metrics_personalized()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def evaluate_personalized(self):
        stats = self.test_metrics_personalized()
        stats_train = self.train_metrics_personalized()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_acc = sum(stats_train[2])*1.0 / sum(stats_train[1])
        train_loss = sum(stats_train[3])*1.0 / sum(stats_train[1])

        # 计算每个客户端的详细指标
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        losses = [l / n for l, n in zip(stats_train[3], stats_train[1])]

        # 保存全局平均指标（个性化变量，用于 auto_break）
        self.rs_test_acc_per.append(test_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)

        # 同时保存到标准变量名（让 base 的 save_results 可以使用）
        self.rs_test_acc.append(test_acc)
        self.rs_test_auc.append(test_auc)
        self.rs_train_loss.append(train_loss)

        # 保存每个客户端的详细指标（与 Ditto 一致）
        self.rs_client_test_acc.append(accs)
        self.rs_client_test_auc.append(aucs)
        self.rs_client_train_loss.append(losses)
        self.rs_client_ids.append(stats[0])  # 客户端IDs

        # 打印详细统计信息（与 Ditto 一致）
        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def save_results(self):
        """
        保存 pFedMe 结果（继承 base 并添加 train_acc）
        """
        # 调用父类的 save_results 保存标准指标和客户端详细指标
        super().save_results()

        # 额外保存 train_acc（父类不保存这个）
        if len(self.rs_train_acc_per) > 0:
            # 重新打开同一个文件，追加 train_acc 数据
            if "/" in self.dataset:
                dataset_name, config_name = self.dataset.split("/", 1)
                result_path = os.path.join(self.result_dir, dataset_name, config_name)
            else:
                result_path = os.path.join(self.result_dir, self.dataset)

            algo = self.algorithm + "_" + self.goal + "_" + str(self.times)
            file_path = os.path.join(result_path, "{}.h5".format(algo))

            with h5py.File(file_path, 'a') as hf:  # 'a' 模式：追加
                if 'rs_train_acc' not in hf:
                    hf.create_dataset('rs_train_acc', data=self.rs_train_acc_per)
