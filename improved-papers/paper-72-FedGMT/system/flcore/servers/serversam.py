import time
from flcore.clients.clientsam import clientSAM
from flcore.servers.serverbase import Server
from tqdm import tqdm
import torch
import os
class FedSAM(Server):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(args, clientSAM)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")


    def train(self):
        for epoch in tqdm(range(1, self.global_rounds+1), desc='server-training'):


            s_t = time.time()
            self.selected_clients = self.select_clients()

            self.send_models(self.selected_clients,self.global_model)
            # loss_before,_ = self.sharpness()
            print()
            self.client_updates = []
            for client in self.selected_clients:
                client.learning_rate = self.learning_rate
                client.train()

            self._lr_scheduler_()
            self.receive_models()
            self.aggregate_parameters()

            self.empty_cache()

            print(f"\n-------------Round number: {epoch}-------------")
            print("\nEvaluate global model")
            
            # loss_after,_ = self.sharpness()
            # self.loss_diff.append(abs(loss_before-loss_after))
            
            print('-'*25, 'This global round time cost', '-'*25, time.time() - s_t)
            self.evaluate()

            if epoch%self.save_gap == 0:
                self.save_results(epoch)
            if epoch == self.global_rounds:
                import statistics
                print(f"Avg last 50 round acc:{sum(self.test_acc[-50:])/50}, std: {statistics.stdev(self.test_acc[-50:])}")
            if self.current_acc < self.test_acc[-1]:
                self.current_acc = self.test_acc[-1]
                self.save_model()
            