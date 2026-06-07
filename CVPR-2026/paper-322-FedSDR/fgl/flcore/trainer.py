import torch
import random
from fgl.data.distributed_dataset_loader import FGLDataset
from fgl.utils.basic_utils import load_client, load_server
from fgl.utils.logger import Logger


class FGLTrainer:
    def __init__(self, args):
        self.args = args
        self.message_pool = {}
        self.device = torch.device(f"cuda:{args.gpuid}" if (
            torch.cuda.is_available() and args.use_cuda) else "cpu")
        fgl_dataset = FGLDataset(args)
        self.clients = [load_client(args, client_id, fgl_dataset.local_data[client_id], fgl_dataset.processed_dir,
                                    self.message_pool, self.device) for client_id in range(self.args.num_clients)]
        self.server = load_server(args, fgl_dataset.global_data,
                                  fgl_dataset.processed_dir, self.message_pool, self.device)

        self.evaluation_result = {"best_round": 0}
        self.evaluation_result[f"best_val_accuracy"] = 0
        self.evaluation_result[f"best_test_accuracy"] = 0

        self.logger = Logger(args, self.message_pool,
                             fgl_dataset.processed_dir, self.server.personalized)

    def train(self):
        for round_id in range(self.args.num_rounds):
            sampled_clients = sorted(random.sample(list(range(self.args.num_clients)), int(
                self.args.num_clients * self.args.client_frac)))
            print(f"round # {round_id}\t\tsampled_clients: {sampled_clients}")
            self.message_pool["round"] = round_id
            self.message_pool["sampled_clients"] = sampled_clients
            self.server.send_message()

            for client_id in sampled_clients:
                self.clients[client_id].execute(round_id)
                self.clients[client_id].send_message()

            if (round_id + 1) % 10 == 0:
                self.server.execute()
            print(f"Aggregation performed at round #{round_id + 1}")

            self.evaluate()
            print("-" * 50)

        self.logger.save()

    def evaluate(self):
        evaluation_result = {"current_round": self.message_pool["round"]}

        evaluation_result[f"current_val_accuracy"] = 0
        evaluation_result[f"current_test_accuracy"] = 0

        tot_samples = 0
        one_time_infer = False

        for client_id in range(self.args.num_clients):
            num_samples = self.clients[client_id].task.num_samples
            result = self.clients[client_id].task.evaluate()

            val_metric, test_metric = result[f"accuracy_val"], result[f"accuracy_test"]
            evaluation_result[f"current_val_accuracy"] += val_metric * num_samples
            evaluation_result[f"current_test_accuracy"] += test_metric * num_samples

            if one_time_infer:
                tot_samples = num_samples
                break
            else:
                tot_samples += num_samples

        evaluation_result[f"current_val_accuracy"] /= tot_samples
        evaluation_result[f"current_test_accuracy"] /= tot_samples

        if evaluation_result[f"current_val_accuracy"] > self.evaluation_result[f"best_val_accuracy"]:
            self.evaluation_result[f"best_val_accuracy"] = evaluation_result[f"current_val_accuracy"]
            self.evaluation_result[f"best_test_accuracy"] = evaluation_result[f"current_test_accuracy"]
            self.evaluation_result[f"best_round"] = evaluation_result[f"current_round"]

        current_output = f"curr_round: {evaluation_result['current_round']}\t" + \
            "\t".join(
                [f"curr_val_accuracy: {evaluation_result[f'current_val_accuracy']:.4f}\tcurr_test_accuracy: {evaluation_result[f'current_test_accuracy']:.4f}"])

        best_output = f"best_round: {self.evaluation_result['best_round']}\t" + \
            "\t".join(
                [f"best_val_accuracy: {self.evaluation_result[f'best_val_accuracy']:.4f}\tbest_test_accuracy: {self.evaluation_result[f'best_test_accuracy']:.4f}"])

        print(current_output)
        print(best_output)

        self.logger.add_log(evaluation_result)

        return evaluation_result
