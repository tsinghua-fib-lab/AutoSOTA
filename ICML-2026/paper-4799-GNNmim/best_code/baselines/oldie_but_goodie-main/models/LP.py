from embedder import embedder
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import datetime
import utils
from tqdm import trange
import pickle
from torch_geometric.nn.models import LabelPropagation


class LP():
    def __init__(self, args):
        self.args = args
    
    def training(self):
        file = utils.set_filename(self.args)
        logger = utils.setup_logger('./', '-', file)

        seed_result = {}
        seed_result['acc'] = []
        seed_result['macro_F'] = []
        
        for seed in trange(0, 0+self.args.n_runs):
            print(f'============== seed:{seed} ==============')
            utils.seed_everything(seed)
            print('seed:', seed, file)
            self.args.seed = seed
            self = embedder(self.args, seed)

            # Main training
            acc_vals = []
            test_results = []
            best_metric = 0

            model = modeler(self.args).to(self.args.device)
            output = model(y=self.labels, edge_index=self.edge_index, mask=self.train_mask)
            
            # Debug
            # with open('lp_output.pickle', 'wb') as f:
            #     pickle.dump(output, f)
            # with open('edge_index.pickle', 'wb') as f:
            #     pickle.dump(self.edge_index, f)
            # with open('adj.pickle', 'wb') as f:
            #     pickle.dump(self.adj, f)
            # with open('labels.pickle', 'wb') as f:
            #     pickle.dump(self.labels, f)                
            # with open('train_mask.pickle', 'wb') as f:
            #     pickle.dump(self.train_mask, f)      

            # Valid
            acc_val, macro_F_val = utils.performance(output[self.val_mask], self.labels[self.val_mask], pre='valid', evaluator=self.evaluator)

            acc_vals.append(acc_val)
            max_idx = acc_vals.index(max(acc_vals))

            if best_metric <= acc_val:
                best_metric = acc_val
                best_model = deepcopy(model)

            # Test
            acc_test, macro_F_test = utils.performance(output[self.test_mask], self.labels[self.test_mask], pre='test', evaluator=self.evaluator)

            test_results.append([acc_test, macro_F_test])
            best_test_result = test_results[max_idx]

            seed_result['acc'].append(float(best_test_result[0]))
            seed_result['macro_F'].append(float(best_test_result[1]))

        acc = seed_result['acc']
        f1 = seed_result['macro_F']

        print('[Averaged result] ACC: {:.2f}+{:.2f}, Macro-F: {:.2f}+{:.2f}'.format(np.mean(acc), np.std(acc), np.mean(f1), np.std(f1)))
        print('{:.2f}+{:.2f} {:.2f}+{:.2f}'.format(np.mean(acc), np.std(acc), np.mean(f1), np.std(f1)))

        logger.info('')
        logger.info(datetime.datetime.now())
        logger.info(file)
        logger.info(f'----------- missing rate: {self.args.missing_rate} -----------')
        logger.info(f'----------- lp alpha: {self.args.lp_alpha} -----------')
        logger.info('{:.2f}+{:.2f} {:.2f}+{:.2f}'.format(np.mean(acc), np.std(acc), np.mean(f1), np.std(f1)))
        logger.info('{:.2f}+{:.2f}'.format(np.mean(acc), np.std(acc)))
        logger.info('{:.2f}+{:.2f}'.format(np.mean(f1), np.std(f1)))
        logger.info(self.args)
        logger.info(f'=================================')

        # print(self.args)


class modeler(nn.Module):
    def __init__(self, args):
        super(modeler, self).__init__()
        self.args = args
        self.model = LabelPropagation(num_layers=50, alpha=args.lp_alpha)

    def forward(self, y, edge_index, mask):
        output = self.model(y=y, edge_index=edge_index, mask=mask)

        return output
