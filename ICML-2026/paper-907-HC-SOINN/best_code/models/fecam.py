import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet,SimpleCosineIncrementalNet,SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

# tune the model at first session with adapter, and then use Mahalanobis distance-based FeCAM Classifier.
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        if 'resnet' in args['backbone_type']:
            self._network = SimpleCosineIncrementalNet(args, True)
            self. batch_size=128
            self.init_lr=args["init_lr"] if args["init_lr"] is not None else  0.01
        else:
            self._network = SimpleVitNet(args, True)
            self. batch_size= args["batch_size"]
            self. init_lr=args["init_lr"]
        
        self.weight_decay=args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr=args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args

        self.args['resume'] = False  # Set to True to load trained checkpoints
        self.cov_mats = []
        self.alpha = 100

    def after_task(self):
        self._known_classes = self._total_classes
    
    def replace_fc(self,trainloader, model, args):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.backbone(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(self.train_dataset.labels)
        proto_list = []
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            cov = torch.cov(embedding.T)
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto
            self.cov_mats.append(torch.corrcoef(self.shrink_cov(cov,self.alpha)))
            
        return model

   
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train",)
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test",)
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        
        self._network.to(self._device)
        
        if self._cur_task == 0:
            # show total parameters and trainable parameters
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')
            if total_params != total_trainable_params:
                for name, param in self._network.named_parameters():
                    if param.requires_grad:
                        print(name, param.numel())
            if self.args['optimizer']=='sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
            elif self.args['optimizer']=='adam':
                optimizer=optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
            scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
            if self.args['resume']:
                self._network.load_state_dict(torch.load("{}_{}_{}_{}_{}.pkl".format(self.args["dataset"],self.args["model_name"],self.args["init_cls"],self.args["increment"],self._cur_task))["model_state_dict"])
                self._network.to(self._device)
            else:
                self._init_train(train_loader, test_loader, optimizer, scheduler)
                # Checkpoint saving disabled to save storage space
                # if self.args.get("save_checkpoint", False):
                #     self.save_checkpoint("{}_{}_{}_{}".format(self.args["dataset"],self.args["model_name"],self.args["init_cls"],self.args["increment"]))
                self._network.to(self._device)
        
        self.replace_fc(train_loader_for_protonet, self._network, None)


    def _eval_cnn(self, loader):
        print('Using FeCAM Classifier')
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)

        dists = self._maha_dist(vectors, self._network.fc.weight.data, self.cov_mats)
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _maha_dist(self, vectors, class_means, covs):
        vectors = torch.tensor(vectors).to(self._device)
        maha_dist = []
        for class_index in range(self._total_classes):
            # To use naive NCM classifier, do not send covariances 
            dist = self._mahalanobis(vectors, class_means[class_index], covs[class_index])     
            maha_dist.append(dist)
        maha_dist = np.array(maha_dist)  # [nb_classes, N]  
        return maha_dist

    def _mahalanobis(self, vectors, class_means, cov=None):
        x_minus_mu = F.normalize(vectors, p=2, dim=-1) - F.normalize(class_means, p=2, dim=-1)
        if cov is None:
            cov = torch.eye(self._network.feature_dim)  # identity covariance matrix for euclidean distance (NCM Classifier)
        inv_covmat = torch.linalg.pinv(cov).float().to(self._device)
        left_term = torch.matmul(x_minus_mu, inv_covmat)
        mahal = torch.matmul(left_term, x_minus_mu.T)
        return torch.diagonal(mahal, 0).cpu().numpy()
    
    def shrink_cov(self, cov, alpha):
        iden = torch.eye(cov.shape[0])
        cov_ = cov + (alpha*iden)
        return cov_
        
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

