from tqdm import tqdm
import torch
import torch.nn.functional as F
import time
import copy
from copy import deepcopy
import numpy as np
import shutil
from tqdm import tqdm
import logging
import os
import pickle
from typing import List
import yaml
from ast import literal_eval
import logging
import clip

def Dirichlet_solver(query_features, query_labels, clip_prototypes, T=30):
    """
    This code is an implementation of the Hard EM-Dirichlet algorithm for transductive inference,
    closely adapted from the original GitHub repository accompanying the associated paper.
    The structure and logic have been preserved with minimal changes for compatibility and reproducibility.
    
    Original source: https://github.com/SegoleneMartin/transductive-CLIP
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as e:
        raise ImportError("Dirichlet_solver requires 'scipy'. Please install it to use this method.") from e
        
    def get_one_hot(y_s, n_class):
        eye = torch.eye(n_class).to(y_s.device)
        one_hot = []
        for y_task in y_s:
            one_hot.append(eye[y_task].unsqueeze(0))
        one_hot = torch.cat(one_hot, 0)
        return one_hot
    
    
    def compute_confidence_interval(data, axis=0):
        """
        Compute 95% confidence interval
        :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
        :return: the 95% confidence interval for this data.
        """
        a = 1.0 * np.array(data)
        m = np.mean(a, axis=axis)
        std = np.std(a, axis=axis)
        pm = 1.96 * (std / np.sqrt(a.shape[axis]))
        return m, pm
    
    
    class CfgNode(dict):
        """
        CfgNode represents an internal node in the configuration tree. It's a simple
        dict-like container that allows for attribute-based access to keys.
        """
    
        def __init__(self, init_dict=None, key_list=None, new_allowed=False):
            # Recursively convert nested dictionaries in init_dict into CfgNodes
            init_dict = {} if init_dict is None else init_dict
            key_list = [] if key_list is None else key_list
            for k, v in init_dict.items():
                if type(v) is dict:
                    # Convert dict to CfgNode
                    init_dict[k] = CfgNode(v, key_list=key_list + [k])
            super(CfgNode, self).__init__(init_dict)
    
        def __getattr__(self, name):
            if name in self:
                return self[name]
            else:
                raise AttributeError(name)
    
        def __setattr__(self, name, value):
            self[name] = value
    
        def __str__(self):
            def _indent(s_, num_spaces):
                s = s_.split("\n")
                if len(s) == 1:
                    return s_
                first = s.pop(0)
                s = [(num_spaces * " ") + line for line in s]
                s = "\n".join(s)
                s = first + "\n" + s
                return s
    
            r = ""
            s = []
            for k, v in sorted(self.items()):
                seperator = "\n" if isinstance(v, CfgNode) else " "
                attr_str = "{}:{}{}".format(str(k), seperator, str(v))
                attr_str = _indent(attr_str, 2)
                s.append(attr_str)
            r += "\n".join(s)
            return r
    
        def __repr__(self):
            return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())
    
    
    def _decode_cfg_value(v):
        if not isinstance(v, str):
            return v
        try:
            v = literal_eval(v)
        except ValueError:
            pass
        except SyntaxError:
            pass
        return v
    
    
    def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
        original_type = type(original)
    
        replacement_type = type(replacement)
    
        # The types must match (with some exceptions)
        if replacement_type == original_type:
            return replacement
    
        def conditional_cast(from_type, to_type):
            if replacement_type == from_type and original_type == to_type:
                return True, to_type(replacement)
            else:
                return False, None
    
        casts = [(tuple, list), (list, tuple)]
        try:
            casts.append((str, unicode))  # noqa: F821
        except Exception:
            pass
    
        for (from_type, to_type) in casts:
            converted, converted_value = conditional_cast(from_type, to_type)
            if converted:
                return converted_value
    
        raise ValueError(
            "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
            "key: {}".format(
                original_type, replacement_type, original, replacement, full_key
            )
        )
    
    
    def load_cfg_from_cfg_file(file: str):
        cfg = {}
        assert os.path.isfile(file) and file.endswith('.yaml'), \
            '{} is not a yaml file'.format(file)
    
        with open(file, 'r') as f:
            cfg_from_file = yaml.safe_load(f)
    
        for key in cfg_from_file:
            for k, v in cfg_from_file[key].items():
                cfg[k] = v
    
        cfg = CfgNode(cfg)
        return cfg
    
    
    def merge_cfg_from_list(cfg: CfgNode,
                            cfg_list: List[str]):
        new_cfg = copy.deepcopy(cfg)
        assert len(cfg_list) % 2 == 0, cfg_list
        for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
            subkey = full_key.split('.')[-1]
            # assert subkey in cfg, 'Non-existent key: {}'.format(full_key)
            if subkey in cfg:
                value = _decode_cfg_value(v)
                value = _check_and_coerce_cfg_value_type(
                    value, cfg[subkey], subkey, full_key
                )
                setattr(new_cfg, subkey, value)
            else:
                value = _decode_cfg_value(v)
                setattr(new_cfg, subkey, value)
        return new_cfg
    
    
    class Logger:
        def __init__(self, module_name, filename):
            self.module_name = module_name
            self.filename = filename
            self.formatter = self.get_formatter()
            self.file_handler = self.get_file_handler()
            self.stream_handler = self.get_stream_handler()
            self.logger = self.get_logger()
    
        def get_formatter(self):
            log_format = '[%(name)s]: [%(levelname)s]: %(message)s'
            formatter = logging.Formatter(log_format)
            return formatter
    
        def get_file_handler(self):
            file_handler = logging.FileHandler(self.filename)
            file_handler.setFormatter(self.formatter)
            return file_handler
    
        def get_stream_handler(self):
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(self.formatter)
            return stream_handler
    
        def get_logger(self):
            logger = logging.getLogger(self.module_name)
            logger.setLevel(logging.INFO)
            logger.addHandler(self.file_handler)
            logger.addHandler(self.stream_handler)
            return logger
    
        def del_logger(self):
            handlers = self.logger.handlers[:]
            for handler in handlers:
                handler.close()
                self.logger.removeHandler(handler)
    
        def info(self, msg):
            self.logger.info(msg)
    
        def debug(self, msg):
            self.logger.debug(msg)
    
        def warning(self, msg):
            self.logger.warning(msg)
    
        def critical(self, msg):
            self.logger.critical(msg)
    
        def exception(self, msg):
            self.logger.exception(msg)
    
    
    def make_log_dir(log_path, dataset, method):
        log_dir = os.path.join(log_path, dataset, method)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        return log_dir
    
    
    def get_log_file(log_path, dataset, method):
        log_dir = make_log_dir(log_path=log_path, dataset=dataset, method=method)
        i = 0
        filename = os.path.join(log_dir, '{}_run_{}.log'.format(method, i))
        while os.path.exists(os.path.join(log_dir, '{}_run_%s.log'.format(method)) % i):
            i += 1
            filename = os.path.join(log_dir, '{}_run_{}.log'.format(method, i))
        return filename
    
    
    def save_pickle(file, data):
        with open(file, 'wb') as f:
            pickle.dump(data, f)
    
    
    def load_pickle(file):
        with open(file, 'rb') as f:
            return pickle.load(f)
    
    
    def extract_features_softmax(model, dataset, loader, set_name, args,
                                 device, list_T=[10, 20, 30, 40, 50]):
        """
            inputs:
                model : The loaded model containing the feature extractor
                train_loader : Train data loader
                args : arguments
                device : GPU device
    
            returns :
                Saves the features in data/args.dataset/saved_features/ for T in list_T under the name 
                '{}_softmax_{}_T{}.plk'.format(set_name, args.backbone, T)
        """
        for T in list_T:
            # Check if features are already saved
            features_save_path = 'data/{}/saved_features/{}_softmax_{}_T{}.plk'.format(
                args.dataset, set_name, args.backbone, T)
            if os.path.exists(features_save_path):
                print('Features already saved for {} set and T = {}, skipping'.format(
                    set_name, T))
                continue
            else:
                print('Extracting {} features on {} for T = {}'.format(
                    set_name, args.dataset, T))
    
            # Create text embeddings for all classes in the dataset
            text_features = clip_weights(
                model, dataset.classnames, dataset.template, device).float()
    
            # Extract features and labels
            for i, (images, labels) in enumerate(tqdm(loader)):
    
                # for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    image_features = model.encode_image(images).float()
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    similarity = (T * image_features @
                                  text_features.T).softmax(dim=-1)
                    if i == 0:
                        all_features = similarity.cpu()
                        all_labels = labels.cpu()
                    else:
                        all_features = torch.cat(
                            (all_features, similarity.cpu()), dim=0)
                        all_labels = torch.cat((all_labels, labels.cpu()), dim=0)
    
            # Save features
            extracted_features_dic = {
                'concat_features': all_features, 'concat_labels': all_labels}
            try:
                os.mkdir('data/{}/saved_features/'.format(args.dataset))
            except:
                pass
            save_pickle(features_save_path, extracted_features_dic)
    
    
    def extract_features_visual(model, dataset, loader, set_name, args,
                                device):
        """
            inputs:
                model : The loaded model containing the feature extractor
                train_loader : Train data loader
                args : arguments
                device : GPU device
    
            returns :
                Saves the features in data/args.dataset/saved_features/ under the name 
                '{}_visual_{}.plk'.format(set_name, args.backbone)
        """
    
        # Check if features are already saved
        features_save_path = 'data/{}/saved_features/{}_visual_{}.plk'.format(
            args.dataset, set_name, args.backbone)
        if os.path.exists(features_save_path):
            print('Features already saved for {} set, skipping'.format(
                set_name))
        else:
            print('Extracting {} features on {}'.format(set_name, args.dataset))
    
            # Create text embeddings for all classes in the dataset
            text_features = clip_weights(
                model, dataset.classnames, dataset.template, device).float()
    
            # Extract features and labels
            for i, (images, labels) in enumerate(tqdm(loader)):
    
                # for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    image_features = model.encode_image(images).float()
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    if i == 0:
                        all_features = image_features
                        all_labels = labels.cpu()
                    else:
                        all_features = torch.cat(
                            (all_features, image_features), dim=0)
                        all_labels = torch.cat((all_labels, labels.cpu()), dim=0)
    
            # Save features
            extracted_features_dic = {
                'concat_features': all_features, 'concat_labels': all_labels}
            try:
                os.mkdir('data/{}/saved_features/'.format(args.dataset))
            except:
                pass
            save_pickle(features_save_path, extracted_features_dic)
    
    
    def clip_weights(model, classnames, template, device):
    
        new_classnames = []
        for classname in classnames:
            classname = classname.replace('_', ' ')
            new_classnames.append(classname)
        classnames = new_classnames
    
        text_inputs = torch.cat([clip.tokenize(
            [template.format(classname) for classname in classnames])]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
        return text_features
    
    
    def compute_graph_matching(preds_q, probs, n_class):
    
        new_preds_q = torch.zeros_like(preds_q)
        n_task = preds_q.shape[0]
        list_clusters = []
        list_A = []
    
        for task in range(n_task):
            clusters = []
            num_clusters = len(torch.unique(preds_q[task]))
            A = np.zeros((num_clusters, int(n_class)))
            for i, cluster in enumerate(preds_q[task]):
                if cluster.item() not in clusters:
                    A[len(clusters), :] = - probs[task, cluster].cpu().numpy()
                    clusters.append(cluster.item())
            list_A.append(A)
            list_clusters.append(clusters)
    
        for task in range(n_task):
            A = list_A[task]
            clusters = list_clusters[task]
            __, matching_classes = linear_sum_assignment(A, maximize=False)
            for i, cluster in enumerate(preds_q[task]):
                new_preds_q[task, i] = matching_classes[clusters.index(cluster)]
    
        return new_preds_q
    
    
    def compute_basic_matching(preds_q, probs, args):
    
        new_preds_q = torch.zeros_like(preds_q)
        n_task = preds_q.shape[0]
    
        for task in range(n_task):
            matching_classes = probs[task].argmax(dim=-1)  # K
            new_preds_q[task] = matching_classes[preds_q[task]]
    
        return new_preds_q
    
    
    class BASE(object):
    
        def __init__(self, model, device, log_file, num_classes, n_query):
            self.device = device
            self.iter = 10
            self.lambd = int(num_classes / 5) * n_query
            self.model = model
            self.init_info_lists()
            self.num_classes = num_classes
            self.n_query = n_query
            self.eps = 1e-15
            self.iter_mm = 1000
    
        def init_info_lists(self):
            self.timestamps = []
            self.criterions = []
            self.test_acc = []
    
        def get_logits(self, samples):
            """
            inputs:
                samples : torch.Tensor of shape [n_task, shot, feature_dim]
            returns :
                logits : torch.Tensor of shape [n_task, shot, num_class]
            """
            l1 = torch.lgamma(self.alpha.sum(-1)).unsqueeze(1)
            l2 = - torch.lgamma(self.alpha).sum(-1).unsqueeze(1)
            l3 = ((self.alpha.unsqueeze(1) - 1) *
                  torch.log(samples + self.eps).unsqueeze(2)).sum(-1)
            logits = l1 + l2 + l3
            return logits  # N x n x K
    
        def record_convergence(self, new_time, criterions):
            """
            inputs:
                new_time : scalar
                criterions : torch.Tensor of shape [n_task]
            """
            self.criterions.append(criterions)
            self.timestamps.append(new_time)
    
        def compute_acc(self, y_q):
            """
            inputs:
                y_q : torch.Tensor of shape [n_task, n_query] :
            """
    
            preds_q = self.u.argmax(2)
            accuracy = (preds_q == y_q).float().mean(1, keepdim=True)
            self.test_acc.append(accuracy)
    
        def compute_acc_clustering(self, query, y_q):
            n_task = query.shape[0]
            preds_q = self.u.argmax(2)
            preds_q_one_hot = get_one_hot(preds_q, self.num_classes)
    
            prototypes = ((preds_q_one_hot.unsqueeze(-1) * query.unsqueeze(2)).sum(1)
                          ) / (preds_q_one_hot.sum(1).clamp(min=self.eps).unsqueeze(-1))
            cluster_sizes = preds_q_one_hot.sum(1).unsqueeze(-1)  # N x K
            nonzero_clusters = cluster_sizes > self.eps
            prototypes = prototypes * nonzero_clusters
    
            
            probs = prototypes
        
    
            new_preds_q = compute_graph_matching(preds_q, probs, n_class=self.num_classes)

            accuracy = (new_preds_q == y_q).float().mean(1, keepdim=True)
            self.test_acc.append(accuracy)
            
            return accuracy
    
        def get_logs(self):
            self.criterions = torch.stack(self.criterions, dim=0).cpu().numpy()
            self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
            return {'timestamps': np.array(self.timestamps).mean(), 'criterions': self.criterions,
                    'acc': self.test_acc}
    
        def run_task(self, task_dic):
            """
            inputs:
                task_dic : dictionnary with n_task few-shot tasks
                shot : scalar, number of shots
            """
    
            # Extract support and query
            y_q = task_dic['y_q']               # [n_task, n_query]
            query = task_dic['x_q']             # [n_task, n_query, feature_dim]
    
            # Transfer tensors to GPU if needed
            query = query.to(self.device).float()
            y_q = y_q.long().squeeze(2).to(self.device)
            del task_dic
    
            # Run adaptation
            self.run_method(query=query, y_q=y_q)
    
            # Extract adaptation logs
            logs = self.get_logs()
            return logs
    
    
    class HARD_EM_DIRICHLET(BASE):
    
        def __init__(self, model, device, log_file, num_classes, n_query):
            super().__init__(model=model, device=device, log_file=log_file, num_classes=num_classes, n_query=n_query)
        def u_update(self, query):
            """
            inputs:
                query : torch.Tensor of shape [n_task, n_query, feature_dim]
    
            updates:
                self.u : torch.Tensor of shape [n_task, n_query, num_class]
            """
            __, n_query = query.size(-1), query.size(1)
            logits = self.get_logits(query)
            self.u = (logits + self.lambd *
                      self.v.unsqueeze(1) / n_query).softmax(2)
    
        def v_update(self):
            """
            updates:
                self.v : torch.Tensor of shape [n_task, num_class]
                --> corresponds to the log of the class proportions
            """
            self.v = torch.log(self.u.sum(1) / self.u.size(1) + self.eps) + 1
    
        def curvature(self, alpha):
            digam = torch.polygamma(0, alpha + 1)
            return torch.where(alpha > 1e-11, abs(2 * (self.log_gamma_1 - torch.lgamma(alpha + 1) + digam * alpha) / alpha**2), self.zero_value), digam
    
        def update_alpha(self, alpha_0, y_cst):
            alpha = deepcopy(alpha_0)
    
            for l in range(self.iter_mm):
                curv, digam = self.curvature(alpha)
                b = digam - \
                    torch.polygamma(0, alpha.sum(-1)).unsqueeze(-1) - curv * alpha
                b = b - y_cst
                a = curv
                delta = b**2 + 4 * a
                alpha_new = (- b + torch.sqrt(delta)) / (2 * a)
    
                if l > 0 and l % 50 == 0:
                    criterion = torch.norm(
                        alpha_new - alpha)**2 / torch.norm(alpha)**2
                    if criterion < 1e-11:
                        break
                alpha = deepcopy(alpha_new)
            self.alpha = deepcopy(alpha_new)
    
        def objective_function(self, support, query, y_s_one_hot):
            l1 = torch.lgamma(self.alpha.sum(-1)).unsqueeze(1)
            l2 = - torch.lgamma(self.alpha).sum(-1).unsqueeze(1)
            l3 = ((self.alpha.unsqueeze(1) - 1) *
                  torch.log(query + self.eps).unsqueeze(2)).sum(-1)
            datafit_query = -(self.u * (l1 + l2 + l3)).sum(-1).sum(1)
            l1 = torch.lgamma(self.alpha.sum(-1)).unsqueeze(1)
            l2 = - torch.lgamma(self.alpha).sum(-1).unsqueeze(1)
            l3 = ((self.alpha.unsqueeze(1) - 1) *
                  torch.log(support + self.eps).unsqueeze(2)).sum(-1)
            datafit_support = -(y_s_one_hot * (l1 + l2 + l3)).sum(-1).sum(1)
            datafit = 1 / 2 * (datafit_query + datafit_support)
    
            reg_ent = (self.u * torch.log(self.u + self.eps)).sum(-1).sum(1)
    
            props = self.u.mean(1)
            part_complexity = - self.lambd * \
                (props * torch.log(props + self.eps)).sum(-1)
    
            return datafit + reg_ent + part_complexity
    
        def run_method(self, query, y_q):
            """
            Corresponds to the Hard EM DIRICHLET inference
            inputs:
                query : torch.Tensor of shape [n_task, n_query, feature_dim]
                y_q : torch.Tensor of shape [n_task, n_query]
    
            updates :from copy import deepcopy
                self.u : torch.Tensor of shape [n_task, n_query, num_class]         (soft labels)
                self.v : torch.Tensor of shape [n_task, num_class]                  (dual variable)
                self.w : torch.Tensor of shape [n_task, num_class, feature_dim]     (centroids)
            """
            self.zero_value = torch.polygamma(1, torch.Tensor(
                [1]).to(self.device)).float()  # .double()
            self.log_gamma_1 = torch.lgamma(
                torch.Tensor([1]).to(self.device)).float()
    
            n_task, n_class = query.shape[0], self.num_classes
    
            # Initialization
            self.v = torch.zeros(n_task, n_class).to(
                self.device)        # dual variable set to zero
            
            self.u = deepcopy(query)
            
            self.alpha = torch.ones((n_task, n_class, n_class)).to(self.device)
            alpha_old = deepcopy(self.alpha)
            t0 = time.time()
    
            pbar = range(self.iter)
            for i in pbar:
    
                # update of dirichlet parameter alpha
                cluster_sizes = self.u.sum(
                    dim=1).unsqueeze(-1).float()  # .double() # N x K
                nonzero_clusters = cluster_sizes > self.eps
                y_cst = ((self.u.unsqueeze(-1) * torch.log(query + self.eps).unsqueeze(2)
                          ).sum(1)) / (self.u.sum(1).clamp(min=self.eps).unsqueeze(-1))
                y_cst = y_cst * nonzero_clusters + \
                    (1 - 1 * nonzero_clusters) * torch.ones_like(y_cst) * (-10)
                self.update_alpha(self.alpha, y_cst)
                alpha_new = self.alpha * nonzero_clusters + \
                    alpha_old * (1 - 1 * nonzero_clusters)
                self.alpha = alpha_new
                del alpha_new
    
                # update on dual variable v (= log of class proportions)
                self.v_update()
    
                # update hard assignment variable u
                self.u_update(query)
                labels = torch.argmax(self.u, dim=-1)
                self.u.zero_()
                self.u.scatter_(2, labels.unsqueeze(-1), 1.0)
    
                alpha_diff = ((alpha_old - self.alpha).norm(dim=(1, 2)
                                                            ) / alpha_old.norm(dim=(1, 2))).mean(0)
                criterions = alpha_diff
                alpha_old = deepcopy(self.alpha)
                t1 = time.time()
    
                t1 = time.time()
                self.record_convergence(
                    new_time=(t1-t0) / n_task, criterions=criterions)

    device = query_features.device

    query_labels = query_labels.to(device).long()
    clip_prototypes = clip_prototypes.to(device).float()
    query_features = query_features.to(device).float()

    # Initial zero-shot predictions
    logits = T * query_features @ clip_prototypes  # [N, K]
    y_hat = F.softmax(logits, dim=-1)              # [N, K]
    
    # Dirichlet expects batched input
    query_features_batch = y_hat      # Shape: [1, N, K]
    query_labels_batch = query_labels.unsqueeze(0).unsqueeze(-1)  # Shape: [1, N, 1]

    
    method = HARD_EM_DIRICHLET(
        model=None,
        device=device,
        log_file=None,
        num_classes=clip_prototypes.shape[0],
        n_query=query_features.shape[0]
    )

    method.run_method(query_features_batch, query_labels_batch)

    z = method.u 

    return y_hat.squeeze().cpu(), z.squeeze().cpu()
     
        
    
