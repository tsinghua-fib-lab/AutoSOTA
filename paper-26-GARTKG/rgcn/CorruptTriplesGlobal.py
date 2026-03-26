import numpy as np
import torch
# from utils.util_functions import cuda, get_true_subject_and_object_per_graph
import os
import pickle
import shelve
# from utils.util_functions import write_to_shelve, write_to_default_dict
from collections import defaultdict
import pdb
import time
def cuda(tensor, device):
    torch.device(device)
    if tensor.device == torch.device('cpu'):
        return tensor.cuda(device)
    else:
        return tensor
    # target_shelve.sync()
class CorruptTriplesGlobal:
    def __init__(self,args,time2quads_train,all_known_entities):
        self.time2quads_train = time2quads_train
        self.args = args
        np.random.seed(self.args.np_seed)
        self.negative_rate = self.args.negative_rate
        self.use_cuda = self.args.use_cuda
        self.nrel=args.num_rels
        self.all_known_entities=all_known_entities
        # print("Constructing train filter")
        self.get_true_subject_object_global()

    # def set_known_entities(self):
    #     self.all_known_entities = self.model.all_known_entities
    #     known_entities = self.model.all_known_entities[self.args.end_time_step - 1] \
    #         if self.args.train_base_model else self.model.known_entities

    def write_to_shelve(self,target_shelve, cur_dict, time,mode):
        for e in cur_dict:
            for r in cur_dict[e]:
                target_shelve["{}+{}+{}".format(time, e, r)] = cur_dict[e][r]
    
    def get_true_subject_and_object_per_graph(self,triples):
        true_head = defaultdict(lambda: defaultdict(list))
        true_tail = defaultdict(lambda: defaultdict(list))
        for head, relation, tail in triples:
            head, relation, tail = head.item(), relation.item(), tail.item()
            true_tail[head][relation].append(tail)
            # true_head[tail][relation+self.nrel].append(head)
        for head, relation, tail in triples:
            head, relation, tail = head.item(), relation.item(), tail.item()
            # true_tail[head][relation].append(tail)
            true_tail[tail][relation+self.nrel].append(head)

        # this is correct
        for head in true_tail:
            for relation in true_tail[head]:
                true_tail[head][relation] = np.array(true_tail[head][relation])

        # for tail in true_head:
        #     for relation in true_head[tail]:
        #         true_head[tail][relation] = np.array(true_head[tail][relation])

        return dict(true_head), dict(true_tail)
    def get_true_subject_object_global(self):
        data_path="../data/{}/".format(self.args.dataset)
        true_subject_path = os.path.join(data_path, "true_subjects_train.db")
        true_object_path = os.path.join(data_path, "true_objects_train.db")

        if os.path.exists(os.path.join(data_path, "true_subjects_train.db.dat")) and \
                os.path.exists(os.path.join(data_path, "true_objects_train.db.dat")):
            print("loading the training shelve")
            self.true_subjects_train_global_dict = shelve.open(true_subject_path)
            self.true_objects_train_global_dict = shelve.open(true_object_path)
        else:
            print("computing the training shelve")
            # true_subjects_train_global_defaultdict = defaultdict(dict)
            # true_objects_train_global_defaultdict = defaultdict(dict)

            self.true_subjects_train_global_dict = shelve.open(true_subject_path)
            self.true_objects_train_global_dict = shelve.open(true_object_path)

            for t, quads in enumerate(self.time2quads_train):
                true_subjects_dict, true_objects_dict = self.get_true_subject_and_object_per_graph(quads)
                self.write_to_shelve(self.true_subjects_train_global_dict, true_subjects_dict, t,mode='sub')
                self.write_to_shelve(self.true_objects_train_global_dict, true_objects_dict, t,mode='obj')

    def negative_sampling(self, quadruples, negative_rate,known_entities,use_fixed_known_entities=True,times=0,mode=None):
        '''
        known_entities: 当前时间片之前的known entity
        '''
        size_of_batch = quadruples.shape[0]

        if use_fixed_known_entities:
            negative_rate = min(negative_rate, len(known_entities))

        neg_object_samples = np.zeros((size_of_batch, 1 + negative_rate), dtype=int)
        # neg_subject_samples = np.zeros((size_of_batch, 1 + negative_rate), dtype=int)
        neg_object_samples[:, 0] = quadruples[:, 2].cpu().numpy()
        # neg_subject_samples[:, 0] = quadruples[:, 0]
        labels = torch.zeros(size_of_batch)
        # t0=time.time()
        # self.true_objects_train_global_dict.update(self.true_subjects_train_global_dict)
        # t1=time.time()
        for i in range(size_of_batch):
            if mode =='re':
                s, r, o, times = quadruples[i].cpu().numpy()
                s, r, o, times = s.item(), r.item(), o.item(),times.item()
            else:
                s, r, o = quadruples[i].cpu().numpy()
                s, r, o = s.item(), r.item(), o.item()
            known_entities = known_entities if use_fixed_known_entities else self.all_known_entities[times]
            tail_samples = self.corrupt_triple(s, r, o, times, negative_rate, self.true_objects_train_global_dict, known_entities, corrupt_object=True)
            # head_samples = self.corrupt_triple(s, r, o, t, negative_rate, self.true_subjects_train_global_dict, known_entities, corrupt_object=False)
            neg_object_samples[i][0] = o
            # neg_subject_samples[i][0] = s
            neg_object_samples[i, 1:] = tail_samples
            # neg_subject_samples[i, 1:] = head_samples

        # neg_object_samples, neg_subject_samples = torch.from_numpy(neg_object_samples), torch.from_numpy(neg_subject_samples)
        neg_object_samples = torch.from_numpy(neg_object_samples)
        # if self.use_cuda:
        #     neg_object_samples, neg_subject_samples, labels = \
        #         cuda(neg_object_samples, self.args.n_gpu), cuda(neg_subject_samples, self.args.n_gpu), cuda(labels, self.args.n_gpu)
        if self.use_cuda:
            neg_object_samples, labels = \
                cuda(neg_object_samples, self.args.gpu), cuda(labels, self.args.gpu)
        return neg_object_samples.long(), labels

    def corrupt_triple(self, s, r, o, t, negative_rate, other_true_entities_dict, known_entities, corrupt_object=True):
        negative_sample_list = []
        negative_sample_size = 0

        true_entities = other_true_entities_dict["{}+{}+{}".format(t, s, r)] if \
            corrupt_object else other_true_entities_dict["{}+{}+{}".format(t, o, r)]

        while negative_sample_size < negative_rate:
            negative_sample = np.random.choice(known_entities, size=negative_rate)
            mask = np.in1d(
                negative_sample,
                true_entities,
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        return np.concatenate(negative_sample_list)[:negative_rate]