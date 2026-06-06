from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np

np.random.seed(123)

class RGCNLinkDataset(object):
    def __init__(self, name, dir):
        self.name = name
        self.dir = dir
        self.dir = os.path.join(self.dir, self.name)
        print(self.dir)

    def load(self, load_time=True):
        stat_path = os.path.join(self.dir,  'stat.txt')
        entity_path = os.path.join(self.dir, 'entity2id.txt')
        relation_path = os.path.join(self.dir, 'relation2id.txt')
        train_path = os.path.join(self.dir, 'train.txt')
        valid_path = os.path.join(self.dir, 'valid.txt')
        test_path = os.path.join(self.dir, 'test.txt')
        entity_dict = _read_dictionary(entity_path)
        relation_dict = _read_dictionary(relation_path)
        self.train = np.array(_read_triplets_as_list(train_path, load_time))
        self.valid = np.array(_read_triplets_as_list(valid_path, load_time))
        self.test = np.array(_read_triplets_as_list(test_path, load_time))
        with open(os.path.join(self.dir, 'stat.txt'), 'r') as f:
            line = f.readline()
            num_nodes, num_rels,_ = line.strip().split("\t")
            num_nodes = int(num_nodes)
            num_rels = int(num_rels)
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.num_rels = len(relation_dict)
        self.relation_dict = relation_dict
        self.entity_dict = entity_dict
        print("# Sanity Check:  entities: {}".format(self.num_nodes))
        print("# Sanity Check:  relations: {}".format(self.num_rels))
        print("# Sanity Check:  edges: {}".format(len(self.train)))

    def load_context(self, load_time=True):
        train_path = os.path.join(self.dir, 'train_w_contextid.txt')
        valid_path = os.path.join(self.dir, 'valid_w_contextid.txt')
        test_path = os.path.join(self.dir, 'test_w_contextid.txt')
        self.train = np.array(_read_contexts_as_list(train_path, load_time))
        self.valid = np.array(_read_contexts_as_list(valid_path, load_time))
        self.test = np.array(_read_contexts_as_list(test_path, load_time))


def load_from_local(dir, dataset, load_context=False):
    data = RGCNLinkDataset(dataset, dir)
    if load_context:
        data.load_context()
    else:
        data.load()
    return data


def _read_dictionary(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[int(line[1])] = line[0]
    return d


def _read_triplets(filename):
    with open(filename, 'r+') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line


def _read_triplets_as_list(filename, load_time):
    l = []
    for triplet in _read_triplets(filename):
        s = int(triplet[0])
        r = int(triplet[1])
        o = int(triplet[2])
        if load_time:
            st = int(triplet[3])
            # et = int(triplet[4])
            # l.append([s, r, o, st, et])
            l.append([s, r, o, st])
        else:
            l.append([s, r, o])
    return l

def _read_contexts_as_list(filename, load_time):
    l = []
    for triplet in _read_triplets(filename):
        contextid = int(triplet[4])
        if load_time:
            st = int(triplet[3])
            # et = int(triplet[4])
            # l.append([s, r, o, st, et])
            l.append([contextid, st])
        else:
            l.append([contextid])
    return l