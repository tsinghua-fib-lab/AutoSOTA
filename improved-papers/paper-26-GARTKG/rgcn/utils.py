import numpy as np
import torch
import dgl
from tqdm import tqdm
import rgcn.knowledge_graph as knwlgrh
from collections import defaultdict
import copy
import json
import os
import pickle
from collections import defaultdict
# from copy import deepcopy

#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################
def temp_func():
    return defaultdict(int)

def get_history(data_list,n_r,dataset):

    query_history=dict()
    query_er_history=dict()
    
    
    for train_sample_num in tqdm(range(len(data_list))):

        output_re=data_list[train_sample_num]

        query_history[str(train_sample_num)]=dict()
        query_er_history[str(train_sample_num)]=dict()
        for i,element in enumerate(output_re):
            
            if str(element[0]) not in query_history[str(train_sample_num)].keys():
                query_history[str(train_sample_num)][str(element[0])]=set()
            if str(element[2]) not in query_history[str(train_sample_num)].keys():
                query_history[str(train_sample_num)][str(element[2])]=set()
            
            query_history[str(train_sample_num)][str(element[2])].add(tuple([element[2],element[1]+n_r,element[0]]))
            query_history[str(train_sample_num)][str(element[0])].add(tuple(element))

            if str(element[0]) not in query_er_history[str(train_sample_num)].keys():
                query_er_history[str(train_sample_num)][str(element[0])]=dict()
                query_er_history[str(train_sample_num)][str(element[0])][str(element[1])]=set()
            else:
                if str(element[1]) not in query_er_history[str(train_sample_num)][str(element[0])].keys():
                    query_er_history[str(train_sample_num)][str(element[0])][str(element[1])]=set()
            query_er_history[str(train_sample_num)][str(element[0])][str(element[1])].add(tuple(element))
            
        file_path="../data/{}/history_snap_v2/snap_{}.pkl".format(dataset,train_sample_num)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)    
        with open(file_path,"wb") as file:
            pickle.dump(query_history[str(train_sample_num)],file)

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


#TODO filer by groud truth in the same time snapshot not all ground truth
def sort_and_rank_time_filter(batch_a, batch_r, score, target, total_triplets):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    for i in range(len(batch_a)):
        ground = indices[i]
    indices = indices[:, 1].view(-1)
    return indices


def sort_and_rank_filter(batch_a, batch_r, score, target, all_ans):
    for i in range(len(batch_a)):
        ans = target[i]
        b_multi = list(all_ans[batch_a[i].item()][batch_r[i].item()])
        ground = score[i][ans]
        score[i][b_multi] = 0
        score[i][ans] = ground
    _, indices = torch.sort(score, dim=1, descending=True)  # indices : [B, number entity]
    indices = torch.nonzero(indices == target.view(-1, 1))  # indices : [B, 2] 第一列递增， 第二列表示对应的答案实体id在每一行的位置
    indices = indices[:, 1].view(-1)
    return indices


def filter_score(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][r.item()])
        ans.remove(t.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score

def filter_score_r(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][t.item()])
        # print(h, r, t)
        # print(ans)
        ans.remove(r.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score


def r2e(triplets, num_rels):
    src, rel, dst = triplets.transpose()
    # get all relations
    uniq_r = np.unique(rel)
    uniq_r = np.concatenate((uniq_r, uniq_r+num_rels))
    # generate r2e
    r_to_e = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        r_to_e[rel].add(src)
        r_to_e[rel].add(dst)
        r_to_e[rel+num_rels].add(src)
        r_to_e[rel+num_rels].add(dst)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_r:
        r_len.append((idx,idx+len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx


def build_sub_graph(num_nodes, num_rels, triples, use_cuda, gpu):
    """
    :param node_id: node id in the large graph
    :param num_rels: number of relation
    :param src: relabeled src id
    :param rel: original rel id
    :param dst: relabeled dst id
    :param use_cuda:
    :return:
    """
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm
    if isinstance(triples, np.ndarray):
        src, rel, dst = triples.transpose()
    else:
        triples=np.array(triples)
        src, rel, dst = triples.transpose()
    # src, rel, dst = triples.transpose()
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)

    uniq_r, r_len, r_to_e = r2e(triples, num_rels)
    g.uniq_r = uniq_r
    g.r_to_e = r_to_e
    g.r_len = r_len
    if use_cuda:
        g = g.to(gpu) 
        g.r_to_e = torch.from_numpy(np.array(r_to_e))
    return g

def get_total_rank(test_triples, score, all_ans, eval_bz, rel_predict=0):
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz
    rank = []
    filter_rank = []
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]
        if rel_predict==1:
            target = test_triples[batch_start:batch_end, 1]
        elif rel_predict == 2:
            target = test_triples[batch_start:batch_end, 0]
        else:
            target = test_triples[batch_start:batch_end, 2]
        rank.append(sort_and_rank(score_batch, target))

        if rel_predict:
            filter_score_batch = filter_score_r(triples_batch, score_batch, all_ans)
        else:
            filter_score_batch = filter_score(triples_batch, score_batch, all_ans)
        filter_rank.append(sort_and_rank(filter_score_batch, target))

    rank = torch.cat(rank)
    filter_rank = torch.cat(filter_rank)
    rank += 1 # change to 1-indexed
    filter_rank += 1
    mrr = torch.mean(1.0 / rank.float())
    filter_mrr = torch.mean(1.0 / filter_rank.float())
    return filter_mrr.item(), mrr.item(), rank, filter_rank


def stat_ranks(rank_list, method):
    hits = [1, 3, 10]
    total_rank = torch.cat(rank_list)

    mrr = torch.mean(1.0 / total_rank.float())
    print("MRR ({}): {:.6f}".format(method, mrr.item()))
    hit_result=[]
    for hit in hits:
        avg_count = torch.mean((total_rank <= hit).float())
        hit_result.append(avg_count)
        print("Hits ({}) @ {}: {:.6f}".format(method, hit, avg_count.item()))
    return mrr

def stat_ranks_val(rank_list, method):
    hits = [1, 3, 10]
    total_rank = torch.cat(rank_list)

    mrr = torch.mean(1.0 / total_rank.float())
    # print("MRR ({}): {:.6f}".format(method, mrr.item()))
    hit_result=[]
    for hit in hits:
        avg_count = torch.mean((total_rank <= hit).float())
        hit_result.append(avg_count)
        # print("Hits ({}) @ {}: {:.6f}".format(method, hit, avg_count.item()))
        
    return mrr
def stat_ranks_record(rank_list, method):
    hits = [1, 3, 10]
    hits_ratio=[]
    # total_rank = torch.cat(rank_list)
    total_rank = rank_list

    mrr = torch.mean(1.0 / total_rank.float())
    for hit in hits:
        avg_count = torch.mean((total_rank <= hit).float())
        hits_ratio.append(avg_count)
    return mrr,hits_ratio


def flatten(l):
    flatten_l = []
    for c in l:
        if type(c) is list or type(c) is tuple:
            flatten_l.extend(flatten(c))
        else:
            flatten_l.append(c)
    return flatten_l

def UnionFindSet(m, edges):
    """

    :param m:
    :param edges:
    :return: union number in a graph
    """
    roots = [i for i in range(m)]
    rank = [0 for i in range(m)]
    count = m

    def find(member):
        tmp = []
        while member != roots[member]:
            tmp.append(member)
            member = roots[member]
        for root in tmp:
            roots[root] = member
        return member

    for i in range(m):
        roots[i] = i
    # print ufs.roots
    for edge in edges:
        print(edge)
        start, end = edge[0], edge[1]
        parentP = find(start)
        parentQ = find(end)
        if parentP != parentQ:
            if rank[parentP] > rank[parentQ]:
                roots[parentQ] = parentP
            elif rank[parentP] < rank[parentQ]:
                roots[parentP] = parentQ
            else:
                roots[parentQ] = parentP
                rank[parentP] -= 1
            count -= 1
    return count

def append_object(e1, e2, r, d):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def add_subject(e1, e2, r, d, num_rel):
    if not e2 in d:
        d[e2] = {}
    if not r+num_rel in d[e2]:
        d[e2][r+num_rel] = set()
    d[e2][r+num_rel].add(e1)


def add_object(e1, e2, r, d, num_rel):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def load_all_answers(total_data, num_rel):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    all_subjects, all_objects = {}, {}
    for line in total_data:
        s, r, o = line[: 3]
        add_subject(s, o, r, all_subjects, num_rel=num_rel)
        add_object(s, o, r, all_objects, num_rel=0)
    return all_objects, all_subjects


def load_all_answers_for_filter(total_data, num_rel, rel_p=False):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    def add_relation(e1, e2, r, d):
        if not e1 in d:
            d[e1] = {}
        if not e2 in d[e1]:
            d[e1][e2] = set()
        d[e1][e2].add(r)

    all_ans = {}
    for line in total_data:
        s, r, o = line[: 3]
        if rel_p:
            add_relation(s, o, r, all_ans)
            add_relation(o, s, r + num_rel, all_ans)
        else:
            add_subject(s, o, r, all_ans, num_rel=num_rel)
            add_object(s, o, r, all_ans, num_rel=0)
    return all_ans


def load_all_answers_for_time_filter(total_data, num_rels, num_nodes, rel_p=False):
    all_ans_list = []
    # all_snap = split_by_time(total_data)
    for snap in total_data:
        all_ans_t = load_all_answers_for_filter(snap, num_rels, rel_p)
        all_ans_list.append(all_ans_t)

    # output_label_list = []
    # for all_ans in all_ans_list:
    #     output = []
    #     ans = []
    #     for e1 in all_ans.keys():
    #         for r in all_ans[e1].keys():
    #             output.append([e1, r])
    #             ans.append(list(all_ans[e1][r]))
    #     output = torch.from_numpy(np.array(output))
    #     output_label_list.append((output, ans))
    # return output_label_list
    return all_ans_list

def split_by_time(data):
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        train = data[i]
        # latest_t表示读取的上一个三元组发生的时刻，要求数据集中的三元组是按照时间发生顺序排序的
        if latest_t != t:  # 同一时刻发生的三元组
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:3])
    # 加入最后一个shapshot
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True)  # relabel
        uniq_r = np.unique(snapshot[:,1])
        edges = np.reshape(edges, (2, -1))
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r)*2)
    print("# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}"
          .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list]), max(union_num), min(union_num)))
    return snapshot_list


def slide_list(snapshots, k=1):
    """
    :param k: padding K history for sequence stat
    :param snapshots: all snapshot
    :return:
    """
    k = k  # k=1 需要取长度k的历史，在加1长度的label
    if k > len(snapshots):
        print("ERROR: history length exceed the length of snapshot: {}>{}".format(k, len(snapshots)))
    for _ in tqdm(range(len(snapshots)-k+1)):
        yield snapshots[_: _+k]



def load_data(dataset, bfs_level=3, relabel=False):
    if dataset in ['aifb', 'mutag', 'bgs', 'am']:
        return knwlgrh.load_entity(dataset, bfs_level, relabel)
    elif dataset in ['FB15k', 'wn18', 'FB15k-237']:
        return knwlgrh.load_link(dataset)
    elif dataset in ['ICEWS18', 'ICEWS14', "GDELT", "SMALL", "ICEWS14s", "ICEWS05-15","YAGO",
                     "WIKI"]:
        return knwlgrh.load_from_local("../data", dataset)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

def construct_snap(test_triples, num_nodes, num_rels, final_score, topK):
    sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_triples = []
    for _ in range(len(test_triples)):
        for index in top_indices[_]:
            h, r = test_triples[_][0], test_triples[_][1]
            if r < num_rels:
                predict_triples.append([test_triples[_][0], r, index])
            else:
                predict_triples.append([index, r-num_rels, test_triples[_][0]])

    # 转化为numpy array
    predict_triples = np.array(predict_triples, dtype=int)
    return predict_triples

def construct_snap_r(test_triples, num_nodes, num_rels, final_score, topK):
    sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_triples = []
    # for _ in range(len(test_triples)):
    #     h, r = test_triples[_][0], test_triples[_][1]
    #     if (sorted_score[_][0]-sorted_score[_][1])/sorted_score[_][0] > 0.3:
    #         if r < num_rels:
    #             predict_triples.append([h, r, indices[_][0]])

    for _ in range(len(test_triples)):
        for index in top_indices[_]:
            h, t = test_triples[_][0], test_triples[_][2]
            if index < num_rels:
                predict_triples.append([h, index, t])
                #predict_triples.append([t, index+num_rels, h])
            else:
                predict_triples.append([t, index-num_rels, h])
                #predict_triples.append([t, index-num_rels, h])

    # 转化为numpy array
    predict_triples = np.array(predict_triples, dtype=int)
    return predict_triples

def spilt_data(data_list):
    train=[]
    vaild=[]
    test=[]
    train_spilt=0.8
    vaild_split=0.9
    for data in data_list:
        # np.random.shuffle(data)
        train.append(data[:int(train_spilt*len(data)),:])
        vaild.append(data[int(train_spilt*len(data)):int(vaild_split*len(data)),:])
        test.append(data[int(vaild_split*len(data)):,:])
    return train,vaild,test

def spilt_data_for_retraining(train_list,vaild_list,test_list):
    train=[]
    vaild=[]
    test=[]
    for i in range(len(train_list)):
        train_item=[]
        vaild_item=[]
        test_item=[]
        for j in range(0,i+1):
            train_item.append(train_list[j])
            vaild_item.append(vaild_list[j])
            test_item.append(test_list[j])
        train.append(train_item)
        vaild.append(vaild_item)
        test.append(test_item)
    return train,vaild,test
        
    # train_spilt=0.8
    # vaild_split=0.9
    # for data in data_list:
    #     # np.random.shuffle(data)
    #     train.append(data[:int(train_spilt*len(data)),:])
    #     vaild.append(data[int(train_spilt*len(data)):int(vaild_split*len(data)),:])
    #     test.append(data[int(vaild_split*len(data)):,:])
    return train,vaild,test

#TIE
def get_add_graph_global(args, time2quads_train):
    added_edges_dict = dict()
    deleted_edges_dict = dict()
    last_edge_set = None
    for time, quads in enumerate(time2quads_train):
        time_s = str(time)
        cur_edge_set = set([(s.item(), r.item(), o.item()) for s, r, o in quads])

        if type(last_edge_set) == type(None):
            added_edges_dict[time_s] = quads
            deleted_edges_dict[time_s] = None
        else:
            added_idx, deleted_idx = cur_edge_set - last_edge_set, last_edge_set - cur_edge_set
            added_edges_dict[time_s] = torch.tensor([list(elem) for elem in added_idx])
            deleted_edges_dict[time_s] = torch.tensor([list(elem) for elem in deleted_idx])

        last_edge_set = cur_edge_set
    return added_edges_dict

def get_known_entity_relation(args, time2quads_train, num_ents, num_rels):
    all_known_entities = {}
    all_known_relations = {}

    occurred_entity_positive_mask = np.zeros(num_ents)
    occurred_relation_positive_mask = np.zeros(num_rels)
    # for t in time2quads_train.keys():
    for t,quads in enumerate(time2quads_train):
        for quad in quads:
            occurred_entity_positive_mask[quad[0]] = 1
            occurred_entity_positive_mask[quad[2]] = 1
            occurred_relation_positive_mask[quad[1]] = 1
        all_known_entities[t] = occurred_entity_positive_mask.nonzero()[0]
        all_known_relations[t] = occurred_relation_positive_mask.nonzero()[0]
        
    return all_known_entities, all_known_relations#存储当前时间片及之前已知的实体和关系的id

def load_quadruples(args, time2quads_train):
    quadrupleList = []
    times = set()
    for time,line in enumerate(time2quads_train):
        for line_split in line:
        # line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            # time = int(line_split[3])
            quadrupleList.append((head, rel, tail, time))
        times.add(time)
    times = list(times)
    times.sort()
    # quadrupleList, times = list(set(quadrupleList)), np.asarray(times)
    return list(set(quadrupleList)), np.asarray(times)

def negative_sampling(self, quadruples, negative_rate,known_entities, use_fixed_known_entities=True):
    size_of_batch = quadruples.shape[0]

    if use_fixed_known_entities:
        negative_rate = min(negative_rate, len(known_entities))

    neg_object_samples = np.zeros((size_of_batch, 1 + negative_rate), dtype=int)
    neg_subject_samples = np.zeros((size_of_batch, 1 + negative_rate), dtype=int)
    neg_object_samples[:, 0] = quadruples[:, 2]
    neg_subject_samples[:, 0] = quadruples[:, 0]
    labels = torch.zeros(size_of_batch)
    for i in range(size_of_batch):
        s, r, o, t = quadruples[i]
        s, r, o, t = s.item(), r.item(), o.item(), t.item()
        # known_entities = known_entities if use_fixed_known_entities else self.all_known_entities[t]
        known_entities = known_entities
        tail_samples = corrupt_triple(s, r, o, t, negative_rate, self.true_objects_train_global_dict, known_entities, corrupt_object=True)
        head_samples = corrupt_triple(s, r, o, t, negative_rate, self.true_subjects_train_global_dict, known_entities, corrupt_object=False)
        neg_object_samples[i][0] = o
        neg_subject_samples[i][0] = s
        neg_object_samples[i, 1:] = tail_samples
        neg_subject_samples[i, 1:] = head_samples

    neg_object_samples, neg_subject_samples = torch.from_numpy(neg_object_samples), torch.from_numpy(neg_subject_samples)
    if self.use_cuda:
        neg_object_samples, neg_subject_samples, labels = \
            cuda(neg_object_samples, self.args.n_gpu), cuda(neg_subject_samples, self.args.n_gpu), cuda(labels, self.args.n_gpu)
    return neg_object_samples.long(), neg_subject_samples.long(), labels

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

def dilate_input(input_list, dilate_len):
    dilate_temp = []
    dilate_input_list = []
    for i in range(len(input_list)):
        if i % dilate_len == 0 and i:
            if len(dilate_temp):
                dilate_input_list.append(dilate_temp)
                dilate_temp = []
        if len(dilate_temp):
            dilate_temp = np.concatenate((dilate_temp, input_list[i]))
        else:
            dilate_temp = input_list[i]
    dilate_input_list.append(dilate_temp)
    dilate_input_list = [np.unique(_, axis=0) for _ in dilate_input_list]
    return dilate_input_list

def emb_norm(emb, epo=0.00001):
    x_norm = torch.sqrt(torch.sum(emb.pow(2), dim=1))+epo
    emb = emb/x_norm.view(-1,1)
    return emb

def shuffle(data, labels):
    shuffle_idx = np.arange(len(data))
    np.random.shuffle(shuffle_idx)
    relabel_output = data[shuffle_idx]
    labels = labels[shuffle_idx]
    return relabel_output, labels


def cuda(tensor):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor


def soft_max(z):
    t = np.exp(z)
    a = np.exp(z) / np.sum(t)
    return a


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_max = float('-inf')  # 初始化为负无穷，表示得分越高越好

    def __call__(self, val_score, model):
        # 监控验证集指标（得分越高越好）
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(val_score, model)
        elif val_score < self.best_score + self.delta:  # 如果得分没有显著提升
            self.counter += 1
            # self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:  # 如果得分有显著提升
            self.best_score = val_score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        """
        保存验证集得分最高的模型。
        """
        if self.verbose:
            self.trace_func(f"Validation score increased ({self.val_score_max:.6f} --> {val_score:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_score_max = val_score