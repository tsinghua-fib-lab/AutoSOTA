import torch
from torch_geometric.datasets import Planetoid,Reddit
import networkx as nx
import numpy as np
from collections import deque
from scipy.sparse import eye, coo_matrix,triu
from torch_geometric.datasets import CitationFull,Coauthor
import argparse
import time
import copy

torch.manual_seed(42)
import heapq

class nodes:
    def __init__(self, index):
        self.index = index
        self.vanished = 0
        self.edgenode = 0
        self.nodes = []
        self.ed_van = []
        self.train_node = False
        self.remain = False
        self.recast = 0
        self.label = -1

    def __lt__(self, other):
        if self.edgenode < other.edgenode:
            return True
        else:
            if self.edgenode == other.edgenode and self.recast < other.recast:
                return True
            if self.edgenode == other.edgenode and self.recast == other.recast and self.vanished < other.vanished:
                return True
            return False

    def __gt__(self, other):
        if self.edgenode > other.edgenode:
            return True
        else:
            if self.edgenode == other.edgenode and self.recast > other.recast:
                return True
            if self.edgenode == other.edgenode and self.recast == other.recast and self.vanished > other.vanished:
                return True
            return False

    def __eq__(self, other):
        if self.edgenode == other.edgenode and self.recast == other.recast and self.vanished == other.vanished:
            return True
        else:
            return False

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def find_component(data):
    n = []
    graph = nx.Graph()
    for i in range(len(data['x'])):
        graph.add_node(i)
        n.append(nodes(i))
        if keep_mask[i] == 1:
            n[i].train_node = True
            n[i].remain = True
    for i in range(len(data['edge_index'][0])):
        n1n1 = int(data['edge_index'][0, i])
        n2n2 = int(data['edge_index'][1, i])
        if n1n1 != n2n2:
            graph.add_edge(n1n1, n2n2)
    cnt_cluster_node = 0
    num_nodes = data['x'].size()[0]
    n_vanished = 0

    print("time:", time.time_ns())
    com_nodes = []
    components = nx.connected_components(graph)
    print(components)
    for component in components:
        com = list(component)
        if len(com) >= 10:
            com_nodes = com_nodes + com
        else:
            add_flag = 0
            for node in com:
                if n[node].remain:
                    add_flag = 1
                    break
            if add_flag == 1:
                com_nodes = com_nodes + com
            else:
                for node in com:
                    n[node].vanished = True
                n_vanished += len(com)
    return com_nodes

def build_sparse_adjacency_matrix(edge_list, num_nodes):
    rows = edge_list[0]
    cols = edge_list[1]
    return coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_nodes, num_nodes))

from PyGdataset import PygNodePropPredDataset,Evaluator

class CoreAlgorithm:
    def __init__(self, M, data,keep_nodes, reduction_ratio,node_label,args,save = True):
        self.data = data
        self.M = M
        # Build list of sets
        self.list_of_set = []
        for i in range(M.shape[0]):
            self.list_of_set.append(set(M.rows[i]))
        self.args = args
        self.num_delete_hetero = args.del_edge
        self.insert_dominated_edge_degree_limit = args.deg2
        self.degree_threshold = args.deg1

        self.keep_nodes = keep_nodes
        self.reduction_ratio = reduction_ratio
        self.reduction_object = int(self.M.shape[0] * (reduction_ratio))
        self.deleted_to_remain = dict(zip(keep_nodes, keep_nodes))
        self.num_remain_nodes = len(self.keep_nodes)
        self.node_degree = np.array([M[i].nnz for i in range(M.shape[0])])
        self.deleted_node = np.full(M.shape[0], fill_value=True, dtype=bool)
        self.dataname = args.dataname
        for i in keep_nodes:
            self.deleted_node[i] = False
        self.degree_threshold_limit = 0.01*self.M.shape[0]

        self.edges = set()
        self.node_degree = node_degree
        self.need_delete = deque([i for i in range(M.shape[0]) if self.deleted_node[i]])
        self.edge_queue = deque()
        self.round = 0
        self.finish = False
        self.insert_dominated_edge = False
        self.keep_mask = ~(copy.deepcopy(self.deleted_node))

        self.heruistic_delete = False
        self.label = node_label.numpy()
        self.heruistic_delete_deg3 = False
        self.save = save
        self.remain_edges = set()

        edges = self.get_all_edges()
        for e in edges:
            self.remain_edges.add(e)

        self.old_to_new = dict()
        self.new_to_old = dict()
        try:
            self.test_mask = data.test_mask
        except:
            self.test_mask = [False for i in range(len(node_label))]
            pass
        self.test_mask = self.test_mask.numpy()
        self.train_mask = data.train_mask.numpy()
        self.val_mask = data.val_mask.numpy()
        self.label_list = [[node_label[i]] if self.test_mask[i] == False else [] for i in range(len(node_label))]
        self.strong_relaxed = False

        self.finish1 = False
        self.finish2 = False
        self.potential_node = set()
        self.potential_edge = set()
        self.isfirst = True
        self.finish_reduction = False
        self.strong_tolerance = 0

        self.node_bloom_filters = {}

    def delete_heterophlic_edge(self,obj):
        obj_num = obj
        node_label = dict()

        for key,value in enumerate(self.label_list):
            values, counts = np.unique(value, return_counts=True)
            if len(values) == 0:
                node_label[key] = -1
            else:
                # Find the most frequent element
                max_label = values[np.argmax(counts)]
                node_label[key] = int(max_label)

        self.supernode_label1 = node_label

        count = 0
        homo = 0

        for e in self.remain_edges:
            if node_label[e[0]] != -1 and node_label[e[1]] != -1:
                count += 1
                if node_label[e[0]] == node_label[e[1]] :
                        homo += 1

        # if count > 0:
        #     print(f"homo edge: {homo/count}")
        # else:
        #     print("homo edge: N/A (no labeled edges)")

        delete = 0
        new_set = self.remain_edges.copy()
        for e in self.remain_edges:
            if  node_label[e[0]] != node_label[e[1]] and  node_label[e[0]] != -1 and node_label[e[1]] != -1:
                if delete >= obj_num:
                    break
                self.list_of_set[e[0]].remove(e[1])
                self.list_of_set[e[1]].remove(e[0])
                new_set.remove(e)
                delete +=1
                self.node_degree[e[0]] -= 1
                self.node_degree[e[1]] -= 1
        self.remain_edges=new_set
        print(f"delete {delete} edges")

        homo = 0
        count = 0

        for e in  self.remain_edges:
            if node_label[e[0]] != -1 and node_label[e[1]] != -1:
                count += 1
                if node_label[e[0]] == node_label[e[1]] :
                        homo += 1

        if count > 0:
            print(f"homo edge: {homo/count}")
        else:
            print("homo edge: N/A (no labeled edges)")
        self.homo_ratio = homo/count
        return

    

    def sorted_list_intersection_with_deleted_set(self, list1, list2,e):
        intersection = np.empty(len(list1),dtype=int)
        count = 0
        n, m = len(list1), len(list2)
        # Traverse two lists with two pointers
        short_list = list1 if n<m else list2
        w = e[1] if n<m else e[0]
        for node in short_list:
            if self.deleted_node[node]:
                continue
            if node in self.list_of_set[w]:
                intersection[count] = node
                count += 1
        return intersection[:count]

    def is_dominated_edge_set(self,e):
        NG_e0 = self.list_of_set[e[0]]
        NG_e1 = self.list_of_set[e[1]]
        NG_e = self.sorted_list_intersection_with_deleted_set(NG_e0, NG_e1,e)
        NG_e = [n for n in NG_e if n not in e ]

        if len(NG_e) == 0:
            return False
        if len(NG_e) == 1:
            return True
        is_dominated = True
        max_degree_node = max(NG_e, key=lambda n: self.node_degree[n])
        if self.node_degree[max_degree_node] < len(NG_e):
            return False
        for w in NG_e:
            is_dominated=True
            for n in NG_e:
                if w==n:
                    continue
                if w not in self.list_of_set[n]:
                    is_dominated = False
                    break

            if is_dominated == False:
                continue
            else:
                is_dominated = True
                return True
        if is_dominated == False:
            return False

    def is_dominated_edge_return_node(self,e):
        NG_e0 = self.list_of_set[e[0]]
        NG_e1 = self.list_of_set[e[1]]
        NG_e = self.sorted_list_intersection_with_deleted_set(NG_e0, NG_e1,e)
        NG_e = [n for n in NG_e if n not in e ]
        if len(NG_e) == 0:
            return False,0
        if len(NG_e) == 1:
            return True,NG_e[0]
        is_dominated = True
        max_degree_node = max(NG_e, key=lambda n: self.node_degree[n])
        if self.node_degree[max_degree_node] < len(NG_e):
            return False,0
        for w in NG_e:
            is_dominated=True
            for n in NG_e:
                if w not in self.list_of_set[n]:
                    is_dominated = False
                    break

            if is_dominated == False:
                continue
            else:
                is_dominated = True
                return True,w
        if is_dominated == False:
            return False,0

    def is_subset_with_deleted_set_Tolerance(self, sorted_list1, sorted_list2,w,tolerance):
        count = 0
        for n_list1 in sorted_list1:
            if self.deleted_node[n_list1] == False and n_list1 not in self.list_of_set[w]:
                count += 1
                if count > tolerance:
                    return False
        return True

    def strong_collapse(self):
        st = time.time()
        pushed_nodes = set(list(self.node_queue))
        collapse_count = dict()
        st_time = time.time()

        set_degree_flag = False

        last_num_nodes = self.num_remain_nodes
        print(f"round {self.round} strong collapse, node_num {len(self.node_queue)}")
        while self.node_queue:
            self.round += 1
            if self.round % 10000 == 0:
                ed = time.time()
                print("round {} strong collapse: {}".format(round,ed-st))
                st = ed
                print(len(self.node_queue))

            v = self.node_queue.popleft()
            pushed_nodes.remove(v)
            if self.isfirst == False and self.heruistic_delete == False and v not  in self.potential_edge:
                continue
            if  self.node_degree[v]>self.degree_threshold:
                continue

            if self.deleted_node[v]:
                continue

            NG_v = self.list_of_set[v]

            same_label_nodes = []
            other_nodes = []

            # Group nodes by label matching in a single pass
            for node in NG_v:
                if (not self.test_mask[node]) and (self.label[node] == self.label[v]):
                    same_label_nodes.append(node)
                else:
                    other_nodes.append(node)

            sorted_other_node = same_label_nodes + other_nodes

            for w in sorted_other_node:
                if w == v or self.deleted_node[w] or (self.node_degree[w]+self.strong_tolerance) < self.node_degree[v]:
                    continue
                NG_w = self.list_of_set[w]

                if self.is_subset_with_deleted_set_Tolerance(NG_v, NG_w,w,self.strong_tolerance):
                    self.deleted_node[v] = True
                    self.node_degree[v] = 1
                    self.deleted_to_remain[v] = w

                    self.label_list[w]+=self.label_list[v]
                    self.num_remain_nodes -= 1
                    if self.num_remain_nodes <= self.reduction_object:
                        self.finish_reduction = True
                        return

                    for neighbor in NG_v:
                        if not self.deleted_node[neighbor]:
                            self.node_degree[neighbor] -= 1

                            if self.isfirst == False:
                                self.potential_node.add(neighbor)
                            if neighbor not in pushed_nodes:
                                pushed_nodes.add(neighbor)
                                self.node_queue.append(neighbor)
                    break

        self.potential_edge = set()
        num_remain_nodes = self.num_remain_nodes
        ed_time = time.time()
        print(f"round {self.round} strong collapse: {ed_time-st_time}s, reduce_nodes total {len((self.keep_nodes))-self.num_remain_nodes}, reduce nodes this round {last_num_nodes-num_remain_nodes} remain_nodes {self.num_remain_nodes}")
        if last_num_nodes - num_remain_nodes == 0:
            self.insert_dominated_edge = True

        if self.heruistic_delete:
            if last_num_nodes - num_remain_nodes < self.degree_threshold_limit:
                self.strong_tolerance+=1
                self.degree_threshold += 1
                print(f"strong tolerance +1 {self.strong_tolerance}")
            return

        else:
            self.finish	 = False
            return

    def edge_collapse(self):
        t0 = time.time()
        edges = list(self.remain_edges)
        self.edge_queue = deque(edges)
        pushed_edges = set(self.edge_queue)
        round = 0
        intersection_time = 0
        st = time.time()
        count = dict()
        dominate_count = dict()
        deleted_edges = 0

        while self.edge_queue:
            round += 1
            if round % 1000000 == 0:
                ed = time.time()
                print("round {} edge collapse: {} intersection time {} delete edge {}".format(round,ed-st,intersection_time,deleted_edges))
                intersection_time = 0
                st = ed

            e = self.edge_queue.popleft()
            if self.deleted_node[e[0]] or self.deleted_node[e[1]]:
                continue
            if e not in self.remain_edges:
                continue
            if self.node_degree[e[0]] + self.node_degree[e[1]] >2*self.degree_threshold:
                continue
            if self.isfirst ==False:
                if e[0] not in self.potential_node or e[1] not in self.potential_node:
                    continue
            try:
                pushed_edges.remove(e)
            except:
                pushed_edges.remove((e[1], e[0]))
            if e[0] == e[1]:
                continue
            NG_e0 = self.list_of_set[e[0]]
            NG_e1 = self.list_of_set[e[1]]

            t1 = time.time()
            is_dominated = self.is_dominated_edge_set(e)
            if self.label[e[0]] == self.label[e[1]]:
                continue
            if is_dominated:
                self.list_of_set[e[0]].remove(e[1])
                self.list_of_set[e[1]].remove(e[0])
                self.node_degree[e[0]] -= 1
                self.node_degree[e[1]] -= 1
                self.remain_edges.remove(e)
                deleted_edges += 1
                for neighbor_e in NG_e0:
                    if e[0]!=e[1] and self.deleted_node[neighbor_e]==False and e[0] in self.list_of_set[neighbor_e] and (e[0], neighbor_e) not in pushed_edges and (neighbor_e,e[0])  not in pushed_edges:
                        self.edge_queue.append((e[0], neighbor_e))
                        pushed_edges.add((e[0], neighbor_e))
                        self.potential_edge.add(neighbor_e)
                for neighbor_e in NG_e1:
                    if e[0] != e[1] and self.deleted_node[neighbor_e]==False  and e[1] in self.list_of_set[neighbor_e]  and (e[1], neighbor_e) not in pushed_edges and (neighbor_e,e[1]) not in pushed_edges:
                        self.edge_queue.append((e[1], neighbor_e))
                        pushed_edges.add((e[1], neighbor_e))
                        self.potential_edge.add(neighbor_e)
        self.potential_node = set()
        print(count)
        print(dominate_count)
        t1 = time.time()
        print("end edge collapse,  delete edges, {}, time{} ".format(deleted_edges,t1-t0))

    def insert_dominated_edges_2(self):
        delete_node = False
        st = time.time()
        last_num_nodes = self.num_remain_nodes
        print("insert dominated edges")
        vertex_queue = []
        check_num = 0
        reduce_num = 0
        has_checked = set()
        for k in self.keep_nodes:
            if self.node_degree[k] >2 :
                heapq.heappush(vertex_queue, (self.node_degree[k], k))
        degree = dict()
        round =0
        while vertex_queue:
            deg_k,k = heapq.heappop(vertex_queue)
            if deg_k != self.node_degree[k] or k  in has_checked or self.deleted_node[k]:
                continue

            if self.node_degree[k] <=2 or self.node_degree[k] > self.insert_dominated_edge_degree_limit:
                continue
            NG_row = self.list_of_set[k]
            check_num +=1
            other_node = [n for n in NG_row if n != k and self.deleted_node[n] == False]

            other_node = np.array(other_node, dtype=np.int32)
            # Get labels
            labels = self.label[other_node]
            target_label = self.label[k]
            mask = np.logical_not(self.test_mask[other_node])

            # Condition matching: test_mask is False and label matches
            same_label_mask = (labels == target_label) & mask
            same_label_nodes = other_node[same_label_mask]
            other_nodes = other_node[~same_label_mask]
            other_node = np.concatenate([same_label_nodes, other_nodes])

            flag = False
            round +=1
            if round % 10000 == 0:
                print("round {} insertDE,check num: {}, reduce num: {} rate: {}".format(round,check_num,reduce_num,reduce_num/check_num))
                check_num =0
                reduce_num = 0
            for i in range(len(other_node)):
                add_edge_np = np.empty((200, 2), dtype=np.int32)
                current_index = 0

                dominate_flag = True
                for j in range(len(other_node)):
                    if j == i:
                        continue
                    if other_node[j] not in self.list_of_set[other_node[i]]:
                        is_dominated,dominating_edge_node = self.is_dominated_edge_return_node((other_node[i],other_node[j]))
                        if not is_dominated:
                            dominate_flag = False
                            break
                        else:
                            add_edge_np[current_index] = [other_node[i], other_node[j]]
                            current_index += 1

                if dominate_flag == False:
                    continue
                else:
                    flag = True
                    dominating_node = other_node[i]
                    break
            if flag==False:
                has_checked.add(k)
                continue
            else:
                reduce_num +=1
                if k in has_checked:
                    print(f"has checked {k}")
                add_edge = add_edge_np[:current_index]
                if self.node_degree[k] not in  degree:
                    degree[self.node_degree[k]] = 1
                else:
                    degree[self.node_degree[k]] += 1
                delete_node = True
                for e in add_edge:
                    self.list_of_set[e[0]].add(e[1])
                    self.list_of_set[e[1]].add(e[0])
                    self.remain_edges.add((e[0],e[1]))
                    self.node_degree[e[0]] += 1
                    self.node_degree[e[1]] += 1

                self.deleted_node[k] = True
                self.node_degree[k] = 1
                self.num_remain_nodes -= 1

                if self.label[k] == self.label[other_node[0] ]:
                    self.deleted_to_remain[k] = other_node[0]
                    self.label_list[other_node[0]]+=self.label_list[k]
                else:
                    self.deleted_to_remain[k] = other_node[1]
                    self.label_list[other_node[1]]+=self.label_list[k]

                if self.num_remain_nodes <= self.reduction_object:
                    self.finish_reduction = True
                    return

                for neighbor in NG_row:
                    self.node_degree[neighbor] -= 1
                    heapq.heappush(vertex_queue, (self.node_degree[neighbor], neighbor))

                heapq.heappush(vertex_queue, (self.node_degree[dominating_node], dominating_node))

                if self.num_remain_nodes <= self.reduction_object:
                    self.finish = True
                    return

        print(degree)
        ed = time.time()
        self.heruistic_delete = True
        self.node_queue = deque(list(set(self.node_queue)))
        print("end insert dominated node,  node remain {} delete {} this round , time cost {}".format(self.num_remain_nodes,last_num_nodes-self.num_remain_nodes,ed-st))

        if delete_node == False:
            self.finish1 = True

        return

    def find_root(self, map, elem):
        # Find the root element that points to itself
        while map[elem] != elem:
            elem = map[elem]
        return elem

    def make_coarsened_graph_old(self):
        old_to_new = {}
        res = copy.deepcopy(self.data)

        for elem in self.deleted_to_remain.keys():
            root = self.find_root(self.deleted_to_remain, elem)
            self.deleted_to_remain[elem] = root

        edge_list = []
        new_node_num = sum(self.deleted_node == False)
        res['new_x'] = torch.zeros(new_node_num, self.data['x'].shape[1])
        index = 0

        # Map remaining nodes to new node indices
        for i, v in enumerate(self.deleted_node):
            if v == False and self.keep_mask[i] == True:
                old_to_new[i] = index
                index += 1

        # Map deleted nodes to their final retained node indices
        for i, v in enumerate(self.deleted_node):
            if v == True and self.keep_mask[i] == True:
                old_to_new[i] = old_to_new[self.deleted_to_remain[i]]

        # Create mapping from new nodes to old nodes
        new_to_old = {}
        for k, v in old_to_new.items():
            if v not in new_to_old.keys():
                new_to_old[v] = [k]
            else:
                new_to_old[v].append(k)

        # Add edges to the new graph
        for i in range(len(self.list_of_set)):
            if self.deleted_node[i] == False:
                for j in self.list_of_set[i]:
                    if i != j and self.deleted_node[j] == False:
                        edge_list.append([old_to_new[i], old_to_new[j]])

        edge_list = np.array(edge_list).T

        # Compute new graph node features as average of old node features
        for v in new_to_old.keys():
            res['new_x'][v] = torch.max(self.data['x'][new_to_old[v]], dim=0)[0]

        # Reset irrelevant data
        res['x'], res['y'], res['edge_index'], res['train_mask'], res['val_mask'], res['test_mask'] = 0, 0, 0, 0, 0, 0

        res['new_edge_index'] = torch.tensor(edge_list, dtype=torch.long)

        old_to_new = dict(sorted(old_to_new.items(), key=lambda item: item[1]))

        supernode_label = dict()
        label_list = dict()

        total_label_count = 0
        for key,value in new_to_old.items():
            value_filtered = [v for v in value if not self.test_mask[v]]
            label_list[key] = self.label[value_filtered]
            total_label_count += len(label_list[key])
        print(f"total label {total_label_count}")

        for key,value in label_list.items():
            if len(value) == 0:
                supernode_label[key] = 0
                continue
            values, counts = np.unique(value, return_counts=True)
            # Find the most frequent element
            max_label = values[np.argmax(counts)]
            supernode_label[key] = max_label

        mis_label_count = 0
        for key,value in label_list.items():
            for label in value:
                if label != supernode_label[key]:
                    mis_label_count += 1

        print(f"mis percentage {mis_label_count/total_label_count}")

        res['node_label'] = supernode_label
        res['test_mask'] = self.test_mask
        res['train_mask'] = self.train_mask
        res['val_mask'] = self.val_mask
        # Save results
        import os
        save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'coarsened_graph')
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f'{self.dataname}_{self.reduction_ratio:.2f}.npy')
        arr = np.empty(2, dtype=object)
        arr[0] = res.cpu()
        arr[1] = old_to_new
        np.save(filename, arr, allow_pickle=True)
        print(f"Saved to {filename}")
        f = open("./log.txt",'a')
        f.write(filename)
        f.close()

    def get_all_edges(self, inverse=False):
        upper_triangle = triu(self.M, k=1)
        rows, cols = upper_triangle.nonzero()
        if not inverse:
            return list(zip(rows, cols))
        else:
            # Include bidirectional edges
            edges = list(zip(rows, cols))
            reverse_edges = list(zip(cols, rows))
            return edges + reverse_edges

    def run_algorithm_relaxed_strong_collapse(self):
        is_first_strong_relaxed = True
        index = 0
        while  self.finish == False or  self.finish1 == False or self.finish2 == False:
            index += 1
            self.node_queue = deque([n for n in self.keep_nodes if not self.deleted_node[n]])
            self.strong_collapse()

            if self.finish_reduction == True:
                break
            else:
                self.edge_collapse()
                self.isfirst = False

            if self.insert_dominated_edge==True and self.heruistic_delete==False:
                self.insert_dominated_edges_2()
                print(f"insert dominated edge remain {self.num_remain_nodes}")

                self.strong_relaxed = True
                self.exact_iter = index
                index = 0
                if self.finish_reduction == True:
                    break

        set_remain_edge = 0
        for id in range(len(self.list_of_set)):
            if self.deleted_node[id] == False:
                for node in self.list_of_set[id]:
                    if self.deleted_node[node] == False:
                        set_remain_edge += 1
        set_remain_edge = set_remain_edge / 2
        print(f"set remain edge {set_remain_edge}")
        new_set = self.remain_edges.copy()

        for e in self.remain_edges:
            if self.deleted_node[e[0]] or self.deleted_node[e[1]]:
                new_set.remove(e)
        self.remain_edges = new_set
        ori_remain_edge = self.remain_edges.copy()
        ori_list_of_set = copy.deepcopy(self.list_of_set)

        self.remain_edges = ori_remain_edge.copy()
        self.list_of_set = copy.deepcopy(ori_list_of_set)
        self.delete_heterophlic_edge(self.num_delete_hetero )
        if self.save == True:
            self.relax_iter = index
            self.make_coarsened_graph_old()
if __name__ == "__main__":

    para_dict_num_edge_01 = {'Cora':{0.5:[25],0.3:[15],0.2:[15],0.1:[15]},
                               "Citeseer":{0.5:[15],0.3:[15],0.2:[15],0.1:[15]},
                                  'dblp':{0.5:[10],0.3:[10],0.2:[10],0.1:[25]},
                               'ogbn-arxiv':{0.5:[25],0.3:[50],0.2:[50],0.1:[50]},
                               'ogbn-products':{0.5:[100],0.3:[100],0.2:[100],0.1:[100]}}
    DELETE_EDGE_DATASETS = {'Cora', 'Citeseer', 'ogbn-arxiv'}

    para_dict_APPNP = {'Cora':{0.5:[15,15,600],0.3:[15,15,2000],0.2:[15,15,200],0.1:[15,15,1000]},"Citeseer":{0.5:[15,15,900],0.3:[15,15,800],0.2:[15,15,700],0.1:[15,15,2000]}}
    degree_threshold =  {}
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='ogbn-arxiv')
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--deg1', type=int, default=20)
    parser.add_argument('--deg2', type=int, default=20)
    parser.add_argument('--del_edge', type=int, default=0)
    parser.add_argument('--compute_betti', type=bool, default=True)
    args = parser.parse_args()
    dataset_name = args.dataname
    args.deg1 = para_dict_num_edge_01[args.dataname][args.ratio][0]
    args.deg2 = args.deg1  # deg2 always equals deg1
  
    if dataset_name == "Cora":
        data = torch.load('./dataset/Cora/processed/data.pt', weights_only=False)[0]
        edges = data['edge_index']
        label = data.y
    elif dataset_name == "dblp":
        dataset = CitationFull(root='./dataset', name=dataset_name)
        data = dataset[0]
        edges = data['edge_index']
        label = data.y
    elif dataset_name == "Physics":
        dataset = Coauthor(root='./dataset', name=dataset_name)
        data = dataset[0]
        label = data.y
        edges = data['edge_index']
    elif dataset_name == "Citeseer":
        dataset_path = './dataset/Citeseer'
        dataset =  Planetoid(root='./dataset/Citeseer', name='Citeseer')
        data = dataset[0]
        edges = data.edge_index
        label = data.y
    elif dataset_name == "pubmed":
        dataset = Planetoid(root='./dataset/pubmed', name='Pubmed')
        data = dataset[0]
        edges = data['edge_index']
        label = data['y']
    elif dataset_name == "ogbn-arxiv":
        dataset_path = './dataset/arxiv'
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./dataset/arxiv')
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator('ogbn-arxiv')
        data = dataset[0]
        edges = data['edge_index']
        reversed_edges = edges[[1, 0]]
        edges = torch.cat([edges, reversed_edges], dim=1)
    elif dataset_name == 'ogbn-products':
        dataset = PygNodePropPredDataset(name='ogbn-products', root='/mnt/ssd2/products/raw')
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator('ogbn-products')
        data = dataset[0]
        edges = data['edge_index']
        reversed_edges = edges[[1, 0]]
    elif dataset_name == 'reddit':
        dataset = Reddit(root='/mnt/ssd2/Reddit/')
        data = dataset[0]
        edges = data['edge_index']
        label = data['y']

    if dataset_name == 'dblp' or dataset_name == 'Physics':
        indices = []
        num_classes = torch.unique(data.y, return_counts=True)[0].shape[0]
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        train_index = torch.cat([i[:int(len(i)*0.7)] for i in indices], dim=0)
        val_index = torch.cat([i[int(len(i)*0.7):int(len(i)*0.8)] for i in indices], dim=0)
        test_index = torch.cat([i[int(len(i)*0.8):] for i in indices], dim=0)
        print(data.num_nodes)
        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(test_index, size=data.num_nodes)
    indices = []
    if dataset_name == "ogbn-arxiv" or dataset_name == "ogbn-products":
        split_idx = dataset.get_idx_split()
        data.train_mask = index_to_mask(split_idx["train"], size=data.num_nodes)
        data.val_mask = index_to_mask(split_idx["valid"], size=data.num_nodes)
        data.test_mask = index_to_mask(split_idx["test"], size=data.num_nodes)
        data.y= data.y.view(-1)
        label = data.y

    num_classes = torch.unique(data['y'], return_counts=True)[0].shape[0]
    index_train = (data['train_mask'] == 1).nonzero().view(-1)
    for i in range(num_classes):
        index = (data['y'] == i).nonzero().view(-1)
        tensor_isin = torch.isin(index, index_train)
        index = index[tensor_isin]
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    keep_index = torch.cat([i[:int(i.size()[0] * 1.0)] for i in indices], dim=0)
    keep_mask = index_to_mask(keep_index, size=data['x'].size(0))
    node_mask = np.zeros(data['x'].size(0))

    if dataset_name == "ogbn-arxiv" or dataset_name == "reddit"  or dataset_name == "dblp":
        keep_nodes = np.arange(data.num_nodes)
    else:
        keep_nodes = find_component(data)
        print(len(keep_nodes))
    edges = np.array(edges)
    num_nodes = max(edges[0].max(), edges[1].max()) + 1

    adj_matrix_sparse = build_sparse_adjacency_matrix(edges,num_nodes)

    sparse_identity_matrix = eye(num_nodes, format='coo')
    print(f"density {len(edges[0])/num_nodes}")
    adj = adj_matrix_sparse + sparse_identity_matrix
    node_degree = np.diff(adj.indptr)
    adj = adj.tolil()
    print(adj.shape[0])
    adj = adj.astype(bool)

    if dataset_name in DELETE_EDGE_DATASETS:
        from scipy.sparse import triu as sp_triu
        num_undirected_edges = sp_triu(adj_matrix_sparse, k=1).nnz
        args.del_edge = int(num_undirected_edges * 0.1)
        print(f"del_edge = {args.del_edge} (10% of {num_undirected_edges} undirected edges)")
    else:
        args.del_edge = 0

    time_start = time.time()
    args.deg2 = args.deg1
    collapse = CoreAlgorithm(adj,data,keep_nodes,args.ratio,label,args,save = True)
    if args.ratio == 1.0:
        collapse.make_coarsened_graph_old()
    else:
        collapse.run_algorithm_relaxed_strong_collapse()
    time_end = time.time()
    print(f"total time {time_end - time_start}")
    f = open("./log.txt",'a')
    f.write(f"total time {time_end - time_start}\n")
    f.close()
    print("remain nodes :{}".format(collapse.num_remain_nodes))
