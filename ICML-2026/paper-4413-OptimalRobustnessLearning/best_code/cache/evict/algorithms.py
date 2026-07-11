from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Optional, Union, Type
from cache.evict.evictor import *
from cache.evict.predictor import *
import numpy as np
import types
import copy
import random
import math

class EvictAlgorithm(ABC):
    """Evict an entry from one cache line
    
    Max size is associativity
    """
    def __init__(self, associativity) -> None:
        self.cache = [None] * associativity
        self.pcs = [None] * associativity
        self.associativity = associativity
    
    def snapshot(self):
        return list(zip(self.cache, self.pcs))
    
    @abstractmethod
    def access(self, pc, address) -> bool:
        pass

    def boost_access(self, pc, address, boost_pred) -> bool:
        return self.access(pc, address)

class PredictAlgorithm(EvictAlgorithm):
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial]) -> None:
        super().__init__(associativity)
        self.timestamp = 0

        cls_type = predictor_type.func if hasattr(predictor_type, 'func') else predictor_type
        if issubclass(cls_type, ReuseDistancePredictor):
            self.preds = [np.inf] * associativity
        elif issubclass(cls_type, BinaryPredictor):
            self.preds = [0] * associativity
        elif issubclass(cls_type, PhasePredictor):
            self.preds = [1] * associativity
        elif issubclass(cls_type, StatePredictor):
            self.preds = [None] * associativity
        else:
            self.preds = None
        
        if issubclass(cls_type, OraclePredictor):
            def oracle_access(self, pc, address, next_access_time):
                self.predictor.oracle_access(pc, address, next_access_time)
            self.oracle_access = types.MethodType(oracle_access, self)
        
        self.evictor = evictor_type()
        self.predictor = predictor_type()

        self.cur_boost_pred = None
        self.cur_boost_type = None

    def snapshot(self):
        return (list(zip(self.cache, self.pcs)), self.preds)
    
    def before_pred(self, pc, address):
        if self.cur_boost_type is not None and self.cur_boost_type == 'before':
            self.preds = self.cur_boost_pred
        else:
            preds = self.predictor.refresh_scores(self.timestamp, pc, address, self.snapshot()[0])
            if preds is not None:
                self.preds = preds
    
    def after_pred(self, pc ,address, target_index):
        if self.cur_boost_type is not None and self.cur_boost_type == 'after':
            self.preds[target_index] = self.cur_boost_pred
        else:
            pred = self.predictor.predict_score(self.timestamp, pc, address, self.snapshot()[0])
            if pred is not None:
                self.preds[target_index] = pred
        self.timestamp += 1
    
    def boost_access(self, pc, address, boost_pred):
        self.cur_boost_pred = boost_pred
        if self.cur_boost_type is None:
            if isinstance(boost_pred, list):
                self.cur_boost_type = 'before'
            else:
                self.cur_boost_type = 'after'
        return self.access(pc, address)

    def access(self, pc, address):
        target_index = -1
        hit = False

        self.before_pred(pc, address)
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            target_index = self.evictor.evict(list(enumerate(self.preds)))
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.after_pred(pc, address, target_index)
        return hit

######################################################################

class PredictiveMarker(PredictAlgorithm):
    """
    PredictiveMarker algorithm

    Designed by Thodoris Lykouris and Sergei Vassilvitskii. 2018. Competitive Caching with Machine Learned Advice.
    https://dl.acm.org/doi/10.1145/3447579
    """
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial]) -> None:
        def harmonic_number(k):
            return sum(1 / i for i in range(1, k + 1))
        super().__init__(associativity, evictor_type, predictor_type)
        self.marked = [0] * associativity
        self.tracking_set = []
        self.h_k = harmonic_number(associativity)
        self.chains_len = []
        self.chains_rep = []
    
    def access(self, pc, address):
        hit = False
        self.before_pred(pc, address)
        target_index = -1

        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if all(mark == 1 for mark in self.marked):
                # new phase
                self.tracking_set = copy.deepcopy(self.cache)
                self.marked = [0] * self.associativity
            if address not in self.tracking_set:
                target_index = self.evictor.evict([(i, self.preds[i]) for i, mark in enumerate(self.marked) if mark == 0])
                self.chains_len.append(1)
                self.chains_rep.append(self.cache[target_index])
            if address in self.tracking_set:
                index = self.chains_rep.index(address)
                if self.chains_len[index] <= self.h_k:
                    target_index = self.evictor.evict([(i, self.preds[i]) for i, mark in enumerate(self.marked) if mark == 0])
                else:
                    target_index = random.choice([i for i, mark in enumerate(self.marked) if mark == 0])
                self.chains_rep[index] = self.cache[target_index]

        self.cache[target_index], self.pcs[target_index] = address, pc
        self.marked[target_index] = 1
        self.after_pred(pc, address, target_index)
        return hit

class LMarker(PredictAlgorithm):
    """
    LMARKER Algorithm

    Designed by Dhruv Rohatgi. 2020. Near-Optimal Bounds for Online Caching with Machine Learned Advice
    https://epubs.siam.org/doi/10.1137/1.9781611975994.112
    """
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial]) -> None:
        super().__init__(associativity, evictor_type, predictor_type)

        self.stale = []
        self.marked = [0] * associativity
    
    def access(self, pc, address):
        target_index = -1
        hit = False

        self.before_pred(pc, address)
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if all(mark == 1 for mark in self.marked):
                self.stale = copy.deepcopy(self.cache)
                self.marked = [0] * self.associativity
            
            if address in self.stale:
                target_index = random.choice([i for i, mark in enumerate(self.marked) if mark == 0])
            else:
                target_index = self.evictor.evict([(i, self.preds[i]) for i, mark in enumerate(self.marked) if mark == 0])
        
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.marked[target_index] = 1
        self.after_pred(pc, address, target_index)
        return hit

class LNonMarker(PredictAlgorithm):
    """
    LNONMARKER Algorithm

    Designed by Dhruv Rohatgi. 2020. Near-Optimal Bounds for Online Caching with Machine Learned Advice
    https://epubs.siam.org/doi/10.1137/1.9781611975994.112
    """
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial]) -> None:
        super().__init__(associativity, evictor_type, predictor_type)

        self.phase = set()
        self.stale = []
        self.marked = [0] * associativity
        self.evicts = {}
    
    def access(self, pc, address):
        target_index = -1
        hit = False
        self.before_pred(pc, address)

        if len(self.phase) == self.associativity:
            self.stale = copy.deepcopy(self.cache)
            self.marked = [0] * self.associativity
            self.evicts = {}
            self.phase = set()

        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if address in self.stale:
                if self.evicts[address] not in self.stale:
                    target_index = random.choice(range(self.associativity))
                else:
                    target_index = random.choice([i for i, mark in enumerate(self.marked) if mark == 0])
            else:
                target_index = self.evictor.evict([(i, self.preds[i]) for i, mark in enumerate(self.marked) if mark == 0])
        
        self.evicts[self.cache[target_index]] = address
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.marked[target_index] = 1
        self.phase.add(address)
        self.after_pred(pc, address, target_index)
        return hit

class Mark0(PredictAlgorithm):
    """
    MARK0 Eviction Strategy

    Designed by Antonios Antoniadis, Joan Boyar, Marek Eliáš, Lene M. Favrholdt, Ruben Hoeksma, Kim S. Larsen, Adam Polak, and Bertrand Simon. 2023. Paging with Succinct Prediction.
    https://dl.acm.org/doi/10.5555/3618408.3618447
    """
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial]):
        super().__init__(associativity, evictor_type, predictor_type)
        if not isinstance(self.predictor, BinaryPredictor):
            raise ValueError('Mark0: predictor must be a BinaryPredictor')
        self.marked = [0] * associativity
        self.S_address = [None] * associativity
        self.S_visited = [0] * associativity
    
    def access(self, pc, address):
        target_index = -1
        hit = False

        self.before_pred(pc, address)
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            if address in self.S_address and 0 in self.S_visited:
                target_index = random.choice([i for i, visited in enumerate(self.S_visited) if visited == 0])
            else:
                target_index = self.cache.index(None)
        else:
            if all(visited == 1 for visited in self.S_visited):
                self.marked = [0] * self.associativity
                self.S_address = copy.deepcopy(self.cache)
                self.S_visited = [0] * self.associativity

            if address in self.S_address and 0 in self.S_visited:
                target_index = random.choice([i for i, visited in enumerate(self.S_visited) if visited == 0])
            else:
                target_index = random.choice([i for i, mark in enumerate(self.marked) if mark == 0])
        
        self.S_address[target_index] = None
        self.S_visited[target_index] = 1
        self.marked[target_index] = 1
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.after_pred(pc, address, target_index)
        if self.preds[target_index] == 1:
            self.cache[target_index], self.pcs[target_index] = None, None
        return hit

class MarkAndPredict(PredictAlgorithm):
    """
    MARK&PREDICT Eviction Strategy

    Designed by Antonios Antoniadis, Joan Boyar, Marek Eliáš, Lene M. Favrholdt, Ruben Hoeksma, Kim S. Larsen, Adam Polak, and Bertrand Simon. 2023. Paging with Succinct Prediction.
    https://dl.acm.org/doi/10.5555/3618408.3618447
    """
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial]):
        super().__init__(associativity, evictor_type, predictor_type)
        if not isinstance(self.predictor, PhasePredictor):
            raise ValueError('MarkAndPredict: predictor must be a PhasePredictor')
        if not isinstance(self.evictor, BinaryEvictor):
            raise ValueError('MarkAndPredict: evictor must be a BinaryEvictor')
        self.marked = [0] * associativity
    
    def access(self, pc, address):
        target_index = -1
        hit = False

        self.before_pred(pc, address)
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if all(mark == 1 for mark in self.marked):
                self.marked = [0] * self.associativity
            target_index = self.evictor.evict([(i, self.preds[i]) for i, mark in enumerate(self.marked) if mark == 0])
        
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.marked[target_index] = 1
        self.after_pred(pc, address, target_index)
        return hit

class FollowerRobust(PredictAlgorithm):
    """
    F&R Algorithm

    Parameters:

    - a

    - lazy_evictor_type

    Designed by Karim Abdel Sadek and Marek Elias. 2024. Algorithms for Caching and MTS with reduced number of predictions.
    https://arxiv.org/abs/2404.06280
    """
    @staticmethod
    def create_windows(S, W, F, k, a):
        def func(i):    
            return (2**(i+1))-1
        for i in range(0, int(np.log2(k)) + 1):
            S.append((int(k - (k // (2 ** i)) + 1)))
        for i in range(1, int(np.log2(k)) + 1):
            n = []
            for h in range(S[i - 1], S[i]):
                n.append(h)
            W.append(n)
        W.append([S[-1]])
        for g in range(0, len(W)-1):
            gap = int(len(W[g])//(func(g+1)-func(g)))
            if (gap >= a):
                for m in W[g][::gap]:
                    F.append(m)
            else:
                for m in range(S[g], S[-1]+1, a):
                    F.append(m)
                break
        if k == 10:
            F = [1,6,9]
        return S, W, F

    @staticmethod
    def differ(a, b):
        aa = list(a).copy()
        bb = list(b).copy()
        for x in bb:
            if x == None:
                continue
            elif x in aa:
                aa.remove(x)
        if aa == []:
            return bb
        return aa

    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial], **kwargs):
        super().__init__(associativity, evictor_type, predictor_type)
        if not isinstance(self.predictor, StatePredictor):
            raise ValueError('FollowerRobust: predictor must be a StatePredictor')

        if 'boost' in kwargs:
            self.boost = kwargs['boost']
        else:
            self.boost = False
        self.boost_beladys = []
        self.online_belady_cache = [None] * associativity
        self.online_belady_dis = [np.inf] * associativity
        self.boost_beladys.append(copy.deepcopy(self.online_belady_cache))
        if self.boost:
            def oracle_access(self, pc, address, next_access_time):
                if address in self.online_belady_cache:
                    target_index = self.online_belady_cache.index(address)
                elif None in self.online_belady_cache:
                    target_index = self.online_belady_cache.index(None)
                else:
                    target_index = self.online_belady_dis.index(max(self.online_belady_dis))
                
                self.online_belady_cache[target_index] = address
                self.online_belady_dis[target_index] = next_access_time
                self.boost_beladys.append(copy.deepcopy(self.online_belady_cache))
                if hasattr(self.predictor, 'oracle_access'):
                    self.predictor.oracle_access(pc, address, next_access_time)
            self.oracle_access = types.MethodType(oracle_access, self)

        if 'a' in kwargs:
            self.a = kwargs['a']
        else:
            self.a = 1
        if 'lazy_evictor_type' in kwargs:
            if kwargs['lazy_evictor_type'] is None:
                self.lazy_evictor = None
            else:
                self.lazy_evictor = kwargs['lazy_evictor_type']()
        else:
            self.lazy_evictor = LRUEvictor()
        self.key_scores = [np.inf] * self.associativity if self.lazy_evictor is not None else None
        self.sim_cache = [None] * associativity
        self.sim_pcs = [None] * associativity
        self.traces = []
        
        self.S = []
        self.W = []
        self.F = []
        if (self.a == 1):
            self.S, self.W, self.F = FollowerRobust.create_windows(self.S, self.W, self.F, self.associativity, self.a)
        self.skip = 0
        self.pred_gap = 0
        self.follow_cost = 0
        self.belady_cost = 0
        self.marked = []
        self.old = []
        self.unmarked = []
        self.unmarked_for_reload = []
        self.clean = []
        self.prediction = [None] * self.associativity

    def online_belady(self):
        if self.boost:
            return self.boost_beladys[self.timestamp]
        else:
            cache = []
            for i, current in enumerate(self.traces):
                if current in cache:
                    continue
                if len(cache) < self.associativity:
                    cache.append(current)
                else:
                    future_uses = {item: self.traces[i + 1:].index(item) if item in self.traces[i + 1:] else float('inf') for item in cache}
                    to_remove = max(future_uses, key=future_uses.get)
                    cache.remove(to_remove)
                    cache.append(current)
            return cache
    
    def follow_robust(self, pc, address):
        target_index = -1
        # get next state
        if self.cur_boost_type is not None and self.cur_boost_type == 'before':
            preds = self.cur_boost_pred
        else:
            preds = self.predictor.refresh_scores(self.timestamp, pc, address, self.snapshot()[0])
        assert(preds is not None)
        f = copy.deepcopy(self.online_belady())
        if address in self.sim_cache:
            target_index = self.sim_cache.index(address)
            self.sim_cache[target_index] = address
        elif None in self.sim_cache:
            index_to_evict = self.sim_cache.index(None)
            self.sim_cache[index_to_evict] = address
            self.prediction = copy.deepcopy(preds)
        if address not in self.sim_cache:
            target_index = None
            if self.skip == 0:
                self.follow_cost += 1
                if address not in f:
                    self.belady_cost +=1
                if address not in self.prediction and (self.follow_cost <= self.belady_cost):
                    if self.pred_gap <= 0:
                        self.prediction = preds
                        self.pred_gap = self.a
                        dd = self.differ(self.sim_cache, self.prediction)
                        target_index = self.sim_cache.index(random.choice(dd))
                        assert(self.sim_cache[target_index] not in self.prediction)
                        self.sim_cache[target_index] = address
                    else:
                        target_index = random.choice(range(self.associativity))
                        self.sim_cache[target_index] = address
                elif address in self.prediction:
                    dd = self.differ(self.sim_cache, self.prediction)
                    target_index = self.sim_cache.index(random.choice(dd))
                    self.sim_cache[target_index] = address
                else:
                    self.follow_cost = 0
                    self.belady_cost = 0
                    self.skip = self.associativity
                    self.old = []
                    for req in self.traces[self.timestamp-1::-1]:
                        if (req not in self.old) and (req != address):
                            self.old.append(req)
                        if len(self.old) >= self.associativity:
                            break
                    assert(len(self.old)==self.associativity)
                    self.unmarked = self.old.copy()
                    self.sim_cache = self.old.copy()
                    assert(address not in self.sim_cache)
                    self.marked = []
                    self.unmarked_for_reload = []
                    self.clean = []
            if self.skip != 0:
                assert(address not in self.sim_cache)
                if address not in self.marked:
                    self.skip -= 1
                    arrival_no = self.associativity-self.skip
                    if address in self.unmarked:
                        self.unmarked.remove(address)
                    if address not in self.marked:
                        self.marked.append(address)
                    assert(len(self.marked) == arrival_no)
                    if address not in self.old:
                        self.clean.append(address)
                    assert(len(self.unmarked) == self.associativity - (arrival_no - len(self.clean)))
                    if ((self.a==1) and (arrival_no in self.F)) or ((self.a > 1) and (self.pred_gap <= 0)):
                        self.pred_gap = self.a
                        self.prediction = copy.deepcopy(preds)
                    if arrival_no in self.S:
                        self.unmarked_for_reload = []
                        for p in self.unmarked:
                            if (p in self.prediction) and (p not in self.sim_cache):
                                self.unmarked_for_reload.append(p)
                    if address in self.unmarked_for_reload:
                        # Lazy sync with predictor
                        assert(address not in self.sim_cache)
                        dd = self.differ(self.sim_cache, self.prediction)
                        target_index = self.sim_cache.index(random.choice(dd))
                        self.sim_cache[target_index] = address
                    if address in self.clean: # Clean arrival
                            assert(address not in self.sim_cache)
                            dd = self.differ(self.sim_cache, self.prediction)
                            target_index = self.sim_cache.index(random.choice(dd))
                            self.sim_cache[target_index] = address
                if address not in self.sim_cache:
                    index_to_evict = None
                    unmarked_slots = []
                    for page in self.sim_cache:
                        if page in self.unmarked:
                            unmarked_slots.append(self.sim_cache.index(page))
                    target_index = random.choice(unmarked_slots)
                    assert(address not in self.sim_cache)
                    self.sim_cache[target_index] = address
                if self.skip == 0:
                    assert(len(self.marked) == self.associativity)
                    assert(len(self.unmarked) == len(self.clean))
        if self.cur_boost_type is not None:
            assert self.cur_boost_type == 'before'
        else:
            pred = self.predictor.predict_score(self.timestamp, pc, address, self.snapshot()[0])
            assert pred is None
        self.pred_gap -= 1
        self.traces.append(address)
    
    def access(self, pc, address):
        self.follow_robust(pc, address)

        ## Lazy
        target_index = -1
        hit = False
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if self.lazy_evictor is None:
                self.cache = copy.deepcopy(self.sim_cache)
                self.pcs = copy.deepcopy(self.sim_pcs)
                target_index = self.cache.index(address)
            else:
                diff_keys = set(self.cache) - set(self.sim_cache)
                target_index = self.lazy_evictor.evict([(self.cache.index(k), self.key_scores[self.cache.index(k)] if self.key_scores is not None else 0) for k in diff_keys])
        
        self.key_scores[target_index] = self.timestamp
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.timestamp += 1
        return hit

class Guard(PredictAlgorithm):
    """
    Guard algorithm

    Parameters:
    
    - follow_if_guarded

    - relax_times

    - relax_prob

    Our work
    """
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial], **kwargs) -> None:
        super().__init__(associativity, evictor_type, predictor_type)
        self.old_unvisited_set = []
        self.unguarded_set = []
        self.phase_evicted_set = set()
        self.error_times = 0

        if 'follow_if_guarded' in kwargs:
            self.follow_if_guarded = kwargs['follow_if_guarded']
        else:
            self.follow_if_guarded = False
        if 'relax_times' in kwargs:
            self.relax_times = kwargs['relax_times']
        else:
            self.relax_times = 0
        if 'relax_prob' in kwargs:
            self.relax_prob = kwargs['relax_prob']
        else:
            self.relax_prob = 0
    
    def access(self, pc, address):
        to_guard = False
        target_index = -1
        hit = False

        self.before_pred(pc, address)
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if not self.old_unvisited_set:
                self.old_unvisited_set = list(range(self.associativity))
                self.unguarded_set = list(range(self.associativity))
                self.phase_evicted_set = set()
                self.error_times = 0
            
            if address in self.phase_evicted_set:
                if self.relax_times != 0:
                    self.error_times += 1
                    if self.error_times >= self.relax_times:
                        to_guard = True
                else:
                    if random.random() > self.relax_prob:
                        to_guard = True

            if to_guard and not self.follow_if_guarded:
                target_index = random.choice(self.old_unvisited_set)
            else:
                target_index = self.evictor.evict([(i, self.preds[i]) for i in self.unguarded_set])
            
            self.phase_evicted_set.add(self.cache[target_index])

        if target_index in self.old_unvisited_set:
            self.old_unvisited_set.remove(target_index)

        if to_guard:
            self.unguarded_set.remove(target_index)
        
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.after_pred(pc, address, target_index)
        return hit
    
class GuardWithoutRandomEvict(PredictAlgorithm):
    """
    Our work
    """
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial], **kwargs) -> None:
        super().__init__(associativity, evictor_type, predictor_type)
        self.old_unvisited_set = []
        self.unguarded_set = []
        self.phase_evicted_set = set()
        self.error_times = 0

        if 'follow_if_guarded' in kwargs:
            self.follow_if_guarded = kwargs['follow_if_guarded']
        else:
            self.follow_if_guarded = False
        if 'relax_times' in kwargs:
            self.relax_times = kwargs['relax_times']
        else:
            self.relax_times = 0
        if 'relax_prob' in kwargs:
            self.relax_prob = kwargs['relax_prob']
        else:
            self.relax_prob = 0
    
    def access(self, pc, address):
        to_guard = False
        target_index = -1
        hit = False

        self.before_pred(pc, address)
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if not self.old_unvisited_set:
                self.old_unvisited_set = list(range(self.associativity))
                self.unguarded_set = list(range(self.associativity))
                self.phase_evicted_set = set()
                self.error_times = 0
            
            if address in self.phase_evicted_set:
                if self.relax_times != 0:
                    self.error_times += 1
                    if self.error_times >= self.relax_times:
                        to_guard = True
                else:
                    if random.random() > self.relax_prob:
                        to_guard = True

            target_index = self.evictor.evict([(i, self.preds[i]) for i in self.unguarded_set])

            self.phase_evicted_set.add(self.cache[target_index])

        if target_index in self.old_unvisited_set:
            self.old_unvisited_set.remove(target_index)

        if to_guard:
            self.unguarded_set.remove(target_index)
        
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.after_pred(pc, address, target_index)
        return hit
    
class GuardWithoutProtection(PredictAlgorithm):
    """
    Our work
    """
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial], **kwargs) -> None:
        super().__init__(associativity, evictor_type, predictor_type)
        self.old_unvisited_set = []
        self.unguarded_set = []
        self.phase_evicted_set = set()
        self.error_times = 0

        if 'follow_if_guarded' in kwargs:
            self.follow_if_guarded = kwargs['follow_if_guarded']
        else:
            self.follow_if_guarded = False
        if 'relax_times' in kwargs:
            self.relax_times = kwargs['relax_times']
        else:
            self.relax_times = 0
        if 'relax_prob' in kwargs:
            self.relax_prob = kwargs['relax_prob']
        else:
            self.relax_prob = 0
    
    def access(self, pc, address):
        target_index = -1
        hit = False

        self.before_pred(pc, address)
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if not self.old_unvisited_set:
                self.old_unvisited_set = list(range(self.associativity))
                self.phase_evicted_set = set()
            
            if address in self.phase_evicted_set:
                target_index = random.choice(self.old_unvisited_set)
            else:
                target_index = self.evictor.evict([(i, self.preds[i]) for i in self.old_unvisited_set])
            
            self.phase_evicted_set.add(self.cache[target_index])

        if target_index in self.old_unvisited_set:
            self.old_unvisited_set.remove(target_index)
        
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.after_pred(pc, address, target_index)
        return hit

#######################################################################

class CombineAlgorithm(EvictAlgorithm):
    def __init__(self, associativity, candidate_algorithms: List[Union[EvictAlgorithm, partial]], lazy_evictor_type: Union[LRUEvictor, RandEvictor, None] = LRUEvictor):
        if lazy_evictor_type is not None and not issubclass(lazy_evictor_type, Evictor):
            raise ValueError('CombineAlgorithm: Invalid Evictor')
        
        super().__init__(associativity)
        self.oracle_algs = []
        self.boost_algs = []
        self.candidate_algs = []
        self.center = 0
        self.timestamp = 0
        self.lazy_evictor = lazy_evictor_type() if lazy_evictor_type is not None else None
        # self.key_scores = {} if lazy_evictor_type == LRUEvictor else None
        self.key_scores = [np.inf] * associativity if lazy_evictor_type == LRUEvictor else None

        for alg_type in candidate_algorithms:
            alg_instance = alg_type(associativity)
            self.candidate_algs.append([alg_instance, 0])
            if hasattr(alg_instance, 'oracle_access'):
                self.oracle_algs.append(alg_instance)
            if hasattr(alg_instance, 'boost_access'):
                self.boost_algs.append(alg_instance)

        if len(self.oracle_algs) != 0:
            def oracle_access(self, pc, address, next_access_time):
                for oracle_alg in self.oracle_algs:
                    oracle_alg.oracle_access(pc, address, next_access_time)
            self.oracle_access = types.MethodType(oracle_access, self)
        
        if len(self.candidate_algs) < 2:
            raise ValueError('CombineAlgorithm: Algorithm Count < 2')

    def __push_candidates__(self, pc, address):
        for i, (alg, _) in enumerate(self.candidate_algs):
            if not alg.access(pc, address):
                self.candidate_algs[i][1] += 1
                self.__trigger_miss__(i, address)
    
    def __push_candidates_boost__(self, pc, address, boost_pred):
        for i, (alg, _) in enumerate(self.candidate_algs):
            if alg in self.boost_algs:
                hit = alg.boost_access(pc, address, boost_pred)
            else:
                hit = alg.access(pc, address)
            if not hit:
                self.candidate_algs[i][1] += 1
                self.__trigger_miss__(i, address)
    
    def __trigger_miss__(self, i, address):
        pass

    @abstractmethod
    def __trigger_elect_center__(self):
        pass

    def __process__(self, pc, address):
        target_index = -1
        hit = False
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            self.__trigger_elect_center__()
            center_cache = self.candidate_algs[self.center][0].cache
            if self.lazy_evictor is None:
                self.cache = copy.deepcopy(center_cache)
                self.pcs = copy.deepcopy(self.candidate_algs[self.center][0].pcs)
                target_index = self.cache.index(address)
            else:
                diff_keys = set(self.cache) - set(center_cache)
                target_index = self.lazy_evictor.evict([(self.cache.index(k), self.key_scores[self.cache.index(k)] if self.key_scores is not None else 0) for k in diff_keys])
        if self.key_scores is not None:
            self.key_scores[target_index] = self.timestamp
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.timestamp += 1
        return hit

    def boost_access(self, pc, address, boost_pred):
        self.__push_candidates_boost__(pc, address, boost_pred)
        return self.__process__(pc, address)

    def access(self, pc, address):
        self.__push_candidates__(pc, address)
        return self.__process__(pc, address)

class CombineDeterministicAlgorithm(CombineAlgorithm):
    """
    black-box algorithm

    Designed by Thodoris Lykouris and Sergei Vassilvitskii. 2018. Competitive Caching with Machine Learned Advice.
    https://dl.acm.org/doi/10.1145/3447579
    """
    def __init__(self, associativity, candidate_algorithms: List[Union[EvictAlgorithm, partial]], switch_bound=2, lazy_evictor_type: Union[LRUEvictor, RandEvictor, None] = LRUEvictor):
        super().__init__(associativity, candidate_algorithms, lazy_evictor_type)
        self.switch_bound = switch_bound

    def __trigger_elect_center__(self):
        this_cost = self.candidate_algs[self.center][1]
        min_center, (_, min_cost) = min(enumerate(self.candidate_algs), key=lambda x: x[1][1])
        if this_cost >= self.switch_bound * min_cost:
            self.center = min_center

class CombineRandomAlgorithm(CombineAlgorithm):
    """
    Algorithm THRESH

    Designed by Avrim Blum and Carl Burch. 1997. On-line learning and the metrical task system problem.
    https://dl.acm.org/doi/10.1145/267460.267475
    """
    def __init__(self, associativity, candidate_algorithms: List[Union[EvictAlgorithm, partial]], alpha=0.0, beta=0.99, lazy_evictor_type: Union[LRUEvictor, RandEvictor, None] = LRUEvictor):
        super().__init__(associativity, candidate_algorithms, lazy_evictor_type)
        self.alpha = alpha
        self.beta = beta
        self.n = len(self.candidate_algs)
        self.weights = [1] * self.n
        self.probs = [1/self.n] * self.n
    
    def __trigger_miss__(self, i, key):
        self.weights[i] *= self.beta
    
    def __trigger_elect_center__(self):
        W = sum(self.weights)
        threshold = self.alpha * W / self.n
        new_probs = [w / W for w in self.weights]
        if new_probs[self.center] < self.probs[self.center]:
            threshold = 1 - new_probs[self.center] / self.probs[self.center]
            if random.random() > threshold:
                self.center = self.center
            else:
                index = list(range(self.n))
                index.remove(self.center)
                probs = copy.deepcopy(new_probs)
                probs.pop(self.center)
                self.center = random.choices(index, weights=probs)[0]
        self.probs = new_probs

        # valid_index, valid_weights = zip(*[(i, weight) for i, weight in enumerate(self.weights) if weight > threshold])
        # if valid_weights:
        #     self.center = random.choices(valid_index, weights=valid_weights)[0]

class CombineWeightsAlgorithm(CombineAlgorithm):
    """
    Imitation learing for Parrot
    """
    def __init__(self, associativity, candidate_algorithms: List[Union[EvictAlgorithm, partial]], weights: Union[List[float], None], lazy_evictor_type: Union[LRUEvictor, RandEvictor, None] = LRUEvictor):
        super().__init__(associativity, candidate_algorithms, lazy_evictor_type)
        self.n = len(self.candidate_algs)
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [1] * self.n
    
    def snapshot(self):
        return (list(zip(self.cache, self.pcs)), self.candidate_algs[self.center][0].preds)

    def reset(self, weights):
        self.weights = weights

    def __trigger_elect_center__(self):
        self.center = random.choices(list(range(self.n)), weights=self.weights)[0]

#######################################################################

class RandAlgorithm(EvictAlgorithm):
    def __init__(self, associativity):
        super().__init__(associativity)
        self.evictor = RandEvictor()
    
    def access(self, pc, address):
        target_index = -1
        hit = False
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            target_index = self.evictor.evict(list(enumerate(self.cache)))
        
        self.cache[target_index] = address
        self.pcs[target_index] = pc
        return hit

class LRUAlgorithm(EvictAlgorithm):
    def __init__(self, associativity):
        super().__init__(associativity)
        self.evictor = LRUEvictor()
        self.scores = [0] * associativity
        self.timestamp = 0
    
    def access(self, pc, address):
        target_index = -1
        hit = False
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            target_index = self.evictor.evict(list(enumerate(self.scores)))
        
        self.cache[target_index] = address
        self.pcs[target_index] = pc
        self.scores[target_index] = self.timestamp
        self.timestamp += 1
        return hit

class MarkerAlgorithm(EvictAlgorithm):
    def __init__(self, associativity):
        super().__init__(associativity)
        self.evictor = MarkerEvictor()
        self.scores = [0] * associativity
    
    def access(self, pc, address):
        if all(x == 1 for x in self.scores):
            self.scores = [0] * self.associativity

        target_index = -1
        hit = False
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            target_index = self.evictor.evict(list(enumerate(self.scores)))
        
        self.cache[target_index] = address
        self.pcs[target_index] = pc
        self.scores[target_index] = 1
        return hit


class OnlineMinAlgorithm(EvictAlgorithm):
    """OnlineMin: A Fast Strongly Competitive Randomized Paging Algorithm.

    This implementation follows the paper's high-level algorithm (Section 3.1):
    - Maintain support layers L1..Lk (Equitable2 update rule with forgiveness).
    - Maintain a random priority (rank) for each page in the support.
    - On a miss, evict the minimum-priority page from a specific prefix of cache
      pages (determined by layers), then update layers.

    Notes for this codebase:
    - The theoretical algorithm starts with k distinct pages already in cache.
      Here we warm up until the set is full, then initialize layers as k
      singleton (revealed) layers in LRU order.
    """

    def __init__(self, associativity: int, max_support_factor: int = 3):
        super().__init__(associativity)
        self.k = associativity
        if max_support_factor < 1:
            raise ValueError('OnlineMin: max_support_factor must be >= 1')
        self.max_support_factor = int(max_support_factor)
        self.max_support = self.max_support_factor * self.k

        # Support layers: index 1..k used; index 0 unused.
        self.layers = [set() for _ in range(self.k + 1)]
        self.support = set()

        # Random priorities (lower = smaller priority). Only for pages in support.
        self.priority = {}
        self._free_priorities = list(range(1, self.max_support + 1))

        # Page -> current layer index (1..k). Only defined for pages in support.
        self.layer_of = {}

        # Warm-up: maintain recency order until cache becomes full.
        self._inited = False
        self._warmup_lru = []  # oldest -> newest

        # Record last access time for pages in support (layers 1..k).
        self.timestamp = 0
        self.support_last_access: Dict[object, int] = {}
        self.support_last_evict: Dict[object, int] = {}

        # Stats: count requests to L0 vs non-L0 (after initialization).
        self.l0_requests = 0
        self.non_l0_requests = 0

    def _assign_priority(self, page):
        if page in self.priority:
            return
        if not self._free_priorities:
            raise ValueError('OnlineMin: priority universe exhausted')
        pr = random.choice(self._free_priorities)
        self._free_priorities.remove(pr)
        self.priority[page] = pr

    def _delete_priority(self, page):
        pr = self.priority.pop(page, None)
        if pr is not None:
            self._free_priorities.append(pr)

    def _rebuild_layer_of(self):
        new_layer_of = {}
        new_support = set()
        for i in range(1, self.k + 1):
            for p in self.layers[i]:
                new_layer_of[p] = i
                new_support.add(p)

        # Reclaim priorities for pages that left the support BEFORE assigning
        # priorities to newly added support pages.
        for p in list(self.priority.keys()):
            if p not in new_support:
                self._delete_priority(p)

        if len(new_support) > self.max_support:
            raise RuntimeError(
                'OnlineMin invariant violated: support size exceeded 3k. '
                f'support_size={len(new_support)} max_support={self.max_support}'
            )

        self.layer_of = new_layer_of
        self.support = new_support

        # Keep last-access records only for pages still in support.
        for p in list(self.support_last_access.keys()):
            if p not in new_support:
                self.support_last_access.pop(p, None)

        # Keep last-evict records only for pages still in support.
        for p in list(self.support_last_evict.keys()):
            if p not in new_support:
                self.support_last_evict.pop(p, None)

        for p in new_support:
            self._assign_priority(p)

    def _init_layers_from_warmup(self):
        # Build k singleton (revealed) layers from warm-up LRU order.
        pages = [p for p in self._warmup_lru if p is not None]
        if len(pages) < self.k:
            return
        pages = pages[-self.k:]
        self.layers = [set() for _ in range(self.k + 1)]
        for i, p in enumerate(pages, start=1):
            self.layers[i] = {p}
            self._assign_priority(p)
        self._rebuild_layer_of()
        self._inited = True

    def _warmup_access(self, pc, address):
        """Handle common pre-initialization cache filling and LRU ordering."""
        if self._inited:
            return False, None, None

        if address in self.cache:
            idx = self.cache.index(address)
            self.pcs[idx] = pc
            if address in self._warmup_lru:
                self._warmup_lru.remove(address)
            self._warmup_lru.append(address)
            return True, True, idx

        if None in self.cache:
            idx = self.cache.index(None)
            self.cache[idx] = address
            self.pcs[idx] = pc
            self._assign_priority(address)
            if address in self._warmup_lru:
                self._warmup_lru.remove(address)
            self._warmup_lru.append(address)
            if None not in self.cache:
                self._init_layers_from_warmup()
            return True, False, idx

        if not self._warmup_lru:
            self._warmup_lru = [p for p in self.cache if p is not None]
        self._init_layers_from_warmup()
        return False, None, None

    def _update_layers_after_request(self, page, layer_i: int, forgiveness: bool):
        # Apply Definition 1 (Equitable2 update rule with forgiveness).
        if layer_i == 0:
            if not forgiveness:
                # (L0\{p}, L1, ..., L_{k-2}, L_{k-1} ∪ L_k, {p})
                if self.k >= 2:
                    self.layers[self.k - 1] |= self.layers[self.k]
                self.layers[self.k] = {page}
            else:
                # (L0\{p} ∪ L1, L2, ..., L_k, {p})  => support loses old L1.
                dropped = self.layers[1]
                for q in dropped:
                    self._delete_priority(q)
                for j in range(1, self.k):
                    self.layers[j] = self.layers[j + 1]
                self.layers[self.k] = {page}
        else:
            i = layer_i
            if i < self.k:
                # (.., L_{i-1} ∪ (L_i \ {p}), L_{i+1}, ..., L_k, {p})
                self.layers[i - 1] |= (self.layers[i] - {page})
                for j in range(i, self.k):
                    self.layers[j] = self.layers[j + 1]
                self.layers[self.k] = {page}
            else:
                # i == k: page already in Lk (revealed); keep singleton.
                self.layers[self.k] = {page}

        self._rebuild_layer_of()

    def access(self, pc, address):
        ts = self.timestamp
        handled, hit, _ = self._warmup_access(pc, address)
        if handled:
            self.support_last_access[address] = ts
            self.timestamp += 1
            return hit

        # OnlineMin proper.
        hit = address in self.cache
        if hit:
            idx = self.cache.index(address)
            self.pcs[idx] = pc

        support_size = len(self.support)
        layer_i = self.layer_of.get(address, 0)

        # Count requests by layer (L0 is implicit as "not in support").
        if layer_i == 0:
            self.l0_requests += 1
        else:
            self.non_l0_requests += 1

        forgiveness = (layer_i == 0 and support_size == self.max_support)
        eviction_layer = 1 if forgiveness else layer_i

        # Fail fast with a clear error instead of crashing with KeyError in
        # `self.layer_of[p]` if invariants were broken earlier.
        cache_set = {p for p in self.cache if p is not None}
        extras = [p for p in cache_set if p not in self.support]
        if extras:
            raise RuntimeError(
                'OnlineMin invariant violated: cache contains non-support pages (pre-evict). '
                f'addr={address} extras={extras} cache={list(self.cache)} support_size={len(self.support)}'
            )

        if not hit:
            if None in self.cache:
                idx = self.cache.index(None)
                self.cache[idx] = address
                self.pcs[idx] = pc
            else:
                cache_pages = [p for p in self.cache if p is not None]
                if eviction_layer == 0:
                    victim = min(cache_pages, key=lambda p: self.priority[p])
                else:
                    # Identify the prefix of
                    # Sort cache pages by increasing layer index.
                    cache_pages.sort(key=lambda p: self.layer_of[p])
                    j = None
                    for jj in range(eviction_layer, self.k + 1):
                        if self.layer_of[cache_pages[jj - 1]] == jj:
                            j = jj
                            break
                    if j is None:
                        # Should not happen; fall back to evicting global min priority.
                        victim = min(cache_pages, key=lambda p: self.priority[p])
                    else:
                        # unexpected = 0
                        # for x in range(j + 1, self.k + 1):
                        #     if len(self.layers[x]) > 1:
                        #         print('OnlineMin warning: unexpected layer structure during eviction')
                        #         unexpected = 1
                        #         break
                        # if unexpected:
                        #     for x in range(j + 1, self.k + 1):
                        #         # print(
                        #         #     f'  Critical Layer {x}: {[(p, ts_p) for p in self.layers[x] for ts_p in [self.support_last_access.get(p)] if self.support_last_evict.get(address) is not None and ts_p is not None and ts_p < self.support_last_evict.get(address)]} '
                        #         #     f'addr_last_evict={self.support_last_evict.get(address)}'
                        #         # )
                        #         print(
                        #             f'  Layer {x}: {[(p, self.support_last_access.get(p)) for p in self.layers[x]]} '
                        #             f'addr_last_evict={self.support_last_evict.get(address)}'
                        #         )
                        prefix = cache_pages[:j]
                        victim = min(prefix, key=lambda p: self.priority[p])

                victim_idx = self.cache.index(victim)
                self.support_last_evict[victim] = ts
                self.cache[victim_idx] = address
                self.pcs[victim_idx] = pc

        # Update layers after cache update (as in the paper).
        self._update_layers_after_request(address, layer_i, forgiveness)

        # Record last access for the accessed page (kept only if in support).
        self.support_last_access[address] = ts

        # Support can be larger than the cache (up to 3k), so support\cache is allowed.
        # But cache must never contain pages outside the support.
        cache_set = {p for p in self.cache if p is not None}
        extras = [p for p in cache_set if p not in self.support]
        if extras:
            raise RuntimeError(
                'OnlineMin invariant violated: cache contains non-support pages. '
                f'addr={address} hit={hit} layer_i={layer_i} forgiveness={forgiveness} '
                f'extras={extras} '
                f'cache={list(self.cache)} support_size={len(self.support)}'
            )

        self.timestamp += 1
        return hit

class PredictiveOnlineMinAlgorithm(OnlineMinAlgorithm):
    """PredictiveOnlineMin: OnlineMin with predictor-driven eviction on L0 misses.

    Behavior:
    - Same layer/support maintenance as `OnlineMinAlgorithm`.
    - On a cache miss for a page in L0 (i.e. not currently in support) AND the
      cache is full, use the provided `predictor` + `evictor` to choose which
      cache slot to evict.
    - Otherwise, fall back to OnlineMin's eviction rule.

    This class uses the same predictor interface as `PredictAlgorithm` so it can
    be constructed via `PredictAlgorithmFactory.generate_predictive_algorithm`.
    """

    def __init__(
        self,
        associativity: int,
        evictor_type: Union[Type[Evictor], partial],
        predictor_type: Union[Predictor, partial],
        max_support_factor: int = 3,
    ) -> None:
        super().__init__(associativity=associativity, max_support_factor=max_support_factor)

        self.timestamp = 0

        cls_type = predictor_type.func if hasattr(predictor_type, 'func') else predictor_type
        if issubclass(cls_type, ReuseDistancePredictor):
            self.preds = [np.inf] * associativity
        elif issubclass(cls_type, BinaryPredictor):
            self.preds = [0] * associativity
        elif issubclass(cls_type, PhasePredictor):
            self.preds = [1] * associativity
        elif issubclass(cls_type, StatePredictor):
            self.preds = [None] * associativity
        else:
            self.preds = None

        if issubclass(cls_type, OraclePredictor):
            def oracle_access(self, pc, address, next_access_time):
                self.predictor.oracle_access(pc, address, next_access_time)
            self.oracle_access = types.MethodType(oracle_access, self)

        self.evictor = evictor_type()
        self.predictor = predictor_type()

    def snapshot(self):
        return (list(zip(self.cache, self.pcs)), self.preds)

    def before_pred(self, pc, address):
        preds = self.predictor.refresh_scores(self.timestamp, pc, address, self.snapshot()[0])
        if preds is not None:
            self.preds = preds

    def after_pred(self, pc, address, target_index):
        pred = self.predictor.predict_score(self.timestamp, pc, address, self.snapshot()[0])
        if pred is not None and self.preds is not None:
            self.preds[target_index] = pred
        self.timestamp += 1

    def access(self, pc, address) -> bool:
        # Keep predictor state up to date.
        self.before_pred(pc, address)

        handled, hit, target_index = self._warmup_access(pc, address)
        if handled:
            self.after_pred(pc, address, target_index)
            return hit

        # OnlineMin proper, with predictor-driven eviction on L0 misses.
        # Cache must always be a subset of support (OnlineMin invariant). If this
        # is violated (e.g. due to an unintended interaction with forgiveness),
        # fail fast with a clear error instead of crashing with KeyError later.
        cache_set = {p for p in self.cache if p is not None}
        extras = [p for p in cache_set if p not in self.support]
        if extras:
            raise RuntimeError(
                'PredictiveOnlineMin invariant violated: cache contains non-support pages (pre-evict). '
                f'addr={address} extras={extras} cache={list(self.cache)} support_size={len(self.support)}'
            )

        hit = address in self.cache
        target_index = None
        if hit:
            target_index = self.cache.index(address)
            self.pcs[target_index] = pc

        support_size = len(self.support)
        layer_i = self.layer_of.get(address, 0)
        forgiveness = (layer_i == 0 and support_size == self.max_support)
        eviction_layer = 1 if forgiveness else layer_i

        if not hit:
            if None in self.cache:
                target_index = self.cache.index(None)
                self.cache[target_index] = address
                self.pcs[target_index] = pc
            else:
                # L0 miss and cache full: consult predictor for victim choice.
                # IMPORTANT: do NOT override OnlineMin's eviction rule during the
                # forgiveness case (support at cap). OnlineMin relies on a
                # specific eviction+layer-update interaction; picking an
                # arbitrary victim can leave cache pages outside the support.
                if layer_i == 0 and (not forgiveness) and self.preds is not None:
                #if layer_i == 0 and self.preds is not None:
                    victim_idx = self.evictor.evict(list(enumerate(self.preds)))
                else:
                    cache_pages = [p for p in self.cache if p is not None]
                    if eviction_layer == 0:
                        victim = min(cache_pages, key=lambda p: self.priority[p])
                    else:
                        cache_pages.sort(key=lambda p: self.layer_of[p])
                        j = None
                        for jj in range(eviction_layer, self.k + 1):
                            if self.layer_of[cache_pages[jj - 1]] == jj:
                                j = jj
                                break
                        if j is None:
                            victim = min(cache_pages, key=lambda p: self.priority[p])
                        else:
                            victim = min(cache_pages[:j], key=lambda p: self.priority[p])
                    victim_idx = self.cache.index(victim)

                target_index = victim_idx
                self.cache[target_index] = address
                self.pcs[target_index] = pc

        # Update layers after cache update (as in OnlineMin).
        self._update_layers_after_request(address, layer_i, forgiveness)

        # Invariant: cache must be subset of support.
        cache_set = {p for p in self.cache if p is not None}
        extras = [p for p in cache_set if p not in self.support]
        if extras:
            raise RuntimeError(
                'PredictiveOnlineMin invariant violated: cache contains non-support pages. '
                f'addr={address} hit={hit} layer_i={layer_i} forgiveness={forgiveness} '
                f'extras={extras} cache={list(self.cache)} support_size={len(self.support)}'
            )

        # Update predictor score for the accessed/inserted slot.
        if target_index is None:
            # Should never happen, but keep safe.
            target_index = 0
        self.after_pred(pc, address, target_index)

        return hit


class PredictiveRPBOnlineMinAlgorithm(PredictiveOnlineMinAlgorithm):
    """RPB-OnlineMin: OnlineMin with a budget-gated predictor override.

    Budget mechanism (matches the paper's RPB functions $\\mathcal{G}_{RPB}$ and
    $\\mathcal{U}_{RPB}$ with $U(\\omega) = k - |R(\\omega)|$):
      - Forgiveness (L0 miss with support at capacity): always use OnlineMin's
        default victim — never override.
      - True L0 miss (a page outside the current support): always use the
        predictor for the victim. Budget is incremented by ``tau`` *only if* the
        previous epoch (since the last L0 miss) contained at least one
        gate-pass — i.e., F (``self.charge_flag``) was set. F is then reset.
        This matches Theorem ``theorem_rpb_om_robustness``'s $\\Delta B \\le \\tau$
        (= ``tau`` if F else 0).
      - Non-L0 miss (page is in the unrevealed support):
        gate-pass condition $U(\\omega) \\le (Y + 2)/e - 2$ where Y is U at
        the *previous miss*. On gate-pass, budget gains +1 and F becomes 1.
        If budget >= 1, use the predictor (and consume 1); else fall back to
        OnlineMin's default victim.

    Implementation invariant required by the robustness proof:
        ``rpb_y`` (= Y in the paper) is refreshed on EVERY miss, so that the
        gate-pass bundle's "previous process" only spans hit-only steps.
        Updating it less often (e.g., only at L0 + gate-pass) would
        double-count the $\\phi$ decrease at intermediate gate-fail misses
        and break the $H_k + O(1)$ bound.

    Parameters:
      - ``pred_budget`` (tau): per-epoch L0 replenishment. Larger tau lets the
        predictor be used more aggressively on non-L0 misses.
      - ``max_support_factor``: passed to the underlying OnlineMin; caps the
        support at ``k * max_support_factor`` pages.
    """

    def __init__(
        self,
        associativity: int,
        evictor_type: Union[Type[Evictor], partial],
        predictor_type: Union[Predictor, partial],
        max_support_factor: int = 3,
        pred_budget: int = 0,
    ) -> None:
        super().__init__(
            associativity=associativity,
            evictor_type=evictor_type,
            predictor_type=predictor_type,
            max_support_factor=max_support_factor,
        )

        # Replenishable prediction budget (paper symbol B) and its per-epoch
        # replenishment increment (paper symbol tau).
        self.pred_budget_init = int(pred_budget)
        self.pred_budget = self.pred_budget_init
        # Paper's F flag: set on gate-pass, gates the L0 += tau.
        self.charge_flag = 0
        # Y in the paper: U at the previous miss (refreshed every miss).
        self.rpb_y: Optional[int] = None

        # RPB instrumentation
        self.gate_pass_count = 0
        self.gate_fail_count = 0
        self.l0_miss_count = 0
        self.non_l0_pred_evictions = 0
        self.non_l0_om_evictions = 0

    def _replenish_l0_budget(self) -> None:
        """Unconditional hard reset: budget = tau on every L0 miss."""
        self.pred_budget = self.pred_budget_init

    def _onlinemin_eviction_candidate_pages(self, eviction_layer: int) -> List[object]:
        cache_pages = [p for p in self.cache if p is not None]
        if not cache_pages:
            return []
        if eviction_layer == 0:
            return cache_pages

        cache_pages.sort(key=lambda p: self.layer_of[p])
        j = None
        for jj in range(eviction_layer, self.k + 1):
            if self.layer_of[cache_pages[jj - 1]] == jj:
                j = jj
                break
        if j is None:
            # Should not happen; treat as full set.
            return cache_pages
        return cache_pages[:j]

    def _num_of_revealed_layers(self) -> int:
        n = 0
        for jj in range(self.k, 0, -1):
            if len(self.layers[jj]) == 1:
                n += 1
            else:
                break
        return n

    def _onlinemin_victim_idx(self, eviction_layer: int) -> int:
        candidate_pages = self._onlinemin_eviction_candidate_pages(eviction_layer)
        if not candidate_pages:
            return 0
        victim = min(candidate_pages, key=lambda p: self.priority[p])
        return self.cache.index(victim)

    def access(self, pc, address) -> bool:
        # Keep predictor state up to date.
        self.before_pred(pc, address)

        handled, hit, target_index = self._warmup_access(pc, address)
        if handled:
            self.after_pred(pc, address, target_index)
            return hit

        # Invariant: cache must always be a subset of support.
        cache_set = {p for p in self.cache if p is not None}
        extras = [p for p in cache_set if p not in self.support]
        if extras:
            raise RuntimeError(
                'PredictiveRPBOnlineMin invariant violated: cache contains non-support pages (pre-evict). '
                f'addr={address} extras={extras} cache={list(self.cache)} support_size={len(self.support)}'
            )

        hit = address in self.cache
        target_index = None
        if hit:
            target_index = self.cache.index(address)
            self.pcs[target_index] = pc

        layer_i = self.layer_of.get(address, 0)
        forgiveness = (layer_i == 0 and len(self.support) == self.max_support)
        eviction_layer = 1 if forgiveness else layer_i

        # rpb_y must be refreshed on EVERY miss (see class docstring).
        update_rpb_y = (not hit)

        if not hit:
            if None in self.cache:
                target_index = self.cache.index(None)
                self.cache[target_index] = address
                self.pcs[target_index] = pc
            else:
                # Never override OnlineMin during forgiveness.
                if forgiveness:
                    victim_idx = self._onlinemin_victim_idx(eviction_layer)
                    target_index = victim_idx
                    self.cache[target_index] = address
                    self.pcs[target_index] = pc
                else:
                    use_predictor = False
                    if self.preds is not None:
                        if layer_i == 0:
                            # True L0 miss: always predictor-evict.
                            use_predictor = True
                            # Delegate the L0 budget update to the hook so
                            # subclasses can override the replenishment policy
                            # (e.g., unconditional += tau without F gating).
                            self._replenish_l0_budget()
                            self.l0_miss_count += 1
                        else:
                            y_prime = self.k - self._num_of_revealed_layers()
                            if self.rpb_y is not None and y_prime <= (self.rpb_y + 2) / math.e - 2:
                                # Gate-pass: +1 budget and set F for next L0.
                                self.pred_budget += 1
                                self.charge_flag = 1
                                self.gate_pass_count += 1
                            else:
                                self.gate_fail_count += 1

                            if self.pred_budget >= 1:
                                use_predictor = True
                                self.pred_budget -= 1
                                self.non_l0_pred_evictions += 1
                            else:
                                self.non_l0_om_evictions += 1

                    if use_predictor:
                        candidate_pages = self._onlinemin_eviction_candidate_pages(eviction_layer)
                        candidate_indices = [self.cache.index(p) for p in candidate_pages]
                        if not candidate_indices:
                            candidate_indices = list(range(self.k))

                        scored_candidates = [(i, self.preds[i]) for i in candidate_indices]
                        victim_idx = self.evictor.evict(scored_candidates)
                        target_index = victim_idx
                        self.cache[target_index] = address
                        self.pcs[target_index] = pc
                    else:
                        victim_idx = self._onlinemin_victim_idx(eviction_layer)
                        target_index = victim_idx
                        self.cache[target_index] = address
                        self.pcs[target_index] = pc

        # Update layers after cache update (as in OnlineMin).
        self._update_layers_after_request(address, layer_i, forgiveness)
        if update_rpb_y:
            self.rpb_y = self.k - self._num_of_revealed_layers()

        # Invariant: cache must be subset of support.
        cache_set = {p for p in self.cache if p is not None}
        extras = [p for p in cache_set if p not in self.support]
        if extras:
            raise RuntimeError(
                'PredictiveRPBOnlineMin invariant violated: cache contains non-support pages. '
                f'addr={address} hit={hit} layer_i={layer_i} forgiveness={forgiveness} '
                f'extras={extras} cache={list(self.cache)} support_size={len(self.support)}'
            )

        # Update predictor score for the accessed/inserted slot.
        if target_index is None:
            target_index = 0
        self.after_pred(pc, address, target_index)

        return hit


####################################################################

class PredictiveRPBOnlineMinHitCreditAlgorithm(PredictiveRPBOnlineMinAlgorithm):
    """RPB-OnlineMin with persistent hit-accumulated credit.

    On each cache hit, accumulates 1 / (k - num_revealed_layers + 1) into
    hit_credit. Unlike the Continuous variant, hit_credit is NOT reset on
    L0 misses — it persists across epochs, reflecting RPB-OM's cumulative
    cache quality advantage.

    On a non-L0 miss, if hit_credit >= 1, a gate-pass is granted:
    pred_budget += 1, hit_credit -= 1.

    Rationale: hits mean the predictor kept the right pages in cache.
    L0 misses (new pages) are not the predictor's fault and should not
    erase accumulated evidence of good performance.
    """

    def __init__(self, associativity, evictor_type, predictor_type,
                 max_support_factor=3, pred_budget=0):
        super().__init__(associativity, evictor_type, predictor_type,
                         max_support_factor=max_support_factor,
                         pred_budget=pred_budget)
        self.hit_credit = 0.0

    def access(self, pc, address) -> bool:
        self.before_pred(pc, address)

        handled, hit, target_index = self._warmup_access(pc, address)
        if handled:
            if hit:
                unrevealed = self.k - self._num_of_revealed_layers() + 1
                self.hit_credit += 1.0 / unrevealed
            self.after_pred(pc, address, target_index)
            return hit

        cache_set = {p for p in self.cache if p is not None}
        extras = [p for p in cache_set if p not in self.support]
        if extras:
            raise RuntimeError(
                'PredictiveRPBOnlineMinHitCredit invariant violated (pre-evict). '
                f'addr={address} extras={extras}'
            )

        hit = address in self.cache
        target_index = None
        if hit:
            target_index = self.cache.index(address)
            self.pcs[target_index] = pc
            unrevealed = self.k - self._num_of_revealed_layers() + 1
            self.hit_credit += 1.0 / unrevealed

        layer_i = self.layer_of.get(address, 0)
        forgiveness = (layer_i == 0 and len(self.support) == self.max_support)
        eviction_layer = 1 if forgiveness else layer_i

        if not hit:
            if None in self.cache:
                target_index = self.cache.index(None)
                self.cache[target_index] = address
                self.pcs[target_index] = pc
            else:
                if forgiveness:
                    victim_idx = self._onlinemin_victim_idx(eviction_layer)
                    target_index = victim_idx
                    self.cache[target_index] = address
                    self.pcs[target_index] = pc
                else:
                    use_predictor = False
                    if self.preds is not None:
                        if layer_i == 0:
                            use_predictor = True
                            self.pred_budget = self.pred_budget_init
                            self.l0_miss_count += 1
                        else:
                            if self.hit_credit >= 1.0:
                                self.pred_budget += 1
                                self.hit_credit -= 1.0
                                self.gate_pass_count += 1
                            else:
                                self.gate_fail_count += 1

                            if self.pred_budget >= 1:
                                use_predictor = True
                                self.pred_budget -= 1
                                self.non_l0_pred_evictions += 1
                            else:
                                self.non_l0_om_evictions += 1

                    if use_predictor:
                        candidate_pages = self._onlinemin_eviction_candidate_pages(eviction_layer)
                        candidate_indices = [self.cache.index(p) for p in candidate_pages]
                        if not candidate_indices:
                            candidate_indices = list(range(self.k))
                        scored_candidates = [(i, self.preds[i]) for i in candidate_indices]
                        victim_idx = self.evictor.evict(scored_candidates)
                        target_index = victim_idx
                        self.cache[target_index] = address
                        self.pcs[target_index] = pc
                    else:
                        victim_idx = self._onlinemin_victim_idx(eviction_layer)
                        target_index = victim_idx
                        self.cache[target_index] = address
                        self.pcs[target_index] = pc

        self._update_layers_after_request(address, layer_i, forgiveness)
        if not hit:
            self.rpb_y = self.k - self._num_of_revealed_layers()

        cache_set = {p for p in self.cache if p is not None}
        extras = [p for p in cache_set if p not in self.support]
        if extras:
            raise RuntimeError(
                'PredictiveRPBOnlineMinHitCredit invariant violated (post). '
                f'addr={address} hit={hit} layer_i={layer_i} forgiveness={forgiveness} '
                f'extras={extras}'
            )

        if target_index is None:
            target_index = 0
        self.after_pred(pc, address, target_index)

        return hit


class PredictAlgorithmFactory:
    predictor_evict_dict = {
        "PLECO": (MaxEvictor, PLECOPredictor),
        "PLECO-State": (DummyEvictor, PLECOStatePredictor),
        "PLECO-Bin": (BinaryEvictor, PLECOBinPredictor),
        "GBM": (BinaryEvictor, GBMBinPredictor),
        "LRB": (BinaryEvictor, LRBPredictor),
        "POPU": (MaxEvictor, POPUPredictor),
        "POPU-State": (DummyEvictor, POPUStatePredictor),
        "Parrot": (MaxEvictor, ParrotPredictor),
        "Parrot-State": (DummyEvictor, ParrotStatePredictor),
        "OracleDis": (ReuseDistanceEvictor, OracleReuseDistancePredictor),
        "OracleBin": (BinaryEvictor, OracleBinaryPredictor),
        "OraclePhase": (BinaryEvictor, OraclePhasePredictor),
        "OracleState": (DummyEvictor, OracleStatePredictor),
    }

    @staticmethod
    def generate_predictive_algorithm(alg_type: Union[Type[PredictAlgorithm], partial], pred_type_str: str, **kwargs) -> partial:
        evictor_type, predictor_type = PredictAlgorithmFactory.predictor_evict_dict[pred_type_str]
        
        evictor_partial = evictor_type
        predictor_partial = predictor_type
        if pred_type_str == 'Parrot' or pred_type_str == 'Parrot-State' or pred_type_str == 'GBM':
            # shared_model
            if 'shared_model' not in kwargs:
                raise ValueError('PredictAlgorithmFactory: Parrot need [shared_model]')

            if pred_type_str == 'Parrot-State':
                if 'associativity' not in kwargs:
                    raise ValueError(f'PredictAlgorithmFactory: {pred_type_str} need [associativity]')
                associativity = kwargs['associativity']
                predictor_partial = partial(predictor_type, shared_model=kwargs['shared_model'], associativity=associativity)
            else:
                predictor_partial = partial(predictor_type, shared_model=kwargs['shared_model'])
        elif pred_type_str == 'LRB':
            if 'shared_model' not in kwargs:
                raise ValueError('PredictAlgorithmFactory: LRB need [shared_model]')
            memory_window = kwargs.pop('memory_window', 1000000)
            predictor_partial = partial(predictor_type, shared_model=kwargs['shared_model'], memory_window=memory_window)
        elif pred_type_str.startswith('Oracle'):
            reuse_dis_noise_sigma = 0
            lognormal = True
            if 'reuse_dis_noise_sigma' in kwargs:
                reuse_dis_noise_sigma = kwargs['reuse_dis_noise_sigma']
            if 'lognormal' in kwargs:
                lognormal = kwargs['lognormal']

            if pred_type_str == 'OracleDis':
                predictor_partial = partial(predictor_type, reuse_dis_noise_sigma=reuse_dis_noise_sigma, lognormal=lognormal)
            else:
                if 'associativity' not in kwargs:
                    raise ValueError(f'PredictAlgorithmFactory: {pred_type_str} need [associativity]')
                associativity = kwargs['associativity']
                if pred_type_str == 'OracleState':
                    predictor_partial = partial(predictor_type, associativity=associativity, reuse_dis_noise_sigma=reuse_dis_noise_sigma, lognormal=lognormal)
                else:
                    bin_noise_prob = 0
                    if 'bin_noise_prob' in kwargs:
                        bin_noise_prob = kwargs['bin_noise_prob']
                    predictor_partial = partial(predictor_type, associativity=associativity, bin_noise_prob=bin_noise_prob, reuse_dis_noise_sigma=reuse_dis_noise_sigma, lognormal=lognormal)
        elif pred_type_str.endswith('State'):
            if 'associativity' not in kwargs:
                raise ValueError(f'PredictAlgorithmFactory: {pred_type_str} need [associativity]')
            associativity = kwargs['associativity']
            predictor_partial = partial(predictor_type, associativity=associativity)
        elif pred_type_str == 'PLECO-Bin':
            if 'threshold' not in kwargs:
                raise ValueError(f'PredictAlgorithmFactory: {pred_type_str} need [threshold]')
            threshold = kwargs['threshold']
            predictor_partial = partial(predictor_type, threshold=threshold)

        if isinstance(alg_type, partial):
            this_partial = copy.deepcopy(alg_type)
            this_partial.keywords['evictor_type'] = evictor_partial
            this_partial.keywords['predictor_type'] = predictor_partial
            return this_partial
        else:
            return partial(alg_type, evictor_type=evictor_partial, predictor_type=predictor_partial)

def format_guard(relax_times, relax_prob):
    if relax_times == 0 and relax_prob == 0:
        return "-no-relax"
    elif relax_times == 0 and relax_prob != 0:
        return f"-relax-prob-{relax_prob}"
    elif relax_times != 0 and relax_prob == 0:
        return f"-relax-times-{relax_times}"
    else:
        raise ValueError('relax_times and relax_prob invaild')

def format_oracle(reuse_dis_noise_sigma, bin_noise_prob):
    if reuse_dis_noise_sigma == 0 and bin_noise_prob == 0:
        return "-oracle"
    elif reuse_dis_noise_sigma == 0 and bin_noise_prob != 0:
        return f"-bin-{bin_noise_prob}"
    elif reuse_dis_noise_sigma != 0 and bin_noise_prob == 0:
        return f"-dis-{reuse_dis_noise_sigma}"
    else:
        return f"-dis-{reuse_dis_noise_sigma}-bin-{bin_noise_prob}"

def pretty_print(callable: Union[EvictAlgorithm, partial], verbose=False) -> str:
    this_cls = callable
    if hasattr(callable, 'func'):
        this_cls = callable.func
    this_cls_name = (
        this_cls.__name__
        .replace("Algorithm", '')
        .replace("CombineDeterministic", 'CombDet')
        .replace('CombineRandomAlgorithm', 'CombRand')
        .replace("MarkAndPredict", "Mark&Predict")
        .replace('PredictiveMarker', 'PredMark')
        # Plan A / Plan B must be replaced BEFORE the generic RPB replacement,
        # otherwise 'PredictiveRPBOnlineMin' gets eaten first.
        # Longer/more-specific names must come BEFORE shorter ones with same prefix.
        .replace('PredictiveRPBOnlineMinHitCredit', 'RPB-OM-HC')
        .replace('PredictiveRPBOnlineMin', 'RPB-OnlineMin')
    )
    metadata = this_cls_name
    if hasattr(callable, 'keywords'):
        kw = callable.keywords
        if issubclass(this_cls, CombineAlgorithm):
            algs = kw['candidate_algorithms']
            alg_names = []
            for alg in algs:
                alg_names.append(pretty_print(alg, verbose))
            metadata += ("[" + (", ".join(alg_names)) + "]")
        
        if 'predictor_type' in kw:
            predictor_type = kw['predictor_type']
            pred_kw = {}
            if hasattr(predictor_type, 'func'):
                pred_kw = predictor_type.keywords
                predictor_type = predictor_type.func
            predictor = predictor_type.__name__.replace("Predictor", '').replace('OracleReuseDistance', 'Belady').replace('OracleBinary', 'FBP')
            metadata += f'[{predictor}]'

            if issubclass(predictor_type, OraclePredictor) and verbose:
                reuse_dis_noise_sigma = bin_noise_prob = 0
                if 'reuse_dis_noise_sigma' in pred_kw:
                    reuse_dis_noise_sigma = pred_kw['reuse_dis_noise_sigma']
                if 'bin_noise_prob' in pred_kw:
                    bin_noise_prob = pred_kw['bin_noise_prob']
                metadata += format_oracle(reuse_dis_noise_sigma, bin_noise_prob) 

        if issubclass(this_cls, Guard):
            follow_if_guarded = False
            relax_times = relax_prob = 0
            if 'follow_if_guarded' in kw:
                follow_if_guarded = kw['follow_if_guarded']
            if follow_if_guarded:
                metadata += '-unv'
            else:
                metadata += '-f-pred'
            if 'relax_times' in kw:
                relax_times = kw['relax_times']
            if 'relax_prob' in kw:
                relax_prob = kw['relax_prob']
            metadata += format_guard(relax_times, relax_prob)

        if issubclass(this_cls, OnlineMinAlgorithm):
            if 'max_support_factor' in kw:
                metadata += f"-msf-{kw['max_support_factor']}"

        # ChargeFlag is the BASE of all RPB variants (incl. the simple/hard-reset
        # `PredictiveRPBOnlineMinAlgorithm` and Plan A/B subclasses), so checking
        # against it catches every RPB variant.
        if issubclass(this_cls, PredictiveRPBOnlineMinAlgorithm):
            metadata += f"-pb-{kw.get('pred_budget', 0)}"

    return metadata
