from abc import ABC, abstractmethod
from typing import Union, List, Tuple
from types import SimpleNamespace
from cache.evict.evictor import MaxEvictor
import numpy as np
import collections
import random
import copy

class Predictor(ABC):
    def refresh_scores(self, ts, pc, address, cache_state: Tuple[List, List]) -> List[Union[int, float, str]]:
        '''
        Before evict, use predictor to refresh all slots' scores

        Scores can be reuse-distance, binary preds and cache's state(keys)

        When evicting, the scores must be the latest.
        '''
        raise NotImplementedError('Predictor: refresh_scores not implemented')
    
    def predict_score(self, ts, pc, address, cache_state) -> Union[int, float, str, None]:
        '''
        Predict this address's score, based on pc, address and cache_state.

        The score is only related to address, with the assistance of other variables

        '''
        raise NotImplementedError('Predictor: predict_score not implemented')

class ReuseDistancePredictor(Predictor):
    '''
    ReuseDistancePredictor only focus on address's score
    '''
    def refresh_scores(self, ts, pc, address, cache_state: Tuple[List, List]) -> List[Union[int, float, str]]:
        return None 

class BinaryPredictor(Predictor):
    '''
    BinaryPredictor only focus on address's score 
    '''
    def refresh_scores(self, ts, pc, address, cache_state: Tuple[List, List]) -> List[Union[int, float, str]]:
        return None 

class PhasePredictor(Predictor):
    def refresh_scores(self, ts, pc, address, cache_state: Tuple[List, List]) -> List[Union[int, float, str]]:
        return None 

class StatePredictor(Predictor):
    def predict_score(self, ts, pc, address, cache_state) -> Union[int, float, None]:
        return None

class OraclePredictor(ABC):
    def __init__(self, reuse_dis_noise_sigma=0, lognormal=True) -> None:
        self.reuse_dis_noise_sigma = reuse_dis_noise_sigma
        self.enable_lognormal = lognormal
    
    def oracle_access(self, pc, address, next_access_time):
        if self.reuse_dis_noise_sigma == 0:
            self.__oracle_access__(pc, address, next_access_time)
        else:
            if self.enable_lognormal:
                self.__oracle_access__(pc, address, next_access_time + np.random.lognormal(0, self.reuse_dis_noise_sigma))
            else:
                self.__oracle_access__(pc, address, next_access_time + np.random.normal(0, self.reuse_dis_noise_sigma))
    @abstractmethod
    def __oracle_access__(self, pc, address, next_access_time):
        pass

class OracleReuseDistancePredictor(ReuseDistancePredictor, OraclePredictor):
    def __init__(self, reuse_dis_noise_sigma=0, lognormal=True, oracle_check=True) -> None:
        super().__init__(reuse_dis_noise_sigma, lognormal)
        self.oracle_preds = collections.deque()
        self.oracle_check = oracle_check
    
    def predict_score(self, ts, pc, address, cache_state):
        oracle_key, next_access_time = self.oracle_preds.popleft()
        if self.oracle_check and oracle_key != address:
            raise ValueError("OracleReuseDistancePredictAlgorithm: oracle key not equals to key")
        return next_access_time

    def __oracle_access__(self, pc, address, next_access_time):
        self.oracle_preds.append((address, next_access_time))

class OracleBinaryPredictor(BinaryPredictor, OraclePredictor):
    def __init__(self, associativity, reuse_dis_noise_sigma=0, bin_noise_prob=0, lognormal=True, oracle_check=True) -> None:
        super().__init__(reuse_dis_noise_sigma, lognormal)
        self.oracle_cache = [None] * associativity
        self.oracle_preds = collections.deque()
        self.oracle_check = oracle_check
        self.oracle_next_access_times = [np.inf] * associativity
        self.oracle_last_time = {}
        self.oracle_t = 0
        self.bin_noise_prob = bin_noise_prob
    
    def predict_score(self, ts, pc, address, cache_state):
        oracle_key, bin_pred = self.oracle_preds.popleft()
        if self.oracle_check and oracle_key != address:
            raise ValueError("OracleBinaryPredictAlgorithm: oracle key not equals to key")

        if self.bin_noise_prob == 0:
            return bin_pred
        else:
            if random.random() < self.bin_noise_prob:
                return 1 - bin_pred
            else:
                return bin_pred

    def __oracle_access__(self, pc, address, next_access_time):
        if address in self.oracle_cache:
            target_index = self.oracle_cache.index(address)
        elif None in self.oracle_cache:
            target_index = self.oracle_cache.index(None)
        else:
            target_index = self.oracle_next_access_times.index(max(self.oracle_next_access_times))
            self.oracle_preds[self.oracle_last_time[self.oracle_cache[target_index]]][1] = 1
        
        self.oracle_cache[target_index] = address
        self.oracle_next_access_times[target_index] = next_access_time 
        self.oracle_preds.append([address, 0])
        self.oracle_last_time[address] = self.oracle_t
        self.oracle_t += 1

class OraclePhasePredictor(PhasePredictor, OraclePredictor):
    def __init__(self, associativity, reuse_dis_noise_sigma=0, bin_noise_prob=0, lognormal=True, oracle_check=True):
        super().__init__(reuse_dis_noise_sigma, lognormal)
        self.associativity = associativity

        self.oracle_t = 0
        self.oracle_preds = collections.deque()
        self.oracle_check = oracle_check
        self.bin_noise_prob = bin_noise_prob

        self.oracle_prev_phase = []
        self.oracle_curr_phase = []
        self.oracle_curr_key_set = set()
    
    def predict_score(self, ts, pc, address, cache_state):
        oracle_key, bin_pred = self.oracle_preds.popleft()
        if self.oracle_check and oracle_key != address:
            raise ValueError("OraclePhasePredictor: oracle key not equals to key")

        if self.bin_noise_prob == 0:
            return bin_pred
        else:
            if random.random() < self.bin_noise_prob:
                return 1 - bin_pred
            else:
                return bin_pred

    def __oracle_access__(self, pc, address, next_access_time):
        if len(self.oracle_curr_key_set) == self.associativity:
            for key, t in self.oracle_prev_phase:
                if key in self.oracle_curr_key_set:
                    self.oracle_preds[t][1] = 0
            self.oracle_prev_phase = self.oracle_curr_phase
            self.oracle_curr_phase = []
            self.oracle_curr_key_set = set()
        self.oracle_curr_phase.append((address, self.oracle_t))
        self.oracle_curr_key_set.add(address)
        self.oracle_preds.append([address, 1])
        self.oracle_t += 1

class OracleStatePredictor(StatePredictor, OraclePredictor):
    def __init__(self, associativity, reuse_dis_noise_sigma=0, lognormal=True, oracle_check=True):
        super().__init__(reuse_dis_noise_sigma, lognormal)
        self.associativity = associativity

        self.oracle_cache = [None] * associativity
        self.oracle_next_access_times = [np.inf] * associativity
        self.oracle_check = oracle_check
        self.oracle_preds = collections.deque()
    
    def refresh_scores(self, ts, pc, address, cache_state: Tuple[List, List]):
        oracle_key, next_cache_state = self.oracle_preds.popleft()
        if self.oracle_check and oracle_key != address:
            raise ValueError("OracleStatePredictor: oracle key not equals to key")
        return next_cache_state

    def __oracle_access__(self, pc, address, next_access_time):
        if address in self.oracle_cache:
            target_index = self.oracle_cache.index(address)
        elif None in self.oracle_cache:
            target_index = self.oracle_cache.index(None)
        else:
            target_index = self.oracle_next_access_times.index(max(self.oracle_next_access_times))
        
        self.oracle_cache[target_index] = address
        self.oracle_next_access_times[target_index] = next_access_time 
        self.oracle_preds.append([address, copy.deepcopy(self.oracle_cache)])

################################################
class SimulateCache:
    '''
    SimulateCache

    Based on normal reuse distance predictor, we can generate other type's predictor

    A simulation cache is necessary.
    '''
    def __init__(self, associativity, reuse_dis_predictor: ReuseDistancePredictor):
        super().__init__()

        self.reuse_dis_predictor = reuse_dis_predictor

        self.sim_cache = [None] * associativity
        self.sim_pcs = [None] * associativity
        # Evict max reuse dis
        self.sim_evictor = MaxEvictor()
        self.sim_scores = [np.inf] * associativity
        self.timestamp = 0
    
    def snapshot(self):
        return (list(zip(self.sim_cache, self.sim_pcs)), self.sim_scores)
    
    def before_pred(self, pc, address):
        preds = self.reuse_dis_predictor.refresh_scores(self.timestamp, pc, address, self.snapshot()[0])
        if preds is not None:
            self.sim_scores = preds
    
    def after_pred(self, pc ,address, target_index):
        pred = self.reuse_dis_predictor.predict_score(self.timestamp, pc, address, self.snapshot()[0])
        if pred is not None:
            self.sim_scores[target_index] = pred
        self.timestamp += 1
    
    def access(self, pc, address):
        self.before_pred(pc, address)
        if address in self.sim_cache:
            target_index = self.sim_cache.index(address)
        elif None in self.sim_cache:
            target_index = self.sim_cache.index(None)
        else:
            target_index = self.sim_evictor.evict(list(enumerate(self.sim_scores)))
        self.sim_cache[target_index], self.sim_pcs[target_index] = address, pc
        self.after_pred(pc, address, target_index)

class HybridStatePredictor(SimulateCache, StatePredictor):
    def __init__(self, associativity, reuse_dis_predictor):
        super().__init__(associativity, reuse_dis_predictor)
    
    def refresh_scores(self, ts, pc, address, cache_state: Tuple[List, List]) -> List[Union[int, float, str]]:
        self.access(pc, address)
        return copy.deepcopy(self.sim_cache)

class ParrotPredictor(ReuseDistancePredictor):
    def __init__(self, shared_model):
        self._model = shared_model

    def predict_score(self, ts, pc, address, cache_state):
        return None

    def refresh_scores(self, ts, pc, address, cache_state: Tuple[List, List]) -> List[Union[int, float, str]]:
        cache_access = SimpleNamespace()
        cache_access.pc = pc
        cache_access.address = address
        cache_access.cache_lines = cache_state
        scores = self._model(cache_access)
        return [scores[0, i].item() for i in range(len(cache_state))]

class ParrotStatePredictor(HybridStatePredictor):
    def __init__(self, associativity, shared_model):
        super().__init__(associativity, ParrotPredictor(shared_model))

class PLECOPredictor(ReuseDistancePredictor):
    def __init__(self):
        super().__init__()
        self.timestamp = 1
        self.weights = []
        self.sum_weights = 0
        self.prev_occs = {}
        self.p = False
    
    def predict_score(self, ts, pc, address, cache_state):
        this_weight = (self.timestamp + 10) ** (-1.8) * np.exp(-self.timestamp / 670)
        self.weights.append(this_weight)
        self.sum_weights += this_weight
        if address not in self.prev_occs:
            self.prev_occs[address] = []
        self.prev_occs[address].append(self.timestamp)
        prob = sum(self.weights[self.timestamp - i] for i in self.prev_occs[address]) / self.sum_weights
        pred = 1 / prob + self.timestamp - 1
        self.timestamp += 1

        return pred

class PLECOBinPredictor(BinaryPredictor):
    def __init__(self, threshold):
        super().__init__()
        self.timestamp = 1
        self.weights = []
        self.sum_weights = 0
        self.prev_occs = {}
        self.p = False
        self.threshold = threshold
    
    def predict_score(self, ts, pc, address, cache_state):
        this_weight = (self.timestamp + 10) ** (-1.8) * np.exp(-self.timestamp / 670)
        self.weights.append(this_weight)
        self.sum_weights += this_weight
        if address not in self.prev_occs:
            self.prev_occs[address] = []
        self.prev_occs[address].append(self.timestamp)
        prob = sum(self.weights[self.timestamp - i] for i in self.prev_occs[address]) / self.sum_weights
        
        if prob > self.threshold:
            return 0
        else:
            return 1

class PLECOStatePredictor(HybridStatePredictor):
    def __init__(self, associativity):
        super().__init__(associativity, PLECOPredictor())

class POPUPredictor(ReuseDistancePredictor):
    def __init__(self):
        super().__init__()
        self.counts = {}
        self.timestamp = 1
    
    def predict_score(self, ts, pc, address, cache_state):
        if address not in self.counts:
            self.counts[address] = 0
        self.counts[address] += 1

        pred = self.timestamp + self.timestamp / self.counts[address]
        self.timestamp += 1
        return pred

class POPUStatePredictor(HybridStatePredictor):
    def __init__(self, associativity):
        super().__init__(associativity, POPUPredictor())

class GBMBinPredictor(BinaryPredictor):
    def __init__(self, shared_model):
        super().__init__()
        self._model = shared_model
        self.delta_nums = self._model.deltanums
        self.edc_nums = self._model.edcnums
        self.deltas = [{} for _ in range(self.delta_nums)]
        self.edcs = [{} for _ in range(self.edc_nums)]
        self.access_time_dict = {}
        self.access_ts = 0
    
    def predict_score(self, ts, pc, address, cache_state):
        if address not in self.access_time_dict:
            self.access_time_dict[address] = collections.deque()
        
        this_access_list = self.access_time_dict[address]
        if len(this_access_list) == self.delta_nums + 1:
            this_access_list.popleft()
            this_access_list.append(self.access_ts)
        else:
            this_access_list.append(self.access_ts)
        
        # delta
        for i in range(1, self.delta_nums + 1):
            this_delta = self.deltas[i-1]
            if len(this_access_list) > i:
                delta_i = this_access_list[-i] - this_access_list[-i-1]
                this_delta[address] = delta_i
            else:
                this_delta[address] = np.inf

        delta1 = self.deltas[0][address]

        for i in range(1, self.edc_nums + 1):
            this_edc = self.edcs[i-1]
            if address not in this_edc:
                this_edc[address] = 0
            this_edc[address] = 1 + this_edc[address] * 2 ** (-delta1 / (2 ** (9 + i)))

        self.access_ts += 1
        return self._model((pc, address, *[self.deltas[i][address] for i in range(self.delta_nums)], *[self.edcs[i][address] for i in range(self.edc_nums)]))

class LRBPredictor(BinaryPredictor):
    """GBM-on-Delta/EDC with a sliding memory window.

    Same feature extraction as GBMBinPredictor, but every 1000 steps purges
    per-address state for addresses last touched longer than `memory_window`
    accesses ago. This bounds memory on long traces, matching the online
    relaxation in Song et al., NSDI'20.
    """
    def __init__(self, shared_model, memory_window=1000000):
        super().__init__()
        self._model = shared_model
        self.delta_nums = self._model.deltanums
        self.edc_nums = self._model.edcnums
        self.memory_window = memory_window

        self.deltas = [{} for _ in range(self.delta_nums)]
        self.edcs = [{} for _ in range(self.edc_nums)]
        self.access_time_dict = {}
        self.access_ts = 0

    def predict_score(self, ts, pc, address, cache_state):
        if address not in self.access_time_dict:
            self.access_time_dict[address] = collections.deque()

        this_access_list = self.access_time_dict[address]
        if len(this_access_list) == self.delta_nums + 1:
            this_access_list.popleft()
        this_access_list.append(self.access_ts)

        for i in range(1, self.delta_nums + 1):
            this_delta = self.deltas[i-1]
            if len(this_access_list) > i:
                this_delta[address] = this_access_list[-i] - this_access_list[-i-1]
            else:
                this_delta[address] = np.inf

        delta1 = self.deltas[0][address]
        for i in range(1, self.edc_nums + 1):
            this_edc = self.edcs[i-1]
            if address not in this_edc:
                this_edc[address] = 0
            this_edc[address] = 1 + this_edc[address] * 2 ** (-delta1 / (2 ** (9 + i)))

        self.access_ts += 1

        features = (pc, address, *[self.deltas[i][address] for i in range(self.delta_nums)], *[self.edcs[i][address] for i in range(self.edc_nums)])
        pred = self._model(features)

        if self.access_ts % 1000 == 0:
            cutoff = self.access_ts - self.memory_window
            stale = [k for k, q in self.access_time_dict.items() if q[-1] < cutoff]
            for key in stale:
                del self.access_time_dict[key]
                for i in range(self.delta_nums):
                    self.deltas[i].pop(key, None)
                for i in range(self.edc_nums):
                    self.edcs[i].pop(key, None)

        return pred