import numpy as np
import numpy.ctypeslib as npct
import ctypes
from pathlib import Path
import time


from . import game, metaparm

c_int = ctypes.c_int
c_float = ctypes.c_float
array_int32 = npct.ndpointer(dtype=np.int32,ndim=1,flags='CONTIGUOUS')
array_float = npct.ndpointer(dtype=np.float32,ndim=1,flags='CONTIGUOUS')
array_double = npct.ndpointer(dtype=np.float64,ndim=1,flags='CONTIGUOUS')
ptr_int = ctypes.POINTER(c_int)
ptr_float = ctypes.POINTER(c_float)
ptr_void = ctypes.c_void_p # same as ctypes.c_voidp

libMCTS_ucb_path = Path(__file__).parent / 'libMCTS_ucb.so'
libMCTS_ucb = npct.load_library(libMCTS_ucb_path,".")

libMCTS_ucb.MCTS_init.restype = ptr_void
libMCTS_ucb.MCTS_init.argtypes = [c_int]*2 + [array_float]*6 + [c_float]

libMCTS_ucb.MCTS_free.restype = None
libMCTS_ucb.MCTS_free.argtypes = [ptr_void]

libMCTS_ucb.MCTS_update.restype = c_int
libMCTS_ucb.MCTS_update.argtypes = [ptr_void, c_int, c_int]

libMCTS_ucb.get_time.restype = None
libMCTS_ucb.get_time.argtypes = [array_double, ptr_void]

libMCTS_ucb.MCTS_lookup.restype = c_int
libMCTS_ucb.MCTS_lookup.argtypes = [array_float, ptr_void, c_int, array_float]

def MCTS_init(stacksize, numSims, root_states, G, P, V, N, Q, c_puct=None):
    if c_puct is None:
        c_puct = metaparm.c_puct
    mcts = libMCTS_ucb.MCTS_init(stacksize,numSims, root_states.ravel(), G, P, V, N, Q, c_puct)
    return mcts

def MCTS_free(mcts):
    libMCTS_ucb.MCTS_free(mcts)

try:
    from taz.build import advice
    hint = advice.hint
except:
    n = game.numActions()
    uniform_prob = np.float32(1.0)/np.float32(n)
    def hint(G):
        assert len(G.shape) == 2
        assert G.shape[1] == game.gameLength()
        m = G.shape[0]
        pi = np.zeros((m,n), dtype=np.float32)
        pi[:] = uniform_prob
        v = np.zeros((m,1),dtype=np.float32)
        return pi, v

def learn_pi_and_v(root_states, numSims, engine, c_puct=None):
    m = len(root_states)
    if engine is None:
        pi0, v0 = hint(root_states)
        pi0.resize((m,game.numActions()))
        v0.resize((m,))
        return pi0, v0
    pi0, v0 = engine(root_states)
    pi0.resize((m,game.numActions()))
    v0.resize((m,))
    if numSims == 1:
        return pi0, v0
    N, Q = learn_N_and_Q(root_states, numSims-1, engine, c_puct=c_puct)
    pi = N / (np.sum(N, axis=1, keepdims=True))
    v = (v0 + (numSims-1)*np.sum(pi*Q, axis=1).flatten()) / numSims
    return pi, v

def learn_N_and_Q(root_states, numSims, engine, c_puct=None, verbosity=0, timings=False, num_threads=4):
    num_root_states = root_states.shape[0]
    chunksize = metaparm.numLanes
    num_chunks = ((num_root_states - 1)//chunksize) + 1
    N = np.zeros((num_root_states,game.numActions()),dtype=np.float32)
    Q = np.zeros((num_root_states,game.numActions()),dtype=np.float32)
    cpu_time = 0.0
    gpu_time = 0.0
    mcts_times = np.zeros(4)

    for i in range(num_chunks):
        res = _learn_N_and_Q(root_states[i*chunksize : (i+1)*chunksize], 
                                numSims, 
                                engine, 
                                c_puct=c_puct,
                                verbosity=verbosity,
                                timings=timings,
                                num_threads=num_threads)
        N[i*chunksize : (i+1)*chunksize] = res[0]
        Q[i*chunksize : (i+1)*chunksize] = res[1]
        if timings:
            cpu_time += res[2]
            gpu_time += res[3]
            mcts_times += res[4]
    if timings:
        return N,Q,cpu_time,gpu_time,mcts_times
    else:
        return N,Q

def _learn_N_and_Q(root_states, numSims, engine, c_puct=None, verbosity=0, timings=False, num_threads=4):
    cpu_time = 0.0
    gpu_time = 0.0
    t0 = time.time()
    n = game.numActions()
    (num_root_states, gamesize) = root_states.shape
    assert root_states.flags['C_CONTIGUOUS']
    assert gamesize == game.gameLength()
    assert type(root_states[0,0]) is np.float32
    assert num_root_states <= metaparm.numLanes

    stacksize = num_root_states

    # flat arrays for mcts object
    G = np.zeros(stacksize * game.gameLength(), dtype=np.float32)
    P = np.zeros(stacksize * n, dtype=np.float32)
    V = np.zeros(stacksize, dtype=np.float32)
    Q = np.zeros(stacksize * n, dtype=np.float32)
    N = np.zeros(stacksize * n, dtype=np.float32)

    
    mcts = MCTS_init(stacksize,numSims,root_states,G,P,V,N,Q, c_puct=c_puct)

    num_requests = 0

    uniform = np.float32(1.0)/np.float32(n)

    while True:
        t0 = time.time()
        num_requests = libMCTS_ucb.MCTS_update(mcts,verbosity,num_threads)
        cpu_time += time.time() - t0
        t0 = time.time()
        
        if num_requests > 0:
            P[:] = uniform
            V[:] = 0.0
            if engine is not None:
                t0 = time.time()
                policy, value = engine(G[:num_requests*game.gameLength()].reshape((num_requests, game.gameLength())))
                P[:num_requests*n] = policy.ravel()
                V[:num_requests] = value.ravel()
                gpu_time += time.time() - t0
                t0 = time.time()
            else:
                policy0, value0 = hint(G[:num_requests*game.gameLength()].reshape((num_requests, game.gameLength())))
                P[:num_requests*n] = policy0.ravel()
                V[:num_requests] = value0.ravel()                
        else:
            break

    N = N.reshape((stacksize,n)) 
    Q = Q.reshape((stacksize,n))

    mcts_times = np.zeros(4, dtype=np.float64)
    libMCTS_ucb.get_time(mcts_times, mcts)

    MCTS_free(mcts)

    if not timings:
        return N, Q
    else:
        return N, Q, cpu_time, gpu_time, mcts_times

def MCTS_lookup(mcts, lane, g):
    n = game.numActions()
    PQNv = np.zeros(3*n+1, dtype=np.float32)
    found = libMCTS_ucb.MCTS_lookup(PQNv, mcts, lane, g)
    return found, PQNv

