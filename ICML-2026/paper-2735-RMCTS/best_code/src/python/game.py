import numpy as np
import random
import numpy.ctypeslib as npct
import ctypes
from pathlib import Path

from . import metaparm

c_int = ctypes.c_int
c_float = ctypes.c_float
array_int32 = npct.ndpointer(dtype=np.int32,ndim=1,flags='CONTIGUOUS')
array_float = npct.ndpointer(dtype=np.float32,ndim=1,flags='CONTIGUOUS')
ptr_int = ctypes.POINTER(c_int)
ptr_float = ctypes.POINTER(c_float)

libgame_path = Path(__file__).parent / 'libgame.so'
libgame = npct.load_library(libgame_path,".")

libnetwork_path = Path(__file__).parent / 'libnetwork.so'
libnetwork = npct.load_library(libnetwork_path,".")

libgame.numActions.restype = c_int
libgame.numActions.argtypes = []

libgame.gameLength.restype = c_int
libgame.gameLength.argtypes = []

libgame.inputLength.restype = c_int
libgame.inputLength.argtypes = []

libgame.rootState.restype = None
libgame.rootState.argtypes = [array_float]

libgame.inputNetwork.restype = None
libgame.inputNetwork.argtypes = [array_float, array_float]

libgame.playerId.restype = c_float
libgame.playerId.argtypes = [array_float]

libgame.gameEnded.restype = c_int
libgame.gameEnded.argtypes = [ptr_float,array_float]

libgame.isValidAction.restype = int
libgame.isValidAction.argtypes = [array_float, c_int]

libgame.getValidActions.restype = c_int
libgame.getValidActions.argtypes = [array_int32,array_float]

libgame.nextState.restype = c_int
libgame.nextState.argtypes = [array_float,array_float,c_int]

libgame.printGame.restype = None
libgame.printGame.argtypes = [array_float]

libnetwork.games_to_nnet_inputs.restype = None
libnetwork.games_to_nnet_inputs.argtypes = [array_float,array_float,c_int]

def numActions():
    return libgame.numActions()

def gameLength():
    return libgame.gameLength()

def inputLength():
    return libgame.inputLength()

def rootState():
    g = np.zeros(gameLength(),dtype=np.float32)
    libgame.rootState(g)
    return g

def inputNetwork(g):
    x = np.zeros(np.prod(metaparm.input_shape), dtype=np.float32)
    libgame.inputNetwork(x,g)
    return x.reshape(metaparm.input_shape)

def inputNetworkMany(G):
    G = np.array(G, dtype=np.float32)
    m = G.shape[0]
    input_len = np.prod(metaparm.input_shape)
    assert input_len == libgame.inputLength()
    X = np.zeros(m*input_len, dtype=np.float32)
    libnetwork.games_to_nnet_inputs(X,G.ravel(),m)
    return X.reshape((m,) + metaparm.input_shape)

def playerId(g):
    return libgame.playerId(g)

def gameEnded(g):
    terminal_score = c_float(0.0)
    ended = libgame.gameEnded(ctypes.byref(terminal_score),g)
    return ended, terminal_score.value

def isValidAction(g,a):
    return libgame.isValidAction(g,a)

def getValidActions(g):
    actions = np.zeros(numActions(),dtype=np.int32)
    num_actions = libgame.getValidActions(actions,g)
    return actions[:num_actions]

def nextState(g,a):
    c_a = c_int(a)
    ga = np.zeros(gameLength(),dtype=np.float32)
    res = libgame.nextState(ga,g,c_a)
    #if res == -1:
    #    print("Warning! The action you took was not valid.")
    #    printGame(g)
    #    print(f"You tried to take action {a}")
    return ga

def printGame(g):
    libgame.printGame(g)

def randomRollout(g=None, verbose=False):
    game_states = []
    if g is None:
        g = rootState()
    if verbose:
        printGame(g)
        print()
    game_states.append(g)
    ended,terminal_score = gameEnded(g)
    while not ended:
        actions = getValidActions(g)
        a = random.choice(actions)
        g = nextState(g,a)
        if verbose:
            print(f"action {a}:")
            printGame(g)
            print()
        ended,terminal_score = gameEnded(g)
        game_states.append(g)
    return game_states
