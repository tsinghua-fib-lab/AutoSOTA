# Monte Carlo tree search parameters
numLanes = 32 # number of concurrent searches (optimizes network predictions request buffer)
numMCTSSims = 256 # budget for Monte Carlo tree search (number of nodes visited)
c_puct = 4.0 # exploration constant
num_threads = 4 # MCTS_ucb.c uses pthreads for parallelism; more than 4 does not seem to help

#network parameters
input_shape = (1,8,8) # input shape for input (image of game state) to the neural network
engine_batchsize = 256 # preferred batch size for neural network inference engine
