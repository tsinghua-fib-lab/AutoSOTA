# Monte Carlo tree search parameters
numLanes = 64 # number of concurrent searches (optimizes network predictions request buffer)
numMCTSSims = 2048 # budget for Monte Carlo tree search (number of nodes visited)
c_puct = 5.0
num_threads = 4

# network parameters
input_shape = (1,6,7) # input shape for input (image of game
engine_batchsize = 512 # preferred batch size for neural network inference engine



