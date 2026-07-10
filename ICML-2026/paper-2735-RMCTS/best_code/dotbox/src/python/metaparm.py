# Monte Carlo tree search parameters
numLanes = 32 # number of concurrent searches (optimizes network predictions request buffer)
numMCTSSims = 512 # budget for Monte Carlo tree search (number of nodes visited)
c_puct = 4.0
num_threads = 4

# network parameters
input_shape = (3,5,5) # shape of the neural network input
engine_batchsize = 512 # batch size for neural network inference engine



