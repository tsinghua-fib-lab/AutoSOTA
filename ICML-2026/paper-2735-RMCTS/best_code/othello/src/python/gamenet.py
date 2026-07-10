# gamenet.py must contain 
# input_shape
# engine_batchsize# untrained_model() : returns an untrained pytorch model (it's ok to access taz.learn.models for that)
#                this model must have the appropriate policy,value outputs

from . import models

N = 8
num_actions = N*N

input_shape = (1,N,N)

def untrained_model():
	return models.SimpleAZModel2d((1,N,N),num_actions,channels=64,blocks=8,kernel_size=3)


