# gamenet.py must contain 
# input_shape
# engine_batchsize# untrained_model() : returns an untrained pytorch model (it's ok to access taz.learn.models for that)
#                this model must have the appropriate policy,value outputs

from . import models

N = 5
num_actions = 2*N*(N-1)

input_shape = (3,N,N)
engine_batchsize = 512

def untrained_model():
	return models.SimpleAZModel2d((3,N,N),num_actions,channels=48,blocks=8,kernel_size=3)


