# gamenet.py must contain 
# input_shape
# engine_batchsize
# untrained_model() : returns an untrained pytorch model (it's ok to access taz.learn.models for that)
#                this model must have the appropriate policy,value outputs

from . import models

input_shape = (1,6,7)
engine_batchsize = 256

def untrained_model():
	return models.SimpleAZModel2d((1,6,7),7,channels=64,blocks=8)



