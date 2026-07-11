from __future__ import print_function
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle as pkl
from collections import defaultdict
import re 
from bs4 import BeautifulSoup 
import sys
import argparse
import os
import time
from keras.callbacks import ModelCheckpoint    
from keras.layers import Dense, Input, Flatten, Add, Multiply, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras import regularizers
from keras import backend as K
from keras.engine.topology import Layer 

import json
import random
from keras import optimizers

BATCH_SIZE = 1000
np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)
# The number of key features for each data set.
ks = {'orange_skin': 4, 'XOR': 2, 'nonlinear_additive': 4, 'switch': 5}

def create_rank(scores, k): 
	"""
	Compute rank of each feature based on weight.
	
	"""
	scores = abs(scores)
	n, d = scores.shape
	ranks = []
	for i, score in enumerate(scores):
		# Random permutation to avoid bias due to equal weights.
		idx = np.random.permutation(d) 
		permutated_weights = score[idx]  
		permutated_rank=(-permutated_weights).argsort().argsort()+1
		rank = permutated_rank[np.argsort(idx)]

		ranks.append(rank)

	return np.array(ranks) #Tal - for each instance, gives each feature a rank from 1 to d. 1 being highest. Randomly breaks ties.

def compute_median_rank(scores, k, datatype_val = None):
	ranks = create_rank(scores, k) #Tal - note k is not used
	if datatype_val is None: 
		median_ranks = np.median(ranks[:,:k], axis = 1)
	else: #looks to be used just for switch - Tal
		datatype_val = datatype_val[:len(scores)]
		median_ranks1 = np.median(ranks[datatype_val == 'orange_skin',:][:,np.array([0,1,2,3,9])], 
			axis = 1)
		median_ranks2 = np.median(ranks[datatype_val == 'nonlinear_additive',:][:,np.array([4,5,6,7,9])], 
			axis = 1)
		median_ranks = np.concatenate((median_ranks1,median_ranks2), 0)
	return median_ranks 

class Sample_Concrete(Layer):
	"""
	Layer for sample Concrete / Gumbel-Softmax variables. 

	"""
	def __init__(self, tau0, k, **kwargs): 
		self.tau0 = tau0
		self.k = k
		super(Sample_Concrete, self).__init__(**kwargs)

	def call(self, logits):   
		# logits: [BATCH_SIZE, d]
		logits_ = K.expand_dims(logits, -2)# [BATCH_SIZE, 1, d]

		batch_size = tf.shape(logits_)[0]
		d = tf.shape(logits_)[2]
		uniform = tf.random_uniform(shape =(batch_size, self.k, d), 
			minval = np.finfo(tf.float32.as_numpy_dtype).tiny,
			maxval = 1.0)

		gumbel = - K.log(-K.log(uniform))
		noisy_logits = (gumbel + logits_)/self.tau0
		samples = K.softmax(noisy_logits)
		samples = K.max(samples, axis = 1) 

		# Explanation Stage output.
		threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted = True)[0][:,-1], -1)
		discrete_logits = tf.cast(tf.greater_equal(logits,threshold),tf.float32)
		
		return K.in_train_phase(samples, discrete_logits) #gives samples in training and discrete logits in inference

	def compute_output_shape(self, input_shape):
		return input_shape 
	
def L2X(datatype,
		num_important_features,
		train = True,
		parent_dir=None,
		data_dict = {},
		activation = 'relu',
		num_classes = None, #added
		task = 'classification',
		return_pred_and_mask = False
		):
	
	if num_classes is None:
		raise ValueError("num_classes must be provided for L2X.")

	required_data_keys = ['x_train', 'y_train', 'x_val']
	assert all(key in data_dict for key in required_data_keys) 

	datatype_val = None #was just used for the switch data
	x_train = data_dict['x_train']
	y_train = data_dict['y_train']
	x_val = data_dict['x_val']
	
	input_shape = x_train.shape[1]

	
	st1 = time.time()
	st2 = st1

	# activation = 'relu' if datatype in ['orange_skin','XOR'] else 'selu'
	# P(S|X)
	model_input = Input(shape=(input_shape,), dtype='float32') 

	net = Dense(100, activation=activation, name = 's/dense1',
		kernel_regularizer=regularizers.l2(1e-3))(model_input)
	net = Dense(100, activation=activation, name = 's/dense2',
		kernel_regularizer=regularizers.l2(1e-3))(net) 

	# A tensor of shape, [batch_size, max_sents, 100]
	logits = Dense(input_shape)(net) 
	# [BATCH_SIZE, max_sents, 1]  
	# k = ks[datatype]; tau = 0.1
	k = num_important_features; tau = 0.1
	samples = Sample_Concrete(tau, k, name = 'sample')(logits)

	# q(X_S)
	new_model_input = Multiply()([model_input, samples]) 
	net = Dense(200, activation=activation, name = 'dense1',
		kernel_regularizer=regularizers.l2(1e-3))(new_model_input) 
	net = BatchNormalization()(net) # Add batchnorm for stability.
	net = Dense(200, activation=activation, name = 'dense2',
		kernel_regularizer=regularizers.l2(1e-3))(net)
	net = BatchNormalization()(net)

	out_activation = 'sigmoid' if task == 'multilabel' else 'softmax'
	preds = Dense(num_classes, activation=out_activation, name = 'dense4',
		kernel_regularizer=regularizers.l2(1e-3))(net)
	model = Model(model_input, preds)

	if train:
		adam = optimizers.Adam(lr = 1e-3)
		compile_loss = 'binary_crossentropy' if task == 'multilabel' else 'categorical_crossentropy'
		model.compile(loss=compile_loss,
					  optimizer=adam,
					  metrics=['acc'])
		filepath="{}/models/{}/L2X.hdf5".format(parent_dir, datatype)
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
			verbose=1, save_best_only=True, mode='max')
		callbacks_list = [checkpoint]

		model.fit(x_train, y_train, validation_data=None,callbacks = callbacks_list, epochs=1, batch_size=BATCH_SIZE)
		st2 = time.time() 
	else:
		model.load_weights("{}/models/{}/L2X.hdf5".format(parent_dir, datatype), 
			by_name=True) 

	# for binary mask (standard l2x)
	pred_model = Model(model_input, samples)
	pred_model.compile(loss=None,
				  optimizer='rmsprop',
				  metrics=[None])
	binary_mask = pred_model.predict(x_val, verbose = 1, batch_size = BATCH_SIZE) #binary
	
	# # for the continuous masks (added - not tested)
    # # Apply a sigmoid to the raw logits to get deterministic 0-1 probabilities
	# continuous_probs = Lambda(lambda x: K.sigmoid(x))(logits)
    # continuous_model = Model(model_input, continuous_probs)
    # continuous_model.compile(loss=None, optimizer='rmsprop', metrics=[None])
    # mask = continuous_model.predict(x_val, verbose=1, batch_size=BATCH_SIZE)
	mask = None

	# for predictions - added
	y_val_pred = model.predict(x_val, verbose=1, batch_size=BATCH_SIZE)
	
	# median_ranks = compute_median_rank(binary_mask, k = ks[datatype],
	# 	datatype_val=datatype_val)

	# return median_ranks, time.time() - st2, st2 - st1
	if return_pred_and_mask == True:
		return binary_mask, mask, y_val_pred
	else:
		return binary_mask
