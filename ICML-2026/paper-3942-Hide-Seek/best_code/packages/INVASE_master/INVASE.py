'''
Personalized Variable Selection Code (PVS)
for ICLR 2019 Conference
'''

#%% Necessary packages
# 1. Keras
from keras.layers import Input, Dense, Multiply
from keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import regularizers
from keras import backend as K

# 2. Others
import tensorflow as tf
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import repo_paths  # noqa: F401
from tools import run_feature_selection_model

#@TAL additions
from joblib import Parallel, delayed
from datetime import datetime
from tqdm import tqdm


#%% Define PVS class
class PVS():
    
    # 1. Initialization
    '''
    x_train: training samples
    data_type: Syn1 to Syn 6
    '''
    def __init__(self, x_train, data_type,
                 batch_size,
                 epochs,
                 lamda,
                 num_classes,
                 task='classification'):
        self.latent_dim1 = 100      # Dimension of actor (generator) network
        self.latent_dim2 = 200      # Dimension of critic (discriminator) network

        self.batch_size = batch_size      # Batch size
        self.epochs = epochs         # Epoch size (large epoch is needed due to the policy gradient framework)
        self.lamda = lamda            # Hyper-parameter for the number of selected features
        self.task = task

        if not isinstance(num_classes, (int, np.integer)):
            raise ValueError('num_classes must be an integer >= 2')
        if num_classes < 2:
            raise ValueError('num_classes must be >= 2')
        self.num_classes = int(num_classes)

        self.input_shape = x_train.shape[1]     # Input dimension
        self.input_shape0 = x_train.shape[0]   # Number of training samples

        # Actionvation. (For Syn1 and 2, relu, others, selu)
        self.activation = 'relu' if data_type in ['Syn1','Syn2'] else 'selu'

        # Use Adam optimizer with learning rate = 0.0001
        optimizer1 = Adam(0.0001) #@TAL updated - 3 optimizers
        optimizer2 = Adam(0.0001) #@TAL updated - 3 optimizers
        optimizer3 = Adam(0.0001) #@TAL updated - 3 optimizers

        pred_loss = 'binary_crossentropy' if task == 'multilabel' else 'categorical_crossentropy'

        # Build and compile the discriminator (critic)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=pred_loss, optimizer=optimizer1, metrics=['acc'])

        # Build the generator (actor)
        self.generator = self.build_generator()
        self.generator.compile(loss=self.my_loss, optimizer=optimizer2)

        # Build and compile the value function
        self.valfunction = self.build_valfunction()
        self.valfunction.compile(loss=pred_loss, optimizer=optimizer3, metrics=['acc'])
        

    #%% Custom loss definition
    def my_loss(self, y_true, y_pred):
        
        # dimension of the features
        d = y_pred.shape[1]
        c = self.num_classes
        
        # Put all three in y_true 
        # 1. selected probability
        sel_prob = y_true[:,:d] #in original code, values are 0 or 1
        # 2. discriminator output
        dis_prob = y_true[:,d:(d+c)]
        # 3. valfunction output
        val_prob = y_true[:,(d+c):(d+2*c)]
        # 4. ground truth
        y_final = y_true[:,(d+2*c):]
        
        # A1/A2. Reward = log-likelihood of predictions given true labels.
        # For multilabel use full BCE; for classification use CE (y is one-hot so (1-y)*log(1-p) terms cancel).
        if self.task == 'multilabel':
            Reward1 = tf.reduce_sum(y_final * tf.math.log(dis_prob + 1e-8) + (1 - y_final) * tf.math.log(1 - dis_prob + 1e-8), axis=1)
            Reward2 = tf.reduce_sum(y_final * tf.math.log(val_prob + 1e-8) + (1 - y_final) * tf.math.log(1 - val_prob + 1e-8), axis=1)
        else:
            Reward1 = tf.reduce_sum(y_final * tf.math.log(dis_prob + 1e-8), axis = 1)
            Reward2 = tf.reduce_sum(y_final * tf.math.log(val_prob + 1e-8), axis = 1)

        # Difference is the rewards
        Reward = Reward1 - Reward2

        # B. Policy gradient loss computation. 
        loss1 = Reward * tf.reduce_sum( sel_prob * tf.math.log(y_pred + 1e-8) + (1-sel_prob) * tf.math.log(1-y_pred + 1e-8), axis = 1) - self.lamda * tf.reduce_mean(y_pred, axis = 1)
        
        # C. Maximize the loss1
        loss = tf.reduce_mean(-loss1)

        return loss

    #%% Generator (Actor)
    def build_generator(self):

        model = Sequential()
        
        model.add(Dense(self.latent_dim1, activation=self.activation, name = 'sdense1', kernel_regularizer=regularizers.l2(1e-3), input_dim = self.input_shape))
        model.add(Dense(self.latent_dim1, activation=self.activation, name = 'sdense2', kernel_regularizer=regularizers.l2(1e-3)))
        model.add(Dense(self.input_shape, activation = 'sigmoid', name = 'sdense3', kernel_regularizer=regularizers.l2(1e-3)))
        
        model.summary()

        feature = Input(shape=(self.input_shape,), dtype='float32')
        select_prob = model(feature)

        return Model(feature, select_prob)

    #%% Discriminator (Critic)
    def build_discriminator(self):

        model = Sequential()
        
        model.add(Dense(self.latent_dim2, activation=self.activation, name = 'dense1', kernel_regularizer=regularizers.l2(1e-3), input_dim = self.input_shape)) 
        model.add(BatchNormalization())     # Use Batch norm for preventing overfitting
        model.add(Dense(self.latent_dim2, activation=self.activation, name = 'dense2', kernel_regularizer=regularizers.l2(1e-3)))
        model.add(BatchNormalization())
        out_activation = 'sigmoid' if self.task == 'multilabel' else 'softmax'
        model.add(Dense(self.num_classes, activation=out_activation, name = 'dense3', kernel_regularizer=regularizers.l2(1e-3)))
        
        model.summary()
        
        # There are two inputs to be used in the discriminator
        # 1. Features
        feature = Input(shape=(self.input_shape,), dtype='float32')
        # 2. Selected Features
        sel_prob = Input(shape=(self.input_shape,), dtype='float32')
        
        # Element-wise multiplication
        model_input = Multiply()([feature, sel_prob])
        prob = model(model_input)
        return Model([feature, sel_prob], prob)

    #%% Value Function
    def build_valfunction(self):

        model = Sequential()
                
        model.add(Dense(200, activation=self.activation, name = 'vdense1', kernel_regularizer=regularizers.l2(1e-3), input_dim = self.input_shape)) 
        model.add(BatchNormalization())     # Use Batch norm for preventing overfitting
        model.add(Dense(200, activation=self.activation, name = 'vdense2', kernel_regularizer=regularizers.l2(1e-3)))
        model.add(BatchNormalization())
        out_activation = 'sigmoid' if self.task == 'multilabel' else 'softmax'
        model.add(Dense(self.num_classes, activation=out_activation, name = 'vdense3', kernel_regularizer=regularizers.l2(1e-3)))
        
        model.summary()
        
        # There are one inputs to be used in the value function
        # 1. Features
        feature = Input(shape=(self.input_shape,), dtype='float32')       
        
        # Element-wise multiplication
        prob = model(feature)

        return Model(feature, prob)

    #%% Sampling the features based on the output of the generator
    def Sample_M(self, gen_prob):
        
        # Shape of the selection probability
        n = gen_prob.shape[0]
        d = gen_prob.shape[1]
                
        # Sampling
        samples = np.random.binomial(1, gen_prob, (n,d))
        
        return samples

    #%% Training procedure
    def train(self, x_train, y_train):
        if y_train.ndim != 2:
            raise ValueError('y_train must be one-hot/class-prob matrix with shape (n_samples, num_classes)')
        if y_train.shape[1] != self.num_classes:
            raise ValueError('y_train second dimension must match num_classes')

        # For each epoch (actually iterations)
        for epoch in range(self.epochs):
            #%% Train Discriminator
            # Select a random batch of samples
            idx = np.random.randint(0, x_train.shape[0], self.batch_size) #note this samples with replacement

            x_batch = x_train[idx,:]
            y_batch = y_train[idx,:]

            # Generate a batch of probabilities of feature selection
            gen_prob = self.generator.predict(x_batch)
            
            # Sampling the features based on the generated probability
            sel_prob = self.Sample_M(gen_prob)
            
            # Compute the prediction of the critic based on the sampled features (used for generator training)
            dis_prob = self.discriminator.predict([x_batch, sel_prob])

            # Train the discriminator
            d_loss = self.discriminator.train_on_batch([x_batch, sel_prob], y_batch)
            #%% Train Valud function

            # Compute the prediction of the critic based on the sampled features (used for generator training) #Tal - I don't think this comment is true
            val_prob = self.valfunction.predict(x_batch)

            # Train the discriminator #Tal - I don't think this comment is true
            v_loss = self.valfunction.train_on_batch(x_batch, y_batch)
            
            #%% Train Generator
            # Use three things as the y_true: sel_prob, dis_prob, and ground truth (y_batch)
            y_batch_final = np.concatenate( (sel_prob, np.asarray(dis_prob), np.asarray(val_prob), y_batch), axis = 1 )

            # Train the generator
            g_loss = self.generator.train_on_batch(x_batch, y_batch_final)

            #%% Plot the progress
            dialog = 'Epoch: ' + str(epoch) + ', d_loss (Acc)): ' + str(d_loss[1]) + ', v_loss (Acc): ' + str(v_loss[1]) + ', g_loss: ' + str(np.round(g_loss,4))

            if epoch % 100 == 0:
                print(dialog)
    
    #%% Selected Features        
    def output(self, x_train):
        
        gen_prob = self.generator.predict(x_train)
        
        return np.asarray(gen_prob)
     
    #%% Prediction Results 
    def get_prediction(self, x_train, m_train):

        val_prediction = self.valfunction.predict(x_train)

        dis_prediction = self.discriminator.predict([x_train, m_train])

        return np.asarray(val_prediction), np.asarray(dis_prediction)
