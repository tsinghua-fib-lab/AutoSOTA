import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Dense, Input, Multiply, Reshape, 
                                     Flatten, Lambda, Activation, BatchNormalization, Layer)
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras import regularizers, layers
from tensorflow.keras import backend as K
import math

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from realx_main.sample import REBAR_Bernoulli_Sampler, Random_Bernoulli_Sampler


# REAL-X Class
class REALX():
    def __init__(self, selector, predictor, lamda, output_channels=1, task='classification'):

        self.selector = selector
        self.selector._name = 'selector'
        self.has_selector = False

        self.task = task
        self.lamda = lamda
        self.input_shape = tuple(selector.input.shape[1:])
        self.mask_size = selector.output.shape[1]
        self.output_channels = output_channels
        self.resize_layer = ResizeMask(self.input_shape, self.mask_size, self.output_channels)
        
        self.predictor = predictor
        self.predictor._name = 'predictor'
        self.build_predictor()
        self.has_predictor = False
        
        self.evalx = clone_model(predictor)
        self.evalx._name = 'evalx'
        self.build_evalx()
        self.has_evalx = False
         
    def build_predictor(self):
        model_input = Input(shape=(*self.input_shape,), dtype='float32')
        
        sample_input = Flatten()(model_input)
        sample_input = Lambda(lambda x: x[:, :self.mask_size])(sample_input)
        r = Random_Bernoulli_Sampler()(sample_input)
        r = self.resize_layer(r)
        masked_input = Multiply()([model_input, r])
        
        preds = self.predictor(masked_input)

        model = Model(model_input, preds)
        model.summary()
        
        self.predictor = model
        print("Predictor Built!")
        
    def build_selector(self):
        self.selector = SELECTOR(self.selector, self.predictor.get_layer('predictor'),
                                 self.lamda, self.output_channels, task=self.task)
        print("Selector Built!")
        
    def build_evalx(self):
        model_input = Input(shape=(*self.input_shape,), dtype='float32')
        
        sample_input = Flatten()(model_input)
        sample_input = Lambda(lambda x: x[:, :self.mask_size])(sample_input)
        r = Random_Bernoulli_Sampler()(sample_input)
        r = self.resize_layer(r)
        masked_input = Multiply()([model_input, r])
        
        preds = self.evalx(masked_input)

        model = Model(model_input, preds)
        model.summary()
        
        self.evalx = model
        print("Evaluator Built!")
        
    def predict(self, x, batch_size):
        if not self.has_predictor:
            
            self.predictor_model = self.predictor.get_layer('predictor')
            self.predictor_model.compile(loss=None, optimizer='rmsprop', metrics=None)
        
        predictions = self.predictor_model.predict(x, verbose=0, batch_size=batch_size)
        
        return predictions
        
    def evaluate(self, x, batch_size):
        if not self.has_evalx:
            model_input = Input(shape=(*self.input_shape, ), dtype='float32')
            logits = self.selector.selector(model_input)
            _, s, _ = REBAR_Bernoulli_Sampler(tau0=.1)(logits)
            s = self.resize_layer(s)
            masked_input = Multiply()([model_input, s])
            preds = self.evalx.get_layer('evalx')(masked_input)
            
            self.evalx_model = Model(model_input, preds)
            self.evalx_model.compile(loss=None, optimizer='rmsprop', metrics=None)
            
        predictions = self.evalx_model.predict(x, verbose=0, batch_size=batch_size)
        
        return predictions
        
    def select(self, x, batch_size, discrete=False):
        
        if not self.has_selector:
            model_input = Input(shape=(*self.input_shape, ), dtype='float32', name='ecg')
            logits = self.selector.selector(model_input)
            s = Activation(tf.keras.activations.sigmoid)(logits)
            s = self.resize_layer(s)
        
            self.selector_model = Model(model_input, s) 
            self.selector_model.compile(loss=None, optimizer='rmsprop', metrics=None)

        explainations = self.selector_model.predict(x, verbose=0, batch_size=batch_size)
        if discrete:
            explainations = (explainations > .5).astype(int)
        # else: #@TAL removed
        #     explainations = K.sigmoid(explainations)
        
        return explainations



# Model Class to Train Selector w/ REBAR gradients
class SELECTOR(keras.Model):
    def __init__(self, selector, predictor, lamda, output_channels, task='classification'):
        super(SELECTOR, self).__init__()

        self.selector = selector
        self.task = task

        self.in_shape = tuple(selector.input.shape[1:])
        self.mask_size = selector.output.shape[1]

        self.resize_layer = ResizeMask(self.in_shape, self.mask_size, output_channels)

        self.predictor = self.build_predictor(predictor)
        self.lamda = lamda
    
    def build_predictor(self, predictor):
        model_input = predictor.input
        s_in = Input(shape=(self.mask_size, ), dtype='float32') 
        
        s = self.resize_layer(s_in)
        masked_input = Multiply()([model_input, s])
        y_pred = predictor(masked_input)
        
        model = Model([model_input, s_in], y_pred)
        model._name = 'predictor'
        
        return model     
        
    def call(self, inputs):
        sel_prob = self.selector(inputs)
        z, s, z_tilde = self.sampler(sel_prob)
        y_pred = self.predictor([inputs, s])
    
        return y_pred

    def train_step(self, data):
        
        x_batch, y_batch = data

        with tf.GradientTape(persistent=True) as tape:
            # Generate a batch of probabilities of feature selection
            sel_prob = self.selector(x_batch, training=True)

            # Sampling the features based on the generated probability
            z, s, z_tilde = REBAR_Bernoulli_Sampler(tau0=.1)(sel_prob)

            # Calculate
            # 1. f(s)
            f_s = self.predictor([x_batch, s], training=True)
            # 2. c(z)
            c_z = self.predictor([x_batch, z], training=True)
            # 3. c(z~)
            c_z_tilde = self.predictor([x_batch, z_tilde], training=True)

            # Compute the log-likelihoods (CE for classification/multiclass, BCE for multilabel)
            if self.task == 'multilabel':
                p_f_s     = tf.reduce_sum(y_batch * K.log(f_s     + 1e-5) + (1 - y_batch) * K.log(1 - f_s     + 1e-5), axis=1)
                p_c_z     = tf.reduce_sum(y_batch * K.log(c_z     + 1e-5) + (1 - y_batch) * K.log(1 - c_z     + 1e-5), axis=1)
                p_c_z_tilde = tf.reduce_sum(y_batch * K.log(c_z_tilde + 1e-5) + (1 - y_batch) * K.log(1 - c_z_tilde + 1e-5), axis=1)
            else:
                p_f_s     = tf.reduce_sum(y_batch * K.log(f_s     + 1e-5), axis=1)
                p_c_z     = tf.reduce_sum(y_batch * K.log(c_z     + 1e-5), axis=1)
                p_c_z_tilde = tf.reduce_sum(y_batch * K.log(c_z_tilde + 1e-5), axis=1)
            # 4. q(s)
            sel_prob = K.sigmoid(sel_prob)
            q_s = tf.reduce_sum( s * K.log(sel_prob + 1e-5) + (1-s) * K.log(1-sel_prob + 1e-5), axis = 1)

            #Compute the Sparisity Regularization
            # 1. R(s)
            R_s = tf.reduce_mean(s, axis = 1)
            R_s_approx = tf.reduce_mean(sel_prob, axis = 1)

            # Reward
            Reward = tf.stop_gradient(p_f_s - p_c_z_tilde)

            # Terms to Make Expection Zero
            E_0 = p_c_z - p_c_z_tilde

            # Losses
            s_loss = Reward*q_s - self.lamda*(R_s_approx) + E_0
              
            # Train
            sf_grads = tape.gradient(-s_loss, self.selector.trainable_variables)
            self.optimizer.apply_gradients(zip(sf_grads, self.selector.trainable_variables),
                                           experimental_aggregate_gradients=False)
        
        # Calculate Objective Loss
        loss = tf.reduce_mean(p_f_s - self.lamda*(R_s))
        obj_dict = {'loss': loss, 'sel%': tf.reduce_mean(R_s)}
        
        #Metrics
        self.compiled_metrics.update_state(y_batch, f_s)
        metrics_dict = {m.name: m.result() for m in self.metrics}

        return {**obj_dict, **metrics_dict}
    
    def test_step(self, data):
        
        x_batch, y_batch = data
        
        # Generate a batch of probabilities of feature selection
        sel_prob = self.selector(x_batch, training=True)

        # Sampling the features based on the generated probability
        z, s, z_tilde = self.sampler(sel_prob)

        # Calculate
        # 1. f(s)
        f_s = self.predictor([x_batch, s], training=True)
        
        # Compute the probabilities
        # 1. f(s)
        if self.task == 'multilabel':
            p_f_s = tf.reduce_sum(y_batch * K.log(f_s + 1e-5) + (1 - y_batch) * K.log(1 - f_s + 1e-5), axis=1)
        else:
            p_f_s = tf.reduce_sum(y_batch * K.log(f_s + 1e-5), axis=1)
        
        #Compute the Sparisity Regularization
        # 1. R(s)
        R_s = tf.reduce_mean(s, axis = 1)
        R_s_approx = tf.reduce_mean(sel_prob, axis = 1)
        
        #Loss
        loss = tf.reduce_mean(p_f_s - self.lamda*(R_s))
        obj_dict = {'loss': loss, 'sel%': tf.reduce_mean(R_s)}
        
        #Metrics
        self.compiled_metrics.update_state(y_batch, f_s)
        metrics_dict = {m.name: m.result() for m in self.metrics if m.name != 'loss'}
        
        return {**obj_dict, **metrics_dict}
    

# Custom Layer to Resize Mask (Allows For Use of "Super Pixel Selections")
class ResizeMask(layers.Layer):
    def __init__(self, in_shape, mask_size, output_channels=1, **kwargs):
        super(ResizeMask, self).__init__(**kwargs)
        
        self.in_shape = in_shape
        self.mask_size = mask_size
        self.output_channels = output_channels
        self.reshape_shape, self.resize_aspect, self.pad_shape = self.get_reshape_shape()
        
    def get_reshape_shape(self):
        
        #Check if Multi Dimensional
        if len(self.in_shape) == 1:
            out_shape = self.mask_size
            resize_aspect = int(math.ceil(self.in_shape[0]/self.mask_size))
            
            #Get Pad Length Used
            resize_shape = out_shape * resize_aspect
            pad_shape = int((resize_shape - self.in_shape[0])/2)
            
            return out_shape, resize_aspect, pad_shape
        else:
            #Get Input Dimensions Ratio
            input_shape = np.array(list(self.in_shape)[:-1])
            gcd = np.gcd.reduce(input_shape)
            ratio = input_shape/gcd
            #Get Working Mask Size and Aspect Ratio
            mask_size = self.mask_size/self.output_channels
            aspect = (mask_size/np.prod(ratio))**(1/len(ratio))
            out_shape = (ratio * aspect).astype(int)
            resize_aspect = int(math.ceil(gcd/aspect))
            
            #Get Pad Length Used
            resize_shape = out_shape * resize_aspect
            pad_shape = ((resize_shape - input_shape)/2).astype(int)
        
            return (*out_shape, self.output_channels), resize_aspect, pad_shape
    
    def call(self, inputs):
        
        if len(self.in_shape) == 1:
            #Resize
            out = Lambda(
                lambda x: K.repeat_elements(x, rep = self.resize_aspect, axis = 1)
            )(inputs)
            
            #Slice to Input Size
            if self.pad_shape > 0:
                out = Lambda(lambda x: x[:, self.pad_shape:-self.pad_shape])(out)
            
        else:
            #Reshape
            out = Reshape(tuple(self.reshape_shape))(inputs)
            
            #Resize
            for i in range(len(self.reshape_shape)-1):
                out = Lambda(
                    lambda x: K.repeat_elements(x, rep = self.resize_aspect, axis = i+1)
                )(out)
                
            #Crop to Input Size
            if len(self.pad_shape) == 1:
                out = Lambda(lambda x: x[:, self.pad_shape[0]:-self.pad_shape[0], :])(out)
            elif len(self.pad_shape) == 2:
                out = Lambda(
                    lambda x: x[:, 
                                self.pad_shape[0]:-self.pad_shape[0],
                                self.pad_shape[1]:-self.pad_shape[1],
                                :]
                )(out)
        
        
        return out  