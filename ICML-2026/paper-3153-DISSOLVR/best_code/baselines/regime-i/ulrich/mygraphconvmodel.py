# mygraphconvmodel.py
""" Klasse MyGraphConvModel, own Keras Model """
import tensorflow as tf 
from deepchem.models.layers import GraphConv, GraphPool, GraphGather
import keras.layers as layers

class MyGraphConvModel(tf.keras.Model):

  def __init__(self,batch_size=50,neuronslayer1=16,neuronslayer2=32,dropout=0.1):
    super(MyGraphConvModel, self).__init__()
    self.gc1 = GraphConv(neuronslayer1, activation_fn=tf.nn.leaky_relu)
    self.batch_norm1 = layers.BatchNormalization()
    self.gp1 = GraphPool()

    self.gc2 = GraphConv(neuronslayer2, activation_fn=tf.nn.leaky_relu)
    self.batch_norm2 = layers.BatchNormalization()
    self.gp2 = GraphPool()

    self.dense1 = layers.Dense(256, activation=tf.nn.leaky_relu)
    self.batch_norm3 = layers.BatchNormalization()
    self.dropout = layers.Dropout(dropout)
    self.readout = GraphGather(batch_size=batch_size, activation_fn=tf.nn.leaky_relu)

    self.regression = layers.Dense(1,activation=None)

  def call(self, inputs):
    gc1_output = self.gc1(inputs)
    batch_norm1_output = self.batch_norm1(gc1_output)
    gp1_output = self.gp1([batch_norm1_output] + inputs[1:])

    gc2_output = self.gc2([gp1_output] + inputs[1:])
    batch_norm2_output = self.batch_norm2(gc2_output)
    gp2_output = self.gp2([batch_norm2_output] + inputs[1:])

    dense1_output = self.dense1(gp2_output)
    batch_norm3_output = self.batch_norm3(dense1_output)
    dropout = self.dropout(batch_norm3_output)
    readout_output = self.readout([dropout] + inputs[1:])

    return self.regression(readout_output)
