from tensorflow.keras.layers import Dense, LeakyReLU
from .masked_model import MaskedModel
import tensorflow as tf
import numpy as np


class MaskedNN(MaskedModel):

    def _get_mse_loss(self, X, W_prime):
        """
        Different model for different nodes to use masked features to predict value for each node
        """
        mse_loss = 0
        for i in range(self.d):
            pns_parents = np.where(self.pns_mask[:, i] == 1)[0]
            possible_parents = [int(j) for j in pns_parents if j != i]
            if len(possible_parents) == 0:  # Root node, don't have to build NN in this case
                continue

            curr_X = tf.gather(X, indices=possible_parents, axis=1)  # Features for current node
            curr_y = tf.gather(X, indices=i, axis=1)  # Label for current node
            curr_W = tf.gather(tf.gather(W_prime, indices=i, axis=1),
                               indices=possible_parents, axis=0)  # Mask for current node

            curr_masked_X = curr_X * curr_W  # Broadcasting
            curr_y_pred = self._forward(curr_masked_X)  # Use masked features to predict value of current node

            mse_loss += tf.reduce_sum(tf.square(tf.squeeze(curr_y_pred) - curr_y))

        return mse_loss

    def _forward(self, x):
        for _ in range(self.num_hidden_layers):    # Hidden layer
            x = Dense(self.hidden_size, activation=None, kernel_initializer=self.initializer)(x)
            x = LeakyReLU(alpha=0.05)(x)

        return Dense(1, kernel_initializer=self.initializer)(x)    # Final output layer
