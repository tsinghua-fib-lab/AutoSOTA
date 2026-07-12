import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from scVAEIT.utils import ModalMaskGenerator
from scVAEIT.nn_utils import Encoder, Decoder, LatentSpace
from tensorflow.keras.layers import Layer, Dense, BatchNormalization
from tensorflow.keras.utils import Progbar

            
            
class VariationalAutoEncoder(tf.keras.Model):
    """
    Combines the encoder, decoder and LatentSpace into an end-to-end model for training and inference.
    """
    def __init__(self, config, masks, name = 'autoencoder', **kwargs):
        '''
        Parameters
        ----------
        dimensions : np.array
            The dimensions of hidden layers of the encoder.
        dim_latent : int
            The latent dimension of the encoder.
        
        dim_block : list of int
            (num_block,) The dimension of each input block.        
        dist_block : list of str
            (num_block,) `'NB'`, `'ZINB'`, `'Bernoulli'` or `'Gaussian'`.
        dim_block_enc : list of int
            (num_block,) The dimension of output of first layer of the encoder for each block.
        dim_block_dec : list of int
            (num_block,) The dimension of output of last layer of the decoder for each block.        
        block_names : list of str, optional
            (num_block,) The name of first layer for each block.
        name : str, optional
            The name of the layer.
        **kwargs : 
            Extra keyword arguments.
        '''
        super(VariationalAutoEncoder, self).__init__(name = name, **kwargs)
        self.config = config
        self.masks = masks
        self.embed_layer = Dense(np.sum(self.config.dim_block_embed), 
                                 activation = tf.nn.tanh, name = 'embed')
        self.encoder = Encoder(self.config.dimensions, self.config.dim_latent,
            self.config.dim_block, self.config.dim_block_enc, self.config.dim_block_embed, self.config.block_names)
        self.decoder = Decoder(self.config.dimensions[::-1], self.config.dim_block,
            self.config.dist_block, self.config.dim_block_dec, self.config.dim_block_embed, self.config.block_names)
        
        self.mask_generator = ModalMaskGenerator(
            config.dim_input_arr, config.p_feat, config.p_modal)
        
        
    def generate_mask(self, inputs, mask=None, p=None):
        return self.mask_generator(inputs, mask, p)
        
        
    def init_latent_space(self, n_clusters, mu, log_pi=None):
        '''Initialze the latent space.

        Parameters
        ----------
        n_clusters : int
            The number of vertices in the latent space.
        mu : np.array
            \([d, k]\) The position matrix.
        log_pi : np.array, optional
            \([1, K]\) \(\\log\\pi\).
        '''
        self.n_states = n_clusters
        self.latent_space = LatentSpace(self.n_states, self.config.dim_latent)
        self.latent_space.initialize(mu, log_pi)

    def call(self, x, masks, batches,
             pre_train = False, L=1, training=True
             # alpha=0.0
            ):
        '''Feed forward through encoder, LatentSpace layer and decoder.

        Parameters
        ----------
        x_normalized : np.array
            \([B, G]\) The preprocessed data.
        c_score : np.array
            \([B, s]\) The covariates \(X_i\), only used when `has_cov=True`.
        x : np.array, optional
            \([B, G]\) The original count data \(Y_i\), only used when data_type is not `'Gaussian'`.
        scale_factor : np.array, optional
            \([B, ]\) The scale factors, only used when data_type is not `'Gaussian'`.
        pre_train : boolean, optional
            Whether in the pre-training phare or not.
        L : int, optional
            The number of MC samples.
        alpha : float, optional
            The penalty parameter for covariates adjustment.

        Returns
        ----------
        losses : float
            the loss.
        '''
        if not pre_train and not hasattr(self, 'latent_space'):
            raise ReferenceError('Have not initialized the latent space.')
                            
        z_mean_obs, z_log_var_obs, z_obs, log_probs_obs = self._get_reconstruction_loss(
            x, masks!=-1., masks!=-1., batches, L, training=training)
        z_mean_unobs_1, z_log_var_unobs_1, z_unobs_1, log_probs_unobs = self._get_reconstruction_loss(
            x, masks==0., masks==1., batches, L, training=training)
        z_mean_unobs_2, z_log_var_unobs_2, z_unobs_2, log_probs_ = self._get_reconstruction_loss(
            x, masks==1., masks==0., batches, L, training=training)
        log_probs_unobs = (1-self.config.beta_reverse) * log_probs_unobs + self.config.beta_reverse*log_probs_

#         if alpha>0.0:
#             zero_in = tf.concat([tf.zeros([z.shape[0],1,z.shape[2]], dtype=tf.keras.backend.floatx()), 
#                                 tf.tile(tf.expand_dims(c_score,1), (1,1,1))], -1)
#             reconstruction_zero_loss = self._get_reconstruction_loss(x, zero_in, scale_factor, 1)
#             reconstruction_z_loss = (1-alpha)*reconstruction_z_loss + alpha*reconstruction_zero_loss
        
        self.add_loss(
            [- (1-self.config.beta_unobs) * 
             tf.reduce_sum(tf.where(self.config.block_names==name, log_probs_obs, 0.)) * 
             self.config.beta_modal[i] for i,name in enumerate(self.config.uni_block_names)]
        )
        self.add_loss(
            [- self.config.beta_unobs * 
             tf.reduce_sum(tf.where(self.config.block_names==name, log_probs_unobs, 0.)) * 
             self.config.beta_modal[i] for i,name in enumerate(self.config.uni_block_names)]
        )
        
        kl = (1-self.config.beta_reverse) * self._get_kl_normal(
            z_mean_unobs_1, z_log_var_unobs_1, z_mean_obs, z_log_var_obs) + \
            self.config.beta_reverse * self._get_kl_normal(
            z_mean_unobs_2, z_log_var_unobs_2, z_mean_obs, z_log_var_obs)
        self.add_loss(self.config.beta_kl * kl)
        
        if not pre_train:
            log_p_z_obs, E_qzx_obs = self._get_kl_loss(z_obs, z_log_var_obs, training=training)
            log_p_z_unobs_1, E_qzx_unobs_1 = self._get_kl_loss(z_unobs_1, z_log_var_unobs_1, training=training)
            log_p_z_unobs_2, E_qzx_unobs_2 = self._get_kl_loss(z_unobs_2, z_log_var_unobs_2, training=training)
                            
            # - E_q[log p(z)]
            self.add_loss(
                -((1-self.config.beta_unobs) * log_p_z_obs + 
                  self.config.beta_unobs * (
                      (1-self.config.beta_reverse) * log_p_z_unobs_1 + 
                      self.config.beta_reverse * log_p_z_unobs_2) )
            )

            # Eq[log q(z|x)]
            self.add_loss(
                (1-self.config.beta_unobs) * E_qzx_obs + 
                  self.config.beta_unobs * (
                      (1-self.config.beta_reverse) * E_qzx_unobs_1 + 
                      self.config.beta_reverse * E_qzx_unobs_2)
            )

        return self.losses
    
    @tf.function
    def _get_reconstruction_loss(self, x, bool_mask_in, bool_mask_out, batches, L, training=True):
        '''
        Parameters
        ----------
        bool_mask_in : tf.Tensor of type tf.bool
            False indicates missing.
        bool_mask_out : tf.Tensor of type tf.bool
            Compute likelihood for entries with value True.
        '''
        _masks = tf.where(bool_mask_in, 0., 1.)
        _x = tf.where(bool_mask_in, x, 0.)
        embed = self.embed_layer(_masks)
        z_mean, z_log_var, z = self.encoder(_x, embed, batches, L, training=training)       
        log_probs = tf.reduce_mean(
            self.decoder(x, embed, bool_mask_out, batches, z, training=training), axis=0)
        return z_mean, z_log_var, z, log_probs
    
    
    @tf.function
    def _get_kl_normal(self, mu_0, log_var_0, mu_1, log_var_1):
        kl = 0.5 * (
            tf.exp(tf.clip_by_value(log_var_0-log_var_1, -6., 6.)) + 
            (mu_1 - mu_0)**2 / tf.exp(tf.clip_by_value(log_var_1, -6., 6.)) - 1.
             + log_var_1 - log_var_0)
        return tf.reduce_mean(tf.reduce_sum(kl, axis=-1))
    
    @tf.function
    def _get_kl_loss(self, z, z_log_var, training=True):
        log_p_z = self.latent_space(z, training=training)

        E_qzx = - tf.reduce_mean(
            0.5 * self.config.dim_latent *
            (tf.math.log(tf.constant(2 * np.pi, tf.keras.backend.floatx())) + 1.0) +
            0.5 * tf.reduce_sum(z_log_var, axis=-1)
            )
        return log_p_z, E_qzx
    
    
    def get_recon(self, dataset_test, masks=None, zero_out=True, L=50):
        if masks is None:
            masks = self.masks
        x_hat = []
        for x,b,id_data in dataset_test:
            m = tf.gather(masks, id_data)
            _m = tf.where(m==0., 0., 1.)
            embed = self.embed_layer(_m)
            if zero_out:
                x = tf.where(m==0, x, 0.)
            _, _, z = self.encoder(x, embed, b, L, False)
            _x_hat = tf.reduce_mean(
                self.decoder(x, embed, tf.ones_like(m,dtype=tf.bool), 
                    b, z, training=False, return_prob=False), axis=1)
            x_hat.append(_x_hat.numpy())
        x_hat = np.concatenate(x_hat)        

        return x_hat
    
    
    def get_z(self, dataset_test, masks=None):
        '''Get \(q(Z_i|Y_i,X_i)\).

        Parameters
        ----------
        dataset_test : tf.Dataset
            Dataset containing (x, batches).

        Returns
        ----------
        z_mean : np.array
            \([B, d]\) The latent mean.
        '''
        if masks is None:
            masks = self.masks
        z_mean = []
        for x,b,id_data in dataset_test:
            m = tf.gather(masks, id_data)
            m = tf.where(m==0., 0., 1.)
            embed = self.embed_layer(m)
            _z_mean, _, _ = self.encoder(x, embed, b, 1, False)         
            z_mean.append(_z_mean.numpy())
        z_mean = np.concatenate(z_mean)        

        return z_mean


    
