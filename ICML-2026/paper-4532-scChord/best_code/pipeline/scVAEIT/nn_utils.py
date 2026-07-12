import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LeakyReLU, Lambda
use_bias = True


###########################################################################
#
# Input and output blocks for multimodal datasets
#
###########################################################################
class InputBlock(tf.keras.layers.Layer):
    def __init__(self, dim_inputs, dim_latents, dim_embed, names=None, bn=False, **kwargs):
        '''
        Parameters
        ----------
        dim_inputs : list of int
            (B+1,) The dimension of each input block, where the last block 
            is assumed to be the batch effects.
        dim_latent : list of int
            (B,) The dimension of output of first layer for each block.
        names : list of str, optional
            (B,) The name of first layer for each block.
        **kwargs : 
            Extra keyword arguments.
        '''
        super(InputBlock, self).__init__()
                
        self.dim_inputs = dim_inputs
        self.dim_embed = dim_embed
        if names is None:
            names = ['Block_{}'.format(i) for i in range(len(dim_latents))]
        self.names = names
        self.dim_latents = dim_latents
        
        self.linear_layers = [
            Dense(d, use_bias=False, activation = LeakyReLU(), name=names[i]) if d>0 else 
            tf.keras.layers.Lambda(lambda x,training: tf.identity(x))
                for i,d in enumerate(self.dim_latents)
        ]
        if bn:
            self.bn = BatchNormalization(center=False)
        else:
            self.bn = Lambda(lambda x,training: tf.identity(x))
        self.concat = tf.keras.layers.Concatenate()
        

    @tf.function
    def call(self, x, embed, batches, training=True):
        x_list = tf.split(x, self.dim_inputs, axis=1)
        embed_list = tf.split(embed, self.dim_embed, axis=1)
        outputs = self.concat([
            self.linear_layers[i](
                tf.concat([x_list[i], embed_list[i], batches], axis=1), training=training
                ) for i in range(len(self.dim_latents))])
        outputs = self.bn(outputs, training=training)
        
        return outputs



def get_dist(dist, x_hat, mask, disp):
    if dist=='NB':
        generative_dist = tfd.Independent(tfd.Masked(
                tfd.NegativeBinomial.experimental_from_mean_dispersion(
                    mean = x_hat * tf.math.log(10**4+1.), 
                    dispersion = disp, name='NB_rv'
                ), mask), reinterpreted_batch_ndims=1)

    elif dist=='ZINB':
        # Not tested in graph mode yet
        dim = tf.cast(tf.shape(x_hat)[-1]/2, tf.int32)
        phi_rna = tf.clip_by_value(x_hat[..., dim:], 1e-5, 1.-1e-5)
        x_hat = x_hat[..., :dim]
        generative_dist = tfd.Independent(tfd.Masked(
            tfd.Mixture(
                cat=tfd.Categorical(
                    probs=tf.stack([phi_rna, 1.0 - phi_rna], axis=-1)),
                components=[tfd.Deterministic(loc=tf.zeros_like(phi_rna)), 
                            tfd.NegativeBinomial.experimental_from_mean_dispersion(
                                mean = x_hat * tf.math.log(10**4+1.),
                                dispersion = disp)],
                name='ZINB_rv'
            ), mask), reinterpreted_batch_ndims=1)

    elif dist=='Bernoulli':
        generative_dist = tfd.Independent(tfd.Masked(
            tfd.Bernoulli(
                probs = tf.clip_by_value(x_hat, 1e-5, 1.-1e-5),
                dtype=tf.float32, name='Bernoulli_rv'
            ), mask), reinterpreted_batch_ndims=1)  

    elif dist=='Gaussian':
        generative_dist = tfd.Independent(tfd.Masked(
            tfd.Normal(
                loc = x_hat, scale = disp, name='Gaussian_rv'
            ), mask), reinterpreted_batch_ndims=1)

    return generative_dist



class OutputBlock(tf.keras.layers.Layer):
    def __init__(self, dim_outputs, dist_outputs, dim_latents, dim_embed, names=None, bn=True, **kwargs):
        '''
        Parameters
        ----------
        dim_outputs : list of int
            (B,) The dimension of each output block.
        dist_outputs : list of str
            (B,) The distribution of each output block.
        dim_latents : list of int
            (B,) The dimension of output of last layer for each block.
        names : list of str, optional
            (B,) The name of last layer for each block.
        bn : boolean
            Whether use batch normalization or not.
        **kwargs : 
            Extra keyword arguments.
        '''        
        super(OutputBlock, self).__init__()
        self.dim_inputs = dim_outputs
        self.dim_embed = dim_embed
        self.dim_outputs = [d*2 if dist_outputs[i]=='ZINB' else d for i,d in enumerate(dim_outputs)]
        self.dist_outputs = dist_outputs
        self.dim_latents = dim_latents
        if names is None:
            names = ['Block_{}'.format(i) for i in range(len(dim_latents))]
        self.names = names        
        
        self.linear_layers = [
            Dense(d, use_bias=use_bias, activation = LeakyReLU(), name=names[i]) if d>0 else 
            Lambda(lambda x,training: tf.identity(x))
                for i,d in enumerate(self.dim_latents)
        ]
        if bn:
            self.bn = [BatchNormalization(center=False) for _ in range(len(dim_latents))]
        else:
            self.bn = [Lambda(lambda x,training: tf.identity(x)) for _ in range(len(dim_latents))]
        out_act = [None if dist=='Gaussian' else tf.nn.sigmoid for dist in self.dist_outputs]
        self.output_layers = [
            Dense(d, use_bias=use_bias, name=names[i], activation = out_act[i]) 
            for i,d in enumerate(self.dim_outputs)
        ]        

        self.disp = [
            Dense(d, use_bias=False, activation = tf.nn.softplus, name="disp".format(names[i])) 
            if self.dist_outputs[i]!='Bernoulli' else 
            Lambda(lambda x,training: tf.zeros((1,d), dtype=tf.float32))
                for i,d in enumerate(self.dim_inputs)
        ]        
        
        self.dists = [Lambda(lambda x: get_dist(x[0], x[1], x[2], x[3])) 
                      for dist in self.dist_outputs]
        self.concat = tf.keras.layers.Concatenate()
        
        
    @tf.function
    def call(self, x, embed, masks, batches, z, training=True):
        '''
        Parameters
        ----------
        x : tf.Tensor
            \([B, D]\) the observed \(x\).
        z : tf.Tensor
            \([B, L, d]\) the sampled \(z\).
        batches : tf.Tensor
            \([B, b]\) the sampled \(z\).
        masks : tf.Tensor
            \([B, D]\) the mask indicating feature missing.
        training : boolean, optional
            whether in the training or inference mode.
        '''

        m_list = tf.split(tf.expand_dims(masks,1), self.dim_inputs, axis=-1)
        x_list = tf.split(tf.expand_dims(x,1), self.dim_inputs, axis=-1)

        L = tf.shape(z)[1]
        probs = self.concat([
            self.dists[i]([
                self.dist_outputs[i],
                self.output_layers[i](self.bn[i](self.linear_layers[i](
                    z,
                    training=training), training=training), training=training), 
                m_list[i], 
                tf.expand_dims(
                    tfp.math.clip_by_value_preserve_gradient(
                    self.disp[i](batches, training=training), 0., 6.), 1)
            ]).log_prob(x_list[i]) for i in range(len(self.dim_latents))
        ])

        return probs

    
    @tf.function
    def get_recon(self, embed, masks, batches, z, training=True):
        m_list = tf.split(tf.expand_dims(masks,1), self.dim_inputs, axis=-1)
        # embed_list = tf.split(embed, self.dim_embed, axis=-1)
        L = tf.shape(z)[1]
        x_hat = self.concat([
            self.dists[i]([
                self.dist_outputs[i],
                self.output_layers[i](self.bn[i](self.linear_layers[i](
                    z,
                    training=training), training=training), training=training), 
                m_list[i], 
                tf.expand_dims(
                    tfp.math.clip_by_value_preserve_gradient(
                    self.disp[i](batches, training=training), 0., 6.), 1)
            ]).mean() for i in range(len(self.dim_latents))
        ])

        return x_hat


###########################################################################
#
# Sampling layers in the latent space
#
###########################################################################
class cdf_layer(Layer):
    '''
    The Normal cdf layer with custom gradients.
    '''
    def __init__(self):
        '''
        '''
        super(cdf_layer, self).__init__()
        
    @tf.function
    def call(self, x):
        return self.func(x)
        
    @tf.custom_gradient
    def func(self, x):
        '''Return cdf(x) and pdf(x).

        Parameters
        ----------
        x : tf.Tensor
            The input tensor.
        
        Returns
        ----------
        f : tf.Tensor
            cdf(x).
        grad : tf.Tensor
            pdf(x).
        '''   
        dist = tfp.distributions.Normal(
            loc = tf.constant(0.0, tf.keras.backend.floatx()), 
            scale = tf.constant(1.0, tf.keras.backend.floatx()), 
            allow_nan_stats=False)
        f = dist.cdf(x)
        def grad(dy):
            gradient = dist.prob(x)
            return dy * gradient
        return f, grad
    

class Sampling(Layer):
    """Sampling latent variable \(z\) from \(N(\\mu_z, \\log \\sigma_z^2\)).    
    Used in Encoder.
    """
    def __init__(self, seed=0, **kwargs):
        super(Sampling, self).__init__(**kwargs)
        self.seed = seed

    @tf.function
    def call(self, z_mean, z_log_var):
        '''Return cdf(x) and pdf(x).

        Parameters
        ----------
        z_mean : tf.Tensor
            \([B, L, d]\) The mean of \(z\).
        z_log_var : tf.Tensor
            \([B, L, d]\) The log-variance of \(z\).

        Returns
        ----------
        z : tf.Tensor
            \([B, L, d]\) The sampled \(z\).
        '''   
   #     seed = tfp.util.SeedStream(self.seed, salt="random_normal")
   #     epsilon = tf.random.normal(shape = tf.shape(z_mean), seed=seed(), dtype=tf.keras.backend.floatx())
        epsilon = tf.random.normal(shape = tf.shape(z_mean), dtype=tf.keras.backend.floatx())
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        z = tf.clip_by_value(z, -1e6, 1e6)
        return z



###########################################################################
#
# Encoder
# 
###########################################################################
class Encoder(Layer):
    '''
    Encoder, model \(p(Z_i|Y_i,X_i)\).
    '''
    def __init__(self, dimensions, dim_latent, 
        dim_block_inputs, dim_block_latents, dim_embed, block_names=None, name='encoder', **kwargs):
        '''
        Parameters
        ----------
        dimensions : np.array
            The dimensions of hidden layers of the encoder.
        dim_latent : int
            The latent dimension of the encoder.
        dim_block_inputs : list of int
            (num_block,) The dimension of each input block, where the last block 
            is assumed to be the batch effects.
        dim_block_latents : list of int
            (num_block,) The dimension of output of first layer for each block.
        block_names : list of str, optional
            (num_block,) The name of first layer for each block.  
        name : str, optional
            The name of the layer.
        **kwargs : 
            Extra keyword arguments.
        ''' 
        super(Encoder, self).__init__(name = name, **kwargs)
        self.input_layer = InputBlock(dim_block_inputs, dim_block_latents, dim_embed, block_names, bn=False)
        self.dense_layers = [Dense(dim, activation = LeakyReLU(),
                                          name = 'encoder_%i'%(i+1)) \
                             for (i, dim) in enumerate(dimensions)]
        self.batch_norm_layers = [BatchNormalization(center=False) \
                                    for _ in range(len(dimensions))]
        self.batch_norm_layers.append(BatchNormalization(center=False))
        self.latent_mean = Dense(dim_latent, name = 'latent_mean')
        self.latent_log_var = Dense(dim_latent, name = 'latent_log_var')
        self.sampling = Sampling()
    
    
    @tf.function
    def call(self, x, embed, batches, L=1, training=True):
        '''Encode the inputs and get the latent variables.

        Parameters
        ----------
        x : tf.Tensor
            \([B, L, d]\) The input.
        L : int, optional
            The number of MC samples.
        training : boolean, optional
            Whether in the training or inference mode.
        
        Returns
        ----------
        z_mean : tf.Tensor
            \([B, L, d]\) The mean of \(z\).
        z_log_var : tf.Tensor
            \([B, L, d]\) The log-variance of \(z\).
        z : tf.Tensor
            \([B, L, d]\) The sampled \(z\).
        '''
        _z = self.input_layer(x, embed, batches, training=training)
        for dense, bn in zip(self.dense_layers, self.batch_norm_layers):
            _z = dense(_z)
            _z = bn(_z, training=training)
        z_mean = self.batch_norm_layers[-1](self.latent_mean(_z), training=training)
        z_log_var = self.latent_log_var(_z)
        _z_mean = tf.tile(tf.expand_dims(z_mean, 1), (1,L,1))
        _z_log_var = tf.tile(tf.expand_dims(z_log_var, 1), (1,L,1))
        z = self.sampling(_z_mean, _z_log_var)
        return z_mean, z_log_var, z



###########################################################################
#
# Decoder
# 
###########################################################################
class Decoder(Layer):
    '''
    Decoder, model \(p(Y_i|Z_i,X_i)\).
    '''
    def __init__(self, dimensions, dim_block_outputs, 
        dist_block_outputs, dim_block_latents, dim_embed, block_names=None,
        name = 'decoder', **kwargs):
        '''
        Parameters
        ----------
        dimensions : np.array
            The dimensions of hidden layers of the encoder.
        dim_block_outputs : list of int
            (B,) The dimension of each output block.
        dist_block_outputs : list of str
            (B,) `'NB'`, `'ZINB'`, `'Bernoulli'` or `'Gaussian'`.
        dim_block_latents : list of int
            (B,) The dimension of output of last layer for each block.
        block_names : list of str, optional
            (B,) The name of last layer for each block.        
        name : str, optional
            The name of the layer.
        '''
        super(Decoder, self).__init__(name = name, **kwargs)
        self.output_layer = OutputBlock(
            dim_block_outputs, dist_block_outputs, dim_block_latents, dim_embed, block_names, bn=False)

        self.dense_layers = [Dense(dim, activation = LeakyReLU(),
                                          name = 'decoder_%i'%(i+1)) \
                             for (i,dim) in enumerate(dimensions)]
        self.batch_norm_layers = [BatchNormalization(center=False) \
                                    for _ in range(len((dimensions)))]

       
    @tf.function
    def call(self, x, embed, masks, batches, z, training=True, return_prob=True):
        '''Decode the latent variables and get the reconstructions.

        Parameters
        ----------
        z : tf.Tensor
            \([B, L, d]\) the sampled \(z\).
        training : boolean, optional
            whether in the training or inference mode.

        Returns
        ----------
        log_probs : tf.Tensor
            \([B, block]\) The log probability.
        '''
#         _z = z
        L = tf.shape(z)[1]
        _z = tf.concat([
            z, 
            tf.tile(tf.expand_dims(tf.concat([embed,batches], axis=-1), 1), (1,L,1))
        ], axis=-1)
        for dense, bn in zip(self.dense_layers, self.batch_norm_layers):
            _z = dense(_z)
            _z = bn(_z, training=training)

        if return_prob:
            log_probs = self.output_layer(x, embed, masks, batches, _z, training=training)
            return log_probs
        else:
            x_hat = self.output_layer.get_recon(embed, masks, batches, _z, training=training)
            return x_hat


###########################################################################
#
# Latent space
# 
###########################################################################
class LatentSpace(Layer):
    '''
    Layer for the Latent Space.
    '''
    def __init__(self, n_clusters, dim_latent,
            name = 'LatentSpace', seed=0, **kwargs):
        '''
        Parameters
        ----------
        n_clusters : int
            The number of vertices in the latent space.
        dim_latent : int
            The latent dimension.
        M : int, optional
            The discretized number of uniform(0,1).
        name : str, optional
            The name of the layer.
        **kwargs : 
            Extra keyword arguments.
        '''
        super(LatentSpace, self).__init__(name=name, **kwargs)
        self.dim_latent = dim_latent
        self.n_states = n_clusters
        self.n_categories = int(n_clusters*(n_clusters+1)/2)

        # nonzero indexes
        # A = [0,0,...,0  , 1,1,...,1,   ...]
        # B = [0,1,...,k-1, 1,2,...,k-1, ...]
        self.A, self.B = np.nonzero(np.triu(np.ones(n_clusters)))
        self.A = tf.convert_to_tensor(self.A, tf.int32)
        self.B = tf.convert_to_tensor(self.B, tf.int32)
        self.clusters_ind = tf.boolean_mask(
            tf.range(0,self.n_categories,1), self.A==self.B)

        # [pi_1, ... , pi_K] in R^(n_categories)
        self.pi = tf.Variable(tf.ones([1, self.n_categories], dtype=tf.keras.backend.floatx()) / self.n_categories,
                                name = 'pi')
        
        # [mu_1, ... , mu_K] in R^(dim_latent * n_clusters)
        self.mu = tf.Variable(tf.random.uniform([self.dim_latent, self.n_states],
                                                minval = -1, maxval = 1, seed=seed, dtype=tf.keras.backend.floatx()),
                                name = 'mu')
        self.cdf_layer = cdf_layer()       
        
    def initialize(self, mu, log_pi):
        '''Initialize the latent space.

        Parameters
        ----------
        mu : np.array
            \([d, k]\) The position matrix.
        log_pi : np.array
            \([1, K]\) \(\\log\\pi\).
        '''
        # Initialize parameters of the latent space
        if mu is not None:
            self.mu.assign(mu)
        if log_pi is not None:
            self.pi.assign(log_pi)

    def normalize(self):
        '''Normalize \(\\pi\).
        '''
        self.pi = tf.nn.softmax(self.pi)

    @tf.function
    def _get_normal_params(self, z):
        batch_size = tf.shape(z)[0]
        L = tf.shape(z)[1]
        
        # [batch_size, L, n_categories]
        temp_pi = tf.tile(
            tf.expand_dims(tf.nn.softmax(self.pi), 1),
            (batch_size,L,1))
                        
        # [batch_size, L, d, n_categories]
        alpha_zc = tf.expand_dims(tf.expand_dims(
            tf.gather(self.mu, self.B, axis=1) - tf.gather(self.mu, self.A, axis=1), 0), 0)
        beta_zc = tf.expand_dims(z,-1) - \
            tf.expand_dims(tf.expand_dims(
            tf.gather(self.mu, self.B, axis=1), 0), 0)
            
        # [batch_size, L, n_categories]
        _inv_sig = tf.reduce_sum(alpha_zc * alpha_zc, axis=2)
        _nu = - tf.reduce_sum(alpha_zc * beta_zc, axis=2) * tf.math.reciprocal_no_nan(_inv_sig)
        _t = - tf.reduce_sum(beta_zc * beta_zc, axis=2) + _nu**2*_inv_sig
        return temp_pi, beta_zc, _inv_sig, _nu, _t
    
    @tf.function
    def _get_pz(self, temp_pi, _inv_sig, beta_zc, log_p_z_c_L):
        # [batch_size, L, n_categories]
        log_p_zc_L = - 0.5 * self.dim_latent * tf.math.log(tf.constant(2 * np.pi, tf.keras.backend.floatx())) + \
            tf.math.log(temp_pi) + \
            tf.where(_inv_sig==0, 
                    - 0.5 * tf.reduce_sum(beta_zc**2, axis=2), 
                    log_p_z_c_L)
        
        # [batch_size, L, 1]
        log_p_z_L = tf.reduce_logsumexp(log_p_zc_L, axis=-1, keepdims=True)
        
        # [1, ]
        log_p_z = tf.reduce_mean(log_p_z_L)
        return log_p_zc_L, log_p_z_L, log_p_z
    
    @tf.function
    def _get_posterior_c(self, log_p_zc_L, log_p_z_L):
        L = tf.shape(log_p_z_L)[1]

        # log_p_c_x     -   predicted probability distribution
        # [batch_size, n_categories]
        log_p_c_x = tf.reduce_logsumexp(
                        log_p_zc_L - log_p_z_L,
                    axis=1) - tf.math.log(tf.cast(L, tf.keras.backend.floatx()))
        return log_p_c_x

    @tf.function
    def _get_inference(self, z, log_p_z_L, temp_pi, _inv_sig, _nu, beta_zc, log_eta0, eta1, eta2):
        batch_size = tf.shape(z)[0]
        L = tf.shape(z)[1]
        dist = tfp.distributions.Normal(
            loc = tf.constant(0.0, tf.keras.backend.floatx()), 
            scale = tf.constant(1.0, tf.keras.backend.floatx()), 
            allow_nan_stats=False)
        
        # [batch_size, L, n_categories, n_clusters]
        _inv_sig = tf.expand_dims(_inv_sig, -1)
        _sig = tf.tile(tf.clip_by_value(tf.math.reciprocal_no_nan(_inv_sig), 1e-12, 1e30), (1,1,1,self.n_states))
        log_eta0 = tf.tile(tf.expand_dims(log_eta0, -1), (1,1,1,self.n_states))
        eta1 = tf.tile(tf.expand_dims(eta1, -1), (1,1,1,self.n_states))
        eta2 = tf.tile(tf.expand_dims(eta2, -1), (1,1,1,self.n_states))
        _nu = tf.tile(tf.expand_dims(_nu, -1), (1,1,1,1))
        A = tf.tile(tf.expand_dims(tf.expand_dims(
            tf.one_hot(self.A, self.n_states, dtype=tf.keras.backend.floatx()), 
            0),0), (batch_size,L,1,1))
        B = tf.tile(tf.expand_dims(tf.expand_dims(
            tf.one_hot(self.B, self.n_states, dtype=tf.keras.backend.floatx()), 
            0),0), (batch_size,L,1,1))
        temp_pi = tf.expand_dims(temp_pi, -1)

        # w_tilde [batch_size, L, n_clusters]
        w_tilde = log_eta0 + tf.math.log(
            tf.clip_by_value(
                (dist.cdf(eta1) - dist.cdf(eta2)) * (_nu * A + (1-_nu) * B)  -
                (dist.prob(eta1) - dist.prob(eta2)) * tf.math.sqrt(_sig) * (A - B), 
                0.0, 1e30)
            )
        w_tilde = - 0.5 * self.dim_latent * tf.math.log(tf.constant(2 * np.pi, tf.keras.backend.floatx())) + \
            tf.math.log(temp_pi) + \
            tf.where(_inv_sig==0, 
                    tf.where(B==1, - 0.5 * tf.expand_dims(tf.reduce_sum(beta_zc**2, axis=2), -1), -np.inf), 
                    w_tilde)
        w_tilde = tf.exp(tf.reduce_logsumexp(w_tilde, 2) - log_p_z_L)

        # tf.debugging.assert_greater_equal(
        #     tf.reduce_sum(w_tilde, -1), tf.ones([batch_size, L], dtype=tf.keras.backend.floatx())*0.99, 
        #     message='Wrong w_tilde', summarize=None, name=None
        # )
        
        # var_w_tilde [batch_size, L, n_clusters]
        var_w_tilde = log_eta0 + tf.math.log(
            tf.clip_by_value(
                (dist.cdf(eta1) -  dist.cdf(eta2)) * ((_sig + _nu**2) * (A+B) + (1-2*_nu) * B)  -
                (dist.prob(eta1) - dist.prob(eta2)) * tf.math.sqrt(_sig) * (_nu *(A+B)-B )*2 -
                (eta1*dist.prob(eta1) - eta2*dist.prob(eta2)) * _sig *(A+B), 
                0.0, 1e30)
            )
        var_w_tilde = - 0.5 * self.dim_latent * tf.math.log(tf.constant(2 * np.pi, tf.keras.backend.floatx())) + \
            tf.math.log(temp_pi) + \
            tf.where(_inv_sig==0, 
                    tf.where(B==1, - 0.5 * tf.expand_dims(tf.reduce_sum(beta_zc**2, axis=2), -1), -np.inf), 
                    var_w_tilde) 
        var_w_tilde = tf.exp(tf.reduce_logsumexp(var_w_tilde, 2) - log_p_z_L) - w_tilde**2  


        w_tilde = tf.reduce_mean(w_tilde, 1)
        var_w_tilde = tf.reduce_mean(var_w_tilde, 1)
        return w_tilde, var_w_tilde

    def get_pz(self, z, eps):
        '''Get \(\\log p(Z_i|Y_i,X_i)\).

        Parameters
        ----------
        z : tf.Tensor
            \([B, L, d]\) The latent variables.

        Returns
        ----------
        temp_pi : tf.Tensor
            \([B, L, K]\) \(\\pi\).
        _inv_sig : tf.Tensor
            \([B, L, K]\) \(\\sigma_{Z_ic_i}^{-1}\).
        _nu : tf.Tensor
            \([B, L, K]\) \(\\nu_{Z_ic_i}\).
        beta_zc : tf.Tensor
            \([B, L, d, K]\) \(\\beta_{Z_ic_i}\).
        log_eta0 : tf.Tensor
            \([B, L, K]\) \(\\log\\eta_{Z_ic_i,0}\).
        eta1 : tf.Tensor
            \([B, L, K]\) \(\\eta_{Z_ic_i,1}\).
        eta2 : tf.Tensor
            \([B, L, K]\) \(\\eta_{Z_ic_i,2}\).
        log_p_zc_L : tf.Tensor
            \([B, L, K]\) \(\\log p(Z_i,c_i|Y_i,X_i)\).
        log_p_z_L : tf.Tensor
            \([B, L]\) \(\\log p(Z_i|Y_i,X_i)\).
        log_p_z : tf.Tensor
            \([B, 1]\) The estimated \(\\log p(Z_i|Y_i,X_i)\). 
        '''        
        temp_pi, beta_zc, _inv_sig, _nu, _t = self._get_normal_params(z)
        temp_pi = tf.clip_by_value(temp_pi, eps, 1.0)

        log_eta0 = 0.5 * (tf.math.log(tf.constant(2 * np.pi, tf.keras.backend.floatx())) - \
                    tf.math.log(tf.clip_by_value(_inv_sig, 1e-12, 1e30)) + _t)
        eta1 = (1-_nu) * tf.math.sqrt(tf.clip_by_value(_inv_sig, 1e-12, 1e30))
        eta2 = -_nu * tf.math.sqrt(tf.clip_by_value(_inv_sig, 1e-12, 1e30))

        log_p_z_c_L =  log_eta0 + tf.math.log(tf.clip_by_value(
            self.cdf_layer(eta1) - self.cdf_layer(eta2),
            eps, 1e30))
        
        log_p_zc_L, log_p_z_L, log_p_z = self._get_pz(temp_pi, _inv_sig, beta_zc, log_p_z_c_L)
        return temp_pi, _inv_sig, _nu, beta_zc, log_eta0, eta1, eta2, log_p_zc_L, log_p_z_L, log_p_z

    def get_posterior_c(self, z):
        '''Get \(p(c_i|Y_i,X_i)\).

        Parameters
        ----------
        z : tf.Tensor
            \([B, L, d]\) The latent variables.

        Returns
        ----------
        p_c_x : np.array
            \([B, K]\) \(p(c_i|Y_i,X_i)\).
        '''  
        _,_,_,_,_,_,_, log_p_zc_L, log_p_z_L, _ = self.get_pz(z)
        log_p_c_x = self._get_posterior_c(log_p_zc_L, log_p_z_L)
        p_c_x = tf.exp(log_p_c_x).numpy()
        return p_c_x

    def call(self, z, training=True):
        '''Get posterior estimations.

        Parameters
        ----------
        z : tf.Tensor
            \([B, L, d]\) The latent variables.
        training : boolean
            Whether in training or inference mode.

        When `inference=False`:

        Returns
        ----------
        log_p_z_L : tf.Tensor
            \([B, 1]\) The estimated \(\\log p(Z_i|Y_i,X_i)\).

        When `inference=True`:

        Returns
        ----------
        res : dict
            The dict of posterior estimations - \(p(c_i|Y_i,X_i)\), \(c\), \(E(\\tilde{w}_i|Y_i,X_i)\), \(Var(\\tilde{w}_i|Y_i,X_i)\), \(D_{JS}\).
        '''                 
        eps = 1e-16 if training else 0.
        temp_pi, _inv_sig, _nu, beta_zc, log_eta0, eta1, eta2, log_p_zc_L, log_p_z_L, log_p_z = self.get_pz(z, eps)

        if training:
            return log_p_z
        else:
            log_p_c_x = self._get_posterior_c(log_p_zc_L, log_p_z_L)
            w_tilde, var_w_tilde = self._get_inference(z, log_p_z_L, temp_pi, _inv_sig, _nu, beta_zc, log_eta0, eta1, eta2)
            
            res = {}
            res['p_c_x'] = tf.exp(log_p_c_x).numpy()
            res['w_tilde'] = w_tilde.numpy()
            res['var_w_tilde'] = var_w_tilde.numpy()
            return res



