import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import numpy as np

    
""" Helper Functions from RELAX https://github.com/duvenaud/relax """
def safe_log_prob(x, eps=1e-8):
    return K.log(tf.clip_by_value(x, eps, 1.0))
  

def safe_clip(x, eps=1e-8):
    return tf.clip_by_value(x, eps, 1.0)


def gs(x):
    return x.get_shape().as_list()


def softplus(x):
    '''
    Let m = max(0, x), then,
    sofplus(x) = log(1 + e(x)) = log(e(0) + e(x)) = log(e(m)(e(-m) + e(x-m)))
                         = m + log(e(-m) + e(x - m))
    The term inside of the log is guaranteed to be between 1 and 2.
    '''
    m = K.maximum(K.zeros_like(x), x)
    return m + K.log(K.exp(-m) + K.exp(x - m))


def logistic_loglikelihood(z, loc, scale=1):
    return K.log(K.exp(-(z-loc)/scale)/scale*K.square((1+K.exp(-(z-loc)/scale))))


def bernoulli_loglikelihood(b, log_alpha):
    return b * (-softplus(-log_alpha)) + (1 - b) * (-log_alpha - softplus(-log_alpha))


def bernoulli_loglikelihood_derivitive(b, log_alpha):
    assert gs(b) == gs(log_alpha)
    sna = K.sigmoid(-log_alpha)
    return b * sna - (1-b) * (1 - sna)


def v_from_u(u, log_alpha, force_same=True, b=None, v_prime=None):
    u_prime = K.sigmoid(-log_alpha)
    if not force_same:
        v = b*(u_prime+v_prime*(1-u_prime)) + (1-b)*v_prime*u_prime
    else:
        v_1 = (u - u_prime) / safe_clip(1 - u_prime)
        v_1 = tf.clip_by_value(v_1, 0, 1)
        v_1 = tf.stop_gradient(v_1)
        v_1 = v_1 * (1 - u_prime) + u_prime
        v_0 = u / safe_clip(u_prime)
        v_0 = tf.clip_by_value(v_0, 0, 1)
        v_0 = tf.stop_gradient(v_0)
        v_0 = v_0 * u_prime
    
        v = tf.where(u > u_prime, v_1, v_0)
        v = tf.debugging.check_numerics(v, 'v sampling is not numerically stable.')
        if force_same:
            v = v + tf.stop_gradient(-v + u)  # v and u are the same up to numerical errors
    return v


def reparameterize(log_alpha, noise):
    return log_alpha + safe_log_prob(noise) - safe_log_prob(1 - noise)


def concrete_relaxation(z, temp):
    return K.sigmoid(z / temp)



""" Sampler Classes """
class REBAR_Bernoulli_Sampler(Layer):
    '''
    Layer to Sample z, s, z~
    '''
    def __init__(self, tau0=.1, **kwargs):
        super(REBAR_Bernoulli_Sampler, self).__init__(**kwargs)
        
        self.tau0 = tau0

        
    def call(self,  logits):
        batch_size = tf.shape(logits)[0]
        d = tf.shape(logits)[1]
        
        u = tf.random.uniform(shape=(batch_size, d),
                                    minval=np.finfo(
                                        tf.float32.as_numpy_dtype).tiny,
                                    maxval=1.0) 
        v_p = tf.random.uniform(shape=(batch_size, d),
                                    minval=np.finfo(
                                        tf.float32.as_numpy_dtype).tiny,
                                    maxval=1.0)
        
        z = reparameterize(logits, u)
        s = K.cast(tf.stop_gradient(z > 0), tf.float32)
        v = v_from_u(u, logits, False, s, v_p)
        z_tilde = reparameterize(logits, v)
        
        sig_z = concrete_relaxation(z, self.tau0)
        sig_z_tilde = concrete_relaxation(z_tilde, self.tau0)
        
        return [sig_z, s, sig_z_tilde]
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'tau0': self.tau0
        })
        return config    

class Random_Bernoulli_Sampler(Layer):
    '''
    Layer to Sample r
    '''
    def __init__(self, **kwargs):
        super(Random_Bernoulli_Sampler, self).__init__(**kwargs)
        
    def call(self,  logits):
        batch_size = tf.shape(logits)[0]
        d = tf.shape(logits)[1]
        
        u = tf.random.uniform(shape=(batch_size, d),
                                    minval=np.finfo(
                                        tf.float32.as_numpy_dtype).tiny,
                                    maxval=1.0) 

        r = K.cast(tf.stop_gradient(u > 0.5), tf.float32)
        
        return r