from abc import ABC, abstractmethod
import tensorflow as tf

from helpers.tf_utils import gumbel_sigmoid

class MaskedModel(ABC):

    def __init__(self, n, d, pns_mask, num_hidden_layers, hidden_size,
                 l1_graph_penalty, learning_rate, seed, use_float64, use_gpu, pretrained_W = None):

        self.n = n
        self.d = d
        self.pns_mask = pns_mask
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.l1_graph_penalty = l1_graph_penalty
        self.learning_rate = learning_rate
        self.seed = seed
        self.tf_float_type = tf.dtypes.float64 if use_float64 else tf.dtypes.float32
        self.use_gpu = use_gpu
        self.pretrained_W = pretrained_W

        # Initializer (for reproducibility)
        self.initializer = tf.keras.initializers.glorot_uniform(seed=self.seed)
        self._build()
        # self._init_session()
        self._init_saver()

    def _init_saver(self):
        self.saver = tf.compat.v1.train.Saver()

    def _preprocess_graph(self, W):
        W_prob = gumbel_sigmoid(W, temperature=self.tau, seed=self.seed, tf_float_type=self.tf_float_type)
        W_prob = tf.linalg.set_diag(W_prob, tf.zeros(W_prob.shape[0], dtype=self.tf_float_type))
        return W_prob

    def _build(self):
        # tf.compat.v1.reset_default_graph()

        mask = tf.convert_to_tensor(self.pns_mask, dtype=self.tf_float_type)
        self.rho = tf.compat.v1.placeholder(self.tf_float_type, name="rho")
        self.alpha = tf.compat.v1.placeholder(self.tf_float_type, name="alpha")
        self.tau = tf.compat.v1.placeholder(self.tf_float_type, shape=[], name="tau")  # Temperature
        self.X = tf.compat.v1.placeholder(self.tf_float_type, shape=[None, self.d], name="X")
        if self.pretrained_W is None:
            self.W = tf.compat.v1.Variable(tf.random.uniform([self.d, self.d], minval=-1e-10, maxval=1e-10,
                                                            dtype=self.tf_float_type, seed=self.seed))
            self.Weight_matrix = tf.compat.v1.Variable(tf.random.uniform([self.d, self.d], minval=-5, maxval=5,
                                                         dtype=self.tf_float_type, seed=self.seed))

            # To be implemented by different models
            self.W_prime = self._preprocess_graph(self.W)
            self.W_prime *= mask  # Preliminary neighborhood selection
        else:
            assert(self.pretrained_W.shape == (self.d, self.d))
            self.W_prime = self.W = tf.constant(value=self.pretrained_W, dtype=self.tf_float_type)
        
        
        self.mse_loss = self._get_mse_loss(self.X, self.W_prime)

        self.h = tf.linalg.trace(tf.linalg.expm(self.W_prime * self.W_prime)) - self.d
        self.loss = 0.5 / self.n * self.mse_loss \
                    + self.l1_graph_penalty * tf.norm(self.W_prime, ord=1) \
                    + self.alpha * self.h + 0.5 * self.rho * self.h * self.h

        self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    @abstractmethod
    def _forward(self, x):
        pass

    @abstractmethod
    def _get_mse_loss(self, X, W_prime):
        pass


