from logging import Logger
import logging
import sys
import numpy as np
from .nonlinear.masked_nn import MaskedNN
from .linear.notears import NoTears
from trainers import ALTrainer, NoTears_ALTrainer
from helpers.train_utils import callback_after_training
from helpers.base import BaseLearner
import tensorflow as tf

class AS_FedDAG(BaseLearner):

    def __init__(self, n, d, num_client, use_gpu=True, seed=1, graph_thres=0.5,
                 num_hidden_layers=4, hidden_size=16,
                 l1_graph_penalty=2e-3, use_float64=False, lr=3e-2,
                 max_iter=25, iter_step=1000, init_iter=3, h_tol=1e-10,
                 init_rho=1e-3, rho_thres=1e14, h_thres=0.25, rho_multiply=5,
                 temperature=0.2, it_fl=200, init_alpha=0.0, num_shared_client=None, logger:Logger=None):

        super().__init__()

        self.n = n
        self.d = d
        self.num_client = num_client
        self.use_gpu = use_gpu
        self.seed = seed
        self.graph_thres = graph_thres
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.l1_graph_penalty = l1_graph_penalty
        self.use_float64 = use_float64
        self.lr = lr
        self.max_iter = max_iter
        self.iter_step = iter_step
        self.init_iter = init_iter
        self.h_tol = h_tol
        self.init_rho = init_rho
        self.rho_thres = rho_thres
        self.h_thres = h_thres
        self.rho_multiply = rho_multiply
        self.temperature = temperature
        self.it_fl = it_fl
        self.init_alpha=init_alpha
        self.num_shared_client = num_shared_client
        self.logger = logger

    def learn(self, dataset):

        pns_mask = np.ones([self.d, self.d])
        causal_matrix = self._iidfedcd(dataset, pns_mask)
        self.causal_matrix = causal_matrix

    def _iidfedcd(self, X, pns_mask):

        Model_set = []
        for i in range(self.num_client):

            model_name = MaskedNN
            with tf.compat.v1.variable_scope('model_{0}'.format(i)):
                model = model_name(self.n, self.d, pns_mask, self.num_hidden_layers,
                                   self.hidden_size, self.l1_graph_penalty, self.lr,
                                   self.seed, self.use_float64, self.use_gpu)
                model.name_scope = 'model_{0}'.format(i)
            Model_set.append(model)

        ops_set = []
        var_num = len(tf.compat.v1.trainable_variables(Model_set[0].name_scope))

        if self.num_shared_client is not None:
            indices = tf.range(0, self.num_client, dtype=tf.int64)
            chosen_indices = tf.random.shuffle(indices, seed=self.seed)[:self.num_shared_client]

        for var_index in range(var_num):
            variables = []
            for model in Model_set:
                variables.append(tf.compat.v1.trainable_variables(model.name_scope)[var_index])
            if self.num_shared_client is not None:
                chosen_variables = tf.gather(variables, chosen_indices, axis=0)
                ave = tf.compat.v1.reduce_mean(chosen_variables, axis=0)
            else:
                ave = tf.compat.v1.reduce_mean(variables, axis=0)
            for variable in variables:
                ops_set.append(tf.compat.v1.assign(variable, ave))

        if self.use_gpu:
            # Use GPU
            sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
                gpu_options=tf.compat.v1.GPUOptions(
                    allow_growth=True,
                )
            ))
        else:
            sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

        trainer = ALTrainer(self.init_rho, self.rho_thres, self.h_thres,
                            self.rho_multiply, self.init_iter, self.h_tol,
                            self.temperature, self.it_fl, sess, ops_set, logger=self.logger)

        W_logits = trainer.train(Model_set, X, self.max_iter, self.iter_step, self.init_alpha)
        W_est = callback_after_training(W_logits, self.temperature, self.graph_thres)

        return W_est

    def post_train(self, trainsets, validsets, epoches:int):
        assert(False)

class GS_FedDAG(BaseLearner):

    def __init__(self, d, num_client,use_gpu=True, seed=1, graph_thres=0.5,
                 num_hidden_layers=4, hidden_size=16,
                 l1_graph_penalty=2e-3, use_float64=False, lr=3e-2,
                 max_iter=25, iter_step=1000, init_iter=3, h_tol=1e-10,
                 init_rho=1e-3, rho_thres=1e14, h_thres=0.25, rho_multiply=5,
                 temperature=0.2, it_fl=200, init_alpha=0.0, num_shared_client=None, logger=None):

        super().__init__()

        self.d = d
        self.num_client = num_client
        self.use_gpu = use_gpu
        self.seed = seed
        self.graph_thres = graph_thres
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.l1_graph_penalty = l1_graph_penalty
        self.use_float64 = use_float64
        self.np_float_type = np.float64 if use_float64 else np.float32
        self.lr = lr
        self.max_iter = max_iter
        self.iter_step = iter_step
        self.init_iter = init_iter
        self.h_tol = h_tol
        self.init_rho = init_rho
        self.rho_thres = rho_thres
        self.h_thres = h_thres
        self.rho_multiply = rho_multiply
        self.temperature = temperature
        self.it_fl = it_fl
        self.init_alpha = init_alpha
        self.num_shared_client = num_shared_client
        if logger is None:
            self.logger = logger
        else:
            self.logger = logging

    def learn(self, dataset):
        pns_mask = np.ones([self.d, self.d])
        causal_matrix = self._noniidfedcd(dataset, pns_mask)
        self.causal_matrix = causal_matrix

    def _noniidfedcd(self, X, pns_mask):

        Model_set = []
        self.model_types = []
        for i in range(self.num_client):
            model_name = MaskedNN
            self.model_types.append(model_name)
            n = X[i].shape[0]
            with tf.compat.v1.variable_scope('model_{0}'.format(i)):
                model = model_name(n, self.d, pns_mask, self.num_hidden_layers,
                                   self.hidden_size, self.l1_graph_penalty, self.lr,
                                   self.seed, self.use_float64, self.use_gpu)
                model.name_scope = 'model_{0}'.format(i)
            Model_set.append(model)

        ops_set, variables = [], []
        for model in Model_set:
            variables.append(model.W)
        if self.num_shared_client is not None:
            # https://github.com/tensorflow/tensorflow/issues/8496#issuecomment-589939975
            indices = tf.range(0, self.num_client, dtype=tf.int64)
            chosen_indices = tf.random.shuffle(indices, seed=self.seed)[:self.num_shared_client]
            chosen_variables = tf.gather(variables, chosen_indices, axis=0)
            ave = tf.compat.v1.reduce_mean(chosen_variables, axis=0)
        else:
            ave = tf.compat.v1.reduce_mean(variables, axis=0)

        for variable in variables:
            ops_set.append(tf.compat.v1.assign(variable, ave))

        if self.use_gpu:
            # Use GPU
            sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
                gpu_options=tf.compat.v1.GPUOptions(
                    allow_growth=True,
                )
            ))
        else:
            sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

        trainer = ALTrainer(self.init_rho, self.rho_thres, self.h_thres,
                            self.rho_multiply, self.init_iter, self.h_tol,
                            self.temperature, self.it_fl, sess, ops_set, self.logger)

        W_logits = trainer.train(Model_set, X, self.max_iter, self.iter_step, self.init_alpha)

        W_est = callback_after_training(W_logits, self.temperature, self.graph_thres)
        self.model_set = Model_set
        self.sess = sess
        return W_est

    def get_mse(self, dataset, repeat=10):
        mses = []
        for _ in range(repeat):
            for model, x in zip(self.model_set, dataset):
                fd = {model.X:x, model.tau:self.temperature}
                mses.append(self.sess.run(model.mse_loss, feed_dict=fd))
        return np.average(mses)
        
    def post_train(self, trainsets, validsets, epoches:int, valid_epoches:int, variable_scope:str):
        self.logger.info("start post train, epochs:{}, valid_epoches:{}".format(epoches, valid_epoches))
        if variable_scope is None:
            variable_scope = 'post_train'
        pns_mask = np.ones([self.d, self.d])
        models = []
        train_ops = []
        num_clients = len(self.model_types)
        assert(len(trainsets) == num_clients and len(validsets) == num_clients)
        with tf.compat.v1.variable_scope(variable_scope):
            for model_type, trainset, i in zip(self.model_types, trainsets, range(num_clients)):
                with tf.compat.v1.variable_scope('model_{0}'.format(i)):
                    n = trainset.shape[0]
                    model = model_type(n, self.d, pns_mask, self.num_hidden_layers,
                                    self.hidden_size, self.l1_graph_penalty, self.lr,
                                    self.seed, self.use_float64, self.use_gpu, pretrained_W = self.causal_matrix)
                    models.append(model)
                train_ops.append(tf.compat.v1.train.AdamOptimizer(learning_rate=model.learning_rate).minimize(model.mse_loss))

        # https://stackoverflow.com/a/36536063/11879605
        all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=variable_scope)
        init_op = tf.compat.v1.variables_initializer(all_variables)
        
        self.sess.run(init_op)

        best_valid_mses = np.array([sys.float_info.max for _ in range(num_clients)], dtype=self.np_float_type)
        best_epoch = 0
        for epoch in range(epoches):
            train_feed_dict = {}
            valid_feed_dict = {}
            for model, trainset, validset in zip(models, trainsets, validsets):
                train_feed_dict[model.X] = trainset
                valid_feed_dict[model.X] = validset
            self.sess.run(train_ops, feed_dict=train_feed_dict)
            if epoch % valid_epoches == 0:
                valid_mses = self.sess.run([model.mse_loss for model in models], feed_dict=valid_feed_dict)
                train_mses = self.sess.run([model.mse_loss for model in models], feed_dict=train_feed_dict)
                self.logger.info("epoch:{} train_mse:{} valid_mse:{}".format(epoch,  np.average(train_mses), np.average(valid_mses)))

                maybe_best_valid_mses = (valid_mses >= best_valid_mses)*best_valid_mses + (valid_mses <= best_valid_mses)*valid_mses
                if np.average(maybe_best_valid_mses) < np.average(best_valid_mses):
                    best_valid_mses = maybe_best_valid_mses
                    best_epoch = epoch
                    self.logger.info("best mse reached: epoch:{} mse:{}".format(best_epoch, np.average(best_valid_mses)))

        return best_valid_mses

class AS_FedDAG_linear(BaseLearner):

    def __init__(self, n, d, num_client, use_gpu=True, seed=1, graph_thres=0.3,
                 l1_graph_penalty=0.1, use_float64=False, lr=1e-3,
                 max_iter=20, iter_step=1500, init_iter=3, h_tol=1e-8,
                 init_rho=1.0, rho_thres=1e16, h_thres=0.25, rho_multiply=10.0,
                 it_fl=300, num_shared_client=None, logger:Logger=None):

        super().__init__()

        self.n = n
        self.d = d
        self.num_client = num_client
        self.use_gpu = use_gpu
        self.seed = seed
        self.graph_thres = graph_thres
        self.l1_graph_penalty = l1_graph_penalty
        self.use_float64 = use_float64
        self.lr = lr
        self.max_iter = max_iter
        self.iter_step = iter_step
        self.init_iter = init_iter
        self.h_tol = h_tol
        self.init_rho = init_rho
        self.rho_thres = rho_thres
        self.h_thres = h_thres
        self.rho_multiply = rho_multiply
        self.it_fl = it_fl
        self.num_shared_client = num_shared_client
        self.logger = logger

    def learn(self, dataset):

        pns_mask = np.ones([self.d, self.d])
        causal_matrix = self._iidfedcd(dataset, pns_mask)
        self.causal_matrix = (abs(causal_matrix) >= self.graph_thres).astype(np.float32)

    def _iidfedcd(self, X, pns_mask):

        Model_set = []
        for i in range(self.num_client):
            model_name = NoTears
            with tf.compat.v1.variable_scope('model_{0}'.format(i)):
                model = model_name(n=self.n, d=self.d, seed=self.seed, l1_lambda=self.l1_graph_penalty, use_float64=self.use_float64)
                model.name_scope = 'model_{0}'.format(i)
            Model_set.append(model)

        ops_set = []
        var_num = len(tf.compat.v1.trainable_variables(Model_set[0].name_scope))

        if self.num_shared_client is not None:
            indices = tf.range(0, self.num_client, dtype=tf.int64)
            chosen_indices = tf.random.shuffle(indices, seed=self.seed)[:self.num_shared_client]

        for var_index in range(var_num):
            variables = []
            for model in Model_set:
                variables.append(tf.compat.v1.trainable_variables(model.name_scope)[var_index])
            if self.num_shared_client is not None:
                chosen_variables = tf.gather(variables, chosen_indices, axis=0)
                ave = tf.compat.v1.reduce_mean(chosen_variables, axis=0)
            else:
                ave = tf.compat.v1.reduce_mean(variables, axis=0)
            for variable in variables:
                ops_set.append(tf.compat.v1.assign(variable, ave))

        if self.use_gpu:
            # Use GPU
            sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
                gpu_options=tf.compat.v1.GPUOptions(
                    allow_growth=True,
                )
            ))
        else:
            sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

        trainer = NoTears_ALTrainer(self.init_rho, self.rho_thres, self.h_thres,
                            self.rho_multiply, self.init_iter, self.h_tol,
                            self.lr, self.it_fl, sess, ops_set, logger=self.logger)

        W_logits = trainer.train(Model_set, X, self.max_iter, self.iter_step, 0.0)
        return W_logits