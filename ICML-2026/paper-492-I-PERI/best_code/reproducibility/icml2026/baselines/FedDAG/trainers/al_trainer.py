import logging
import numpy as np
from helpers.analyze_utils import compute_acyclicity, convert_logits_to_sigmoid

class ALTrainer(object):
    """
    Augmented Lagrangian method with first-order gradient-based optimization
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, init_rho, rho_thres, h_thres, rho_multiply,
                 init_iter, h_tol, temperature, it_fl, sess, ops_set, logger:logging.Logger=None):
        self.init_rho = init_rho
        self.rho_thres = rho_thres
        self.h_thres = h_thres
        self.rho_multiply = rho_multiply
        self.init_iter = init_iter
        self.h_tol = h_tol
        self.temperature = temperature
        self.sess = sess
        self.ops_set = ops_set
        self.it_fl = it_fl
        self.logger = logger

    def train(self, Model_set, X, max_iter, iter_step, init_alpha):
        self.num_client = len(X)
        _, self.d = X[0].shape
        rho, alpha, h, h_new = self.init_rho, init_alpha, np.inf, np.inf

        for i in range(1, max_iter + 1):
            # print the mid data
            print('<----------------------------------->')
            if self.logger:
                self.logger.info('This is the {} iteration. h_loss is {}, alpha is {}, rho is {}'.format(i, h_new, alpha, rho))
            while rho < self.rho_thres:
                h_new, W_logits_new \
                    = self.train_step(Model_set, iter_step, X, rho, alpha, self.temperature)
                if h_new > self.h_thres * h:
                    rho *= self.rho_multiply
                else:
                    break

            # Use two stopping criterions
            h_logits = compute_acyclicity(convert_logits_to_sigmoid(W_logits_new / self.temperature))
            if h_new <= self.h_tol and h_logits <= self.h_tol and i > self.init_iter:
                break

            # Update h and alpha
            h = h_new
            alpha += rho * h_new

        return W_logits_new

    def train_step(self, Model_set, iter_step, X, rho, alpha, temperature):

        train_ops, losses, mse_losses, hs, Ws = [], [], [], [], []

        for model in Model_set:
            train_ops.append(model.train_op)
            losses.append(model.loss)
            mse_losses.append(model.mse_loss)
            hs.append(model.h)
            Ws.append(model.W)

        feed_dict = {}
        for client_index in range(self.num_client):
            model = Model_set[client_index]
            feed_dict[model.X] = X[client_index]
            feed_dict[model.rho] = rho
            feed_dict[model.alpha] = alpha
            feed_dict[model.tau] = temperature

        # curr_loss, curr_mse and curr_h are single-sample estimation
        for it_s in range(iter_step):
            run_result = self.sess.run(
                {'train_ops': train_ops, 'losses': losses, 'mse_losses': mse_losses, 'hs': hs, 'Ws': Ws},
                feed_dict=feed_dict )

            if ((it_s +1) % self.it_fl == 0) | ((it_s+1) == iter_step):
                self.sess.run(self.ops_set)
                if (it_s+1) == iter_step:
                    final_h, final_W = run_result['hs'][0], run_result['Ws'][0]

        return final_h, final_W

class NoTears_ALTrainer(object):
    """
    Augmented Lagrangian method with first-order gradient-based optimization
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, init_rho, rho_thres, h_thres, rho_multiply,
                 init_iter, h_tol, lr, it_fl, sess, ops_set, logger:logging.Logger=None):
        self.init_rho = init_rho
        self.rho_thres = rho_thres
        self.h_thres = h_thres
        self.rho_multiply = rho_multiply
        self.init_iter = init_iter
        self.lr = lr
        self.h_tol = h_tol
        self.sess = sess
        self.ops_set = ops_set
        self.it_fl = it_fl
        self.logger = logger

    def train(self, Model_set, X, max_iter, iter_step, init_alpha):
        self.num_client = len(X)
        _, self.d = X[0].shape
        rho, alpha, h, h_new = self.init_rho, init_alpha, np.inf, np.inf

        for i in range(1, max_iter + 1):
            # print the mid data
            print('<----------------------------------->')
            if self.logger:
                self.logger.info('This is the {} iteration. h_loss is {}, alpha is {}, rho is {}'.format(i, h_new, alpha, rho))
            while rho < self.rho_thres:
                h_new, W_logits_new \
                    = self.train_step(Model_set, iter_step, X, rho, alpha)
                if h_new > self.h_thres * h:
                    rho *= self.rho_multiply
                else:
                    break

            # Update h and alpha
            h = h_new
            alpha += rho * h_new

            if h <= self.h_tol and i > self.init_iter:
                self.logger.info('Early stopping at {}-th iteration'.format(i))
                break

        return W_logits_new

    def train_step(self, Model_set, iter_step, X, rho, alpha):

        train_ops, losses, mse_losses, hs, Ws = [], [], [], [], []

        for model in Model_set:
            train_ops.append(model.train_op)
            losses.append(model.loss)
            mse_losses.append(model.mse_loss)
            hs.append(model.h)
            Ws.append(model.W_prime)

        feed_dict = {}
        for client_index in range(self.num_client):
            model = Model_set[client_index]
            feed_dict[model.X] = X[client_index]
            feed_dict[model.rho] = rho
            feed_dict[model.alpha] = alpha
            feed_dict[model.lr] = self.lr

        # curr_loss, curr_mse and curr_h are single-sample estimation
        for it_s in range(iter_step):
            run_result = self.sess.run(
                {'train_ops': train_ops, 'losses': losses, 'mse_losses': mse_losses, 'hs': hs, 'Ws': Ws},
                feed_dict=feed_dict )

            if ((it_s +1) % self.it_fl == 0) | ((it_s+1) == iter_step):
                self.sess.run(self.ops_set)
                if (it_s+1) == iter_step:
                    final_h, final_W = run_result['hs'][0], run_result['Ws'][0]

        return final_h, final_W

