# -*- coding: utf-8 -*-
from typing import Optional

from scVAEIT.utils import Early_Stopping

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar
import tensorflow_addons as tfa

from time import time

def clear_session():
    '''Clear Tensorflow sessions.
    '''
    tf.keras.backend.clear_session()
    return None


def train(dataset_train, dataset_valid, vae, checkpoint_dir, 
              learning_rate: float, L: int, alpha: float,
              num_epoch: int, num_step_per_epoch: int, save_every_epoch: int,
              es_patience: int, es_tolerance: int, es_relative: bool,
              verbose: bool = True, eval_func=None):
    '''Pretraining.

    Parameters
    ----------
    dataset_train : tf.Dataset
        The Tensorflow Dataset object.
    dataset_valid : tf.Dataset
        The Tensorflow Dataset object.
    vae : VariationalAutoEncoder
        The model.
    learning_rate : float
        The initial learning rate for the Adam optimizer.
    L : int
        The number of MC samples.
    alpha : float, optional
        The value of alpha in [0,1] to encourage covariate adjustment. Not used if there is no covariates.
    num_epoch : int
        The maximum number of epoches.
    num_step_per_epoch : int
        The number of step per epoch, it will be inferred from number of cells and batch size if it is None.            
    es_patience : int
        The maximum number of epoches if there is no improvement.
    es_tolerance : float
        The minimum change of loss to be considered as an improvement.
    es_relative : bool, optional
        Whether monitor the relative change of loss or not.        
    es_warmup : int, optional
        The number of warmup epoches.

    Returns
    ----------
    vae : VariationalAutoEncoder
        The pretrained model.
    '''
    
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, net=vae, step=tf.Variable(1),)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, 
                                         max_to_keep=None if dataset_valid is None else es_patience+2)
    n_modal = len(vae.config.uni_block_names)
    loss_train = tf.metrics.Mean('train_loss', dtype=tf.float32)
    loss_train_list = {}
    loss_train_list['obs'] = [tf.metrics.Mean('train_loss_%d'%i, dtype=tf.float32) for i in range(n_modal)]
    loss_train_list['unobs'] = [tf.metrics.Mean('train_loss_%d'%i, dtype=tf.float32) for i in range(n_modal)]
    loss_train_list['kl'] = [tf.metrics.Mean('train_loss_kl_%d'%i, dtype=tf.float32) for i in range(1)]
    
    if dataset_valid is not None:
        loss_val = tf.metrics.Mean('val_loss', dtype=tf.float32)
        loss_val_list = {}
        loss_val_list['obs'] = [tf.metrics.Mean('val_loss_%d'%i, dtype=tf.float32) for i in range(n_modal)]
        loss_val_list['unobs'] = [tf.metrics.Mean('val_loss_%d'%i, dtype=tf.float32) for i in range(n_modal)]
        loss_val_list['kl'] = [tf.metrics.Mean('val_loss_kl_%d'%i, dtype=tf.float32) for i in range(1)]
        
    early_stopping = Early_Stopping(patience=es_patience, tolerance=es_tolerance, relative=es_relative)

    if not verbose:
        progbar = Progbar(num_epoch)
    
    start_time = time()
    for epoch in range(1,num_epoch+1):

        if verbose:
            progbar = Progbar(num_step_per_epoch)
            print('Pretrain - Start of epoch %d' % (epoch,))
        else:
            if (epoch+1)%2==0 or epoch+1==num_epoch:
                progbar.update(epoch+1)

        # Iterate over the batches of the dataset.
        for step, (x, b, id_data) in enumerate(dataset_train):
            m = tf.gather(vae.masks, id_data)
            m = vae.generate_mask(x, m)
            with tf.GradientTape() as tape:
                losses = vae(x, m, b, pre_train=True, L=L)
                # Compute reconstruction loss
                loss = tf.reduce_sum(losses)
            grads = tape.gradient(loss, vae.trainable_weights,
                        unconnected_gradients=tf.UnconnectedGradients.ZERO)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))
            
            for i, l in enumerate(loss_train_list['obs']):
                l(losses[i])
            for i, l in enumerate(loss_train_list['unobs']):
                l(losses[i+n_modal])
            loss_train_list['kl'][0](losses[-1])
            loss_train(loss)

            if verbose:
                if (step+1)%10==0 or step+1==num_step_per_epoch:
                    progbar.update(step+1, [('Reconstructed Loss', float(loss))])
        
        if dataset_valid is not None:
            for step, (x, b, id_data) in enumerate(dataset_valid):
                m = tf.gather(vae.masks, id_data)
                m = vae.generate_mask(x, m, p=0.)
                losses = vae(x, m, b, pre_train=True, L=L, training=False)

                for i, l in enumerate(loss_val_list['obs']):
                    l(losses[i])
                for i, l in enumerate(loss_val_list['unobs']):
                    l(losses[i+n_modal])
                loss_val_list['kl'][0](losses[-1])
                loss_val(tf.reduce_sum(losses))

        if verbose:
            if dataset_valid is not None:
                print('Epoch {}, Train ELBO: {:5.02f}, Val ELBO: {:5.02f}, Time elapsed: {} minutes'.\
                    format(epoch, float(loss_train.result()), 
                       float(loss_val.result()), round((time() - start_time) / 60, 2)))
                print(', '.join('{:>7s}'.format(i) for i in 
                    np.append(
                        np.tile(vae.config.uni_block_names, 2), ['kl'])))
                print(', '.join('{:>5.02f}'.format(l.result()) for key in loss_train_list for l in loss_train_list[key]))
                print(', '.join('{:>5.02f}'.format(l.result()) for key in loss_val_list for l in loss_val_list[key]))
            else:
                print('Epoch {}, Train ELBO: {:5.02f}, Time elapsed: {} minutes'.\
                    format(epoch, float(loss_train.result()), round((time() - start_time) / 60, 2)))
                print(', '.join('{:>7s}'.format(i) for i in 
                    np.append(
                        np.tile(vae.config.uni_block_names, 2), ['kl'])))
                print(', '.join('{:>5.02f}'.format(l.result()) for key in loss_train_list for l in loss_train_list[key]))
                
        if dataset_valid is not None:
            save_path = manager.save()
            print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))
            if early_stopping(float(loss_val.result())):
                print('Early stopping.')
                break
                
            loss_val.reset_states()
            [l.reset_states() for key in loss_val_list for l in loss_val_list[key]]
        else:
            if int(checkpoint.step) % save_every_epoch == 0:
                print(checkpoint.step)
                save_path = manager.save()
                print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))
                
                if eval_func is not None:
                    eval_func(vae)
        checkpoint.step.assign_add(1)
        
        loss_train.reset_states()
        [l.reset_states() for key in loss_train_list for l in loss_train_list[key]]

    print('Pretrain Done.')
    return vae

