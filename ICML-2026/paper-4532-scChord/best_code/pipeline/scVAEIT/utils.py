import numpy as np
import tensorflow as tf

class MaskGenerator(object):
    def __init__(self, p=0.95):
        self.p = p

    def __call__(self, inputs):
        # (batch_size, num_features)
        mask = np.random.choice(2, size=inputs.shape,
                                p=[1 - self.p, self.p]).astype(tf.keras.backend.floatx())
        return mask


class MixtureMaskGenerator(object):

    def __init__(self, generators, weights):
        self.generators = generators
        self.weights = np.array(weights/np.sum(weights)).astype(tf.keras.backend.floatx())

    def __call__(self, inputs, p=None):
        c_ids = np.random.choice(len(self.weights), inputs.shape[0], True, self.weights)
        mask = np.zeros_like(inputs)

        for i, gen in enumerate(self.generators):
            ids = np.where(c_ids == i)[0]
            if len(ids) == 0:
                continue
            samples = gen(tf.gather(inputs, ids, axis=0), p)
            mask[ids] = samples
        return mask    
    

class FixedGenerator(object):
    def __init__(self, masks, p=None):
        self.masks = masks
        self.n = self.masks.shape[0]
        if p is None:
            self.p = np.ones(self.n)/self.n
        else:
            self.p = p

    def __call__(self, inputs, mask=None, p=None):
        i_masks = np.random.choice(self.n, size=inputs.shape[0],
                                p=self.p).astype(int)
        
        return self.masks[i_masks,:].copy()
    
    
class ModalMaskGenerator(object):
    def __init__(self, dim_arr, p_feat=0.05, p_modal=None):
        self.p_feat = p_feat
        self.dim_arr = dim_arr
        self.segment_ids = np.repeat(np.arange(len(dim_arr)), dim_arr)
        if p_modal is None:
            p_modal = np.array(dim_arr, dtype=float)
            p_modal /= np.sum(p_modal)
        self.p_modal = p_modal

    def __call__(self, inputs, missing_mask=None, p=None):
        if p is None:
            p = self.p_feat
        # (batch_size, dim_rna + dim_adt)
        mask = np.random.choice(2, size=inputs.shape,
                                p=[1 - p, p]).astype(tf.keras.backend.floatx())
        
        # No random missing
        mask_modal = np.random.choice(2, size=(inputs.shape[0], ))
        mask[mask_modal==0, :] = 0.
        #mask[:, self.segment_ids==2] = 0.
        
        # Modality missing
        mask_modal = np.random.choice(len(self.dim_arr), size=(inputs.shape[0], ), p=self.p_modal)
        if missing_mask is None:
            for i in np.arange(len(self.dim_arr)):                
                mask[np.ix_(mask_modal==i, self.segment_ids==i)] = 1.
        else:
            has_modal = tf.transpose(
                tf.math.segment_sum(tf.transpose(missing_mask+1), self.segment_ids)).numpy()
            for i in np.arange(len(self.dim_arr)):                
                mask[np.ix_(
                    (mask_modal==i)&
                     np.any(has_modal[:,np.arange(len(self.dim_arr))!=i]>0., axis=-1), 
                     self.segment_ids==i)] = 1.
            mask = np.where(missing_mask==-1., -1., mask)
        
        return mask
    
    
class Early_Stopping():
    '''
    The early-stopping monitor.
    '''
    def __init__(self, warmup=0, patience=10, tolerance=1e-3, 
            relative=False, is_minimize=True):
        self.warmup = warmup
        self.patience = patience
        self.tolerance = tolerance
        self.is_minimize = is_minimize
        self.relative = relative

        self.step = 0
        self.best_step = 0
        self.best_metric = np.inf

        if not self.is_minimize:
            self.factor = -1.0
        else:
            self.factor = 1.0

    def __call__(self, metric):
        self.step += 1
        
        if self.step < self.warmup:
            return False
        elif (self.best_metric==np.inf) or \
                (self.relative and (self.best_metric-metric)/self.best_metric > self.tolerance) or \
                ((not self.relative) and self.factor*metric<self.factor*self.best_metric-self.tolerance):
            self.best_metric = metric
            self.best_step = self.step
            return False
        elif self.step - self.best_step>self.patience:
            print('Best Epoch: %d. Best Metric: %f.'%(self.best_step, self.best_metric))
            return True
        else:
            return False    
