import os
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
from .objectives import objective_value_and_grad_fn
from .estimators import fit_reinforce, fit_analytic

class PACER:
    def __init__(self, n_vars,
                 n_layers=2,
                 hdim=4,
                 density='gaussian',
                 optimize_bernoulli=True,
                 fit_analytic=False,
                 n_steps=5000,
                 lr=1e-2,
                 batch_size=64,
                 n_mc_samples=200,
                 lambd=1,
                 mask_TF_path=None,
                 mask_TF_colname='Ensembl',
                 null_bernoulli_prob=0.05,
                 gene_names=None,
                 logits_nonTF_offset=-5,
                 seed=0):
        self.n_vars = n_vars
        self.n_layers = n_layers
        self.hdim = hdim
        self.optimize_bernoulli = optimize_bernoulli
        self.density = density
        self.fit_analytic = fit_analytic
        self.n_steps = n_steps
        self.lr = lr
        self.batch_size = batch_size
        self.n_mc_samples = n_mc_samples
        self.lambd = lambd
        self.out = None
        self.seed = seed
        self.mask_TF_path = mask_TF_path
        self.mask_TF_colname = mask_TF_colname
        self.null_bernoulli_prob = null_bernoulli_prob
        self.gene_names = gene_names
        self.logits_nonTF_offset = logits_nonTF_offset
        self.params = self.prepare_params()
        if self.mask_TF_path is not None:
            self.mask_nonTFs()

    def prepare_params(self):
        n_vars = self.n_vars
        n_layers = self.n_layers
        hdim = self.hdim
        density = self.density
        assert density in ['gaussian', 'zinb'], "Choose a valid density function for your data: gaussian / zinb"
        
        logits = jnp.zeros((n_vars), dtype=jax.numpy.float32)
        odim = 3 if density == 'zinb' else 2
        glorot_normal = jax.nn.initializers.glorot_normal()
        
        if self.fit_analytic:
            params = {
                'weight_premask': glorot_normal(jax.random.PRNGKey(self.seed), (n_vars, n_vars)),
                'bias_premask': jnp.zeros((n_vars)),
                'stds_2': jnp.zeros((n_vars), dtype=jnp.float32),
                'bernoulli_logits': jnp.zeros((n_vars, n_vars), dtype=jax.numpy.float32),
                'logits': logits,
                'layer_weights': [],
                'layer_biases': [],
            } 
        else:
            if n_layers == 1:
                params = {
                    'weight_premask': glorot_normal(jax.random.PRNGKey(self.seed), (n_vars, n_vars, hdim)),
                    'bias_premask': jnp.zeros((n_vars, hdim)),
                    'layer_weights': [glorot_normal(jax.random.PRNGKey(self.seed+1), (n_vars, hdim, odim))],
                    'layer_biases': [jnp.zeros((n_vars, odim))],
                    'bernoulli_logits': jnp.zeros((n_vars, n_vars), dtype=jax.numpy.float32),
                    'logits': logits,
                }
            elif n_layers > 1:
                params = {
                    'weight_premask': glorot_normal(jax.random.PRNGKey(self.seed), (n_vars, n_vars, hdim)),
                    'bias_premask': jnp.zeros((n_vars, hdim)),
                    'layer_weights': [glorot_normal(jax.random.PRNGKey(self.seed+i+1), (n_vars, hdim, hdim)) for i in range(n_layers)] +
                                    [glorot_normal(jax.random.PRNGKey(self.seed+n_layers+2), (n_vars, hdim, odim))],
                    'layer_biases': [jnp.zeros((n_vars, hdim)) for _ in range(n_layers)] + [jnp.zeros((n_vars, odim))],
                    'bernoulli_logits': jnp.zeros((n_vars, n_vars), dtype=jax.numpy.float32),
                    'logits': logits,
                }
            elif n_layers == 0:
                params = {
                    'weight_premask': glorot_normal(jax.random.PRNGKey(self.seed), (n_vars, n_vars, odim)),
                    'bias_premask': jnp.zeros((n_vars, odim)),
                    'layer_weights': [],
                    'layer_biases': [],
                    'bernoulli_logits': jnp.zeros((n_vars, n_vars), dtype=jax.numpy.float32),
                    'logits': logits,
                }
        return params
    
    def mask_nonTFs(self):
        assert self.gene_names is not None, "Please provide gene names to use the TF masking functionality."
        n_genes = self.n_vars
        if self.mask_TF_path is not None and os.path.exists(self.mask_TF_path):
            TF_df = pd.read_csv(self.mask_TF_path, sep="\t")
            tf_list = TF_df[self.mask_TF_colname]
            print('Masking non-TF genes. Number of TFs:', len(tf_list))
            print('Intersection with data genes:', len(set(tf_list).intersection(set(self.gene_names))))
            print(tf_list, self.gene_names)

            mask_genes = set()
            for tf in tf_list:
                if tf in self.gene_names:
                    col = list(self.gene_names).index(tf)
                    mask_genes.add(col)
            mask_genes = set(range(n_genes)) - mask_genes
            mask_genes = np.array(list(mask_genes))
            
            # Set null logits for the masked genes
            p = self.null_bernoulli_prob
            null = np.log(p) - np.log(1 - p)
            self.params['bernoulli_logits'] = self.params['bernoulli_logits'].at[:, mask_genes].set(null)
            self.params['logits'] = self.params['logits'].at[mask_genes].set(self.params['logits'][mask_genes] + self.logits_nonTF_offset)

    @staticmethod
    def sample_batch_fn(objective_args, key, batch_size=64):
        x, masks, regime_idxs = objective_args
        n_samples = x.shape[0]
        
        # Sample data
        if batch_size >= n_samples:
            return x, masks, regime_idxs
            
        idxs = jax.random.choice(key, n_samples, shape=(batch_size,), replace=False)
        return x[idxs], masks[idxs], regime_idxs[idxs]

    def fit(self, x_train, masks_train, regimes_train,
            x_val=None,
            masks_val=None,
            regimes_val=None,
            dict_metric_fns=None,
            optimize_params=None,
            ):
        if optimize_params is None:
            optimize_params = ['logits', 'weight_premask', 'bias_premask', 'layer_weights', 'layer_biases']
        if self.optimize_bernoulli:
            optimize_params += ['bernoulli_logits']
        
        # Encapsulate data
        objective_args_train = (x_train, masks_train, regimes_train)
        objective_args_val = (x_val, masks_val, regimes_val) if x_val is not None else None

        # Fit model
        key = jax.random.key(self.seed)
        if self.fit_analytic:
            out = fit_analytic(key,
                    objective_args_train,
                    self.params,
                    objective_args_val,
                    n_steps=self.n_steps,
                    learning_rate=self.lr,
                    batch_size=self.batch_size,
                    sample_batch_fn=self.sample_batch_fn,
                    optimize_params=['logits', 'weight_premask', 'bias_premask', 'bernoulli_logits', 'stds_2'],
                    dict_metric_fns=dict_metric_fns,
                    density=self.density,
                    lambd=self.lambd
                    )
        else:
            out = fit_reinforce(key,
                                objective_value_and_grad_fn(self.density),
                                objective_args_train,
                                self.params,
                                objective_args_val,
                                n_steps=self.n_steps,
                                learning_rate=self.lr,
                                n_mc_samples=self.n_mc_samples,
                                batch_size=self.batch_size,
                                sample_batch_fn=self.sample_batch_fn,
                                optimize_params=optimize_params,
                                dict_metric_fns=dict_metric_fns,
                                density=self.density,
                                lambd=self.lambd
                                )
        self.out = out
        return out
    
    def predict_proba(self):
        assert self.out is not None, "You must fit the model before predicting the graph."
        out = self.out
        bernoulli_probs = jax.nn.sigmoid(out['best_params']['bernoulli_logits'])
        logits = jnp.exp(out['best_params']['logits'])
        n = len(logits)
        prob_edge_direction = logits[None, :] / (logits[:, None] + logits[None, :]) # each column corresponds to a different parent
        mask = 1.0 - jnp.eye(n, dtype=prob_edge_direction.dtype)
        prob_edge_direction = prob_edge_direction * mask
        edge_probs = np.array(prob_edge_direction * bernoulli_probs).T
        return edge_probs
    
    def predict(self, threshold=0.5):
        edge_probs = self.predict_proba()
        edges = (edge_probs >= threshold).astype(int)
        return edges
    
