import jax
import jax.numpy as jnp

def init_model(model_params, x, key):
    if model_params['type'] == 'relu-mlp':
        from .relu_mlp import ReLU_MLP
        model = ReLU_MLP()
        params = model.init(key, x)
    elif model_params['type'] == 'two_layer_mlp':
        from .two_layer_mlp import Two_Layer_ReLU_MLP

        model = Two_Layer_ReLU_MLP() 
        params = model.init(key, x)
    elif model_params['type'] == 'varpro-mlp':
        from .relu_mlp import VarPro_MLP
        model = VarPro_MLP()
        params = model.init(key, x)
   
    else:
      raise ValueError("This model is currently not implemented.")
    
    def loss(params, data_batch, data_labels):
        preds = model.apply(params, data_batch)
        return ((preds-data_labels)**2).mean()
    
    return params, model, loss