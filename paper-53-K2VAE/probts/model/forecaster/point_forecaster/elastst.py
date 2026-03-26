import torch
import torch.nn as nn
from typing import Union
from probts.model.forecaster import Forecaster
from probts.model.nn.arch.ElasTSTModule.ElasTST_backbone import ElasTST_backbone
from probts.utils import convert_to_list, weighted_average
from probts.data.data_utils.data_scaler import InstanceNorm

class ElasTST(Forecaster):
    def __init__(
        self,
        l_patch_size: Union[str, int, list] = '8_16_32',
        k_patch_size: int = 1,
        stride: int = None,
        rotate: bool = True, 
        addv: bool = False,
        bin_att: bool = False,
        rope_theta_init: str = 'exp',
        min_period: float = 1, 
        max_period: float = 1000,
        learn_tem_emb: bool = False,
        learnable_rope: bool = True, 
        abs_tem_emb: bool = False,
        structured_mask: bool = True,
        max_seq_len: int = 1024,
        theta_base: float = 10000,
        t_layers: int = 1, 
        v_layers: int = 0,
        patch_share_backbone: bool = True,
        n_heads: int = 16, 
        d_k: int = 8, 
        d_v: int = 8,
        d_inner: int = 256, 
        dropout: float = 0.,
        in_channels: int = 1,
        f_hidden_size: int = 40,
        use_norm: bool = True,
        **kwargs
    ):
        """
        ElasTST model.

        Parameters
        ----------
        l_patch_size : Union[str, int, list]
            Patch sizes configuration.
        k_patch_size : int
            Patch size for variables.
        stride : int
            Stride for patch splitting. If None, uses patch size as default.
        rotate : bool
            Apply rotational positional embeddings.
        addv : bool
            Whether to add RoPE information to value in attention. If False, only rotate the key and query embeddings.
        bin_att : bool
            Use binary attention biases to encode variate indices (any-variate attention).
        rope_theta_init : str
            Initialization for TRoPE, default is 'exp', as used in the paper. Options: ['exp', 'linear', 'uniform', 'rope'].
        min_period : float
            Minimum initialized period coefficient for rotary embeddings.
        max_period : float
            Maximum initialized period coefficient for rotary embeddings.
        learn_tem_emb : bool
            Whether to use learnable temporal embeddings.
        learnable_rope : bool
            Make period coefficient in TRoPE learnable.
        abs_tem_emb : bool
            Use absolute temporal embeddings if True.
        structured_mask : bool
            Apply structured mask or not.
        max_seq_len : int
            Maximum sequence length for the input time series.
        theta_base : int
            Base frequency of vanilla RoPE.
        t_layers : int
            Number of temporal attention layers.
        v_layers : int
            Number of variable attention layers.
        patch_share_backbone : bool
            Share Transformer backbone across patches.
        n_heads : int
            Number of attention heads in the multi-head attention mechanism.
        d_k : int
            Dimensionality of key embeddings in attention.
        d_v : int
            Dimensionality of value embeddings in attention.
        d_inner : int
            Size of inner layers in the feed-forward network.
        dropout : float
            Dropout rate for regularization during training.
        in_channels : int
            Number of input channels in the time series data. We only consider univariable.
        f_hidden_size : int
            Hidden size for the feed-forward layers.
        use_norm : bool
            Whether to apply instance normalization.
        **kwargs : dict
            Additional keyword arguments for extended functionality.
        """

        super().__init__(**kwargs)
        
        self.l_patch_size = convert_to_list(l_patch_size)
        self.use_norm = use_norm
        # Model
        self.model = ElasTST_backbone(l_patch_size=self.l_patch_size, 
            stride=stride, 
            k_patch_size=k_patch_size, 
            in_channels=in_channels,
            t_layers=t_layers, 
            v_layers=v_layers, 
            hidden_size=f_hidden_size, 
            d_inner=d_inner,
            n_heads=n_heads, 
            d_k=d_k, 
            d_v=d_v,
            dropout=dropout,
            rotate=rotate, 
            max_seq_len=max_seq_len, 
            theta=theta_base,
            addv=addv, 
            bin_att=bin_att,
            learn_tem_emb=learn_tem_emb, 
            abs_tem_emb=abs_tem_emb, 
            learnable_theta=learnable_rope, 
            structured_mask=structured_mask,
            rope_theta_init=rope_theta_init, 
            min_period=min_period, 
            max_period=max_period,
            patch_share_backbone=patch_share_backbone
        )
        
        self.loss_fn = nn.MSELoss(reduction='none')
        self.instance_norm = InstanceNorm()
    
    def forward(self, batch_data, pred_len, dataset_name=None):
        new_pred_len = pred_len
        for p in self.l_patch_size:
            new_pred_len = self.check_divisibility(new_pred_len, p)
        
        B, _, K = batch_data.past_target_cdf.shape
        past_target = batch_data.past_target_cdf
        past_observed_values = batch_data.past_observed_values
        
        if self.use_norm:
            past_target = self.instance_norm(past_target, 'norm')

        # future_observed_values is the mask indicate whether there is a value in a position
        future_observed_values = torch.zeros([B, new_pred_len, K]).to(batch_data.future_observed_values.device)

        pred_len = batch_data.future_observed_values.shape[1]
        future_observed_values[:,:pred_len] = batch_data.future_observed_values

        # target placeholder
        future_placeholder = torch.zeros([B, new_pred_len, K]).to(batch_data.past_target_cdf.device)

        x, pred_list = self.model(past_target, future_placeholder, past_observed_values, future_observed_values, dataset_name=dataset_name)
        dec_out = x[:, :pred_len]
        if self.use_norm:
            dec_out = self.instance_norm(dec_out, 'denorm')

        return dec_out # [b l k], [b l k #patch_size]


    def loss(self, batch_data, reduce='none'):
        max_pred_len = batch_data.max_prediction_length if batch_data.max_prediction_length is not None else self.max_prediction_length
            
        predict = self(batch_data, max_pred_len, dataset_name=None, )
        target = batch_data.future_target_cdf
        
        observed_values = batch_data.future_observed_values
        loss = self.loss_fn(target, predict)

        loss = self.get_weighted_loss(observed_values, loss, reduce=reduce)
        
        if reduce=='mean':
            loss = loss.mean()
        return loss

    def forecast(self, batch_data, num_samples=None):
        # max_pred_len = batch_data.max_prediction_length if batch_data.max_prediction_length is not None else max(self.prediction_length)
        max_pred_len = batch_data.future_target_cdf.shape[1]
        outputs = self(batch_data, max_pred_len, dataset_name=None, )
        return outputs.unsqueeze(1)
    
    def check_divisibility(self, pred_len, patch_size):
        if pred_len % patch_size == 0:
            return pred_len
        else:  
            return (pred_len // patch_size + 1) * patch_size  

    def get_weighted_loss(self, observed_values, loss, reduce='mean'):
        loss_weights, _ = observed_values.min(dim=-1, keepdim=True)
        loss = weighted_average(loss, weights=loss_weights, dim=1, reduce=reduce)
        return loss