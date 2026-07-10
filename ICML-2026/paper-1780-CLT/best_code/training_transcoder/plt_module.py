import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import os
import esm
from esm.model.esm2 import ESM2 
from plt_model import PerLayerTranscoder
import sys
sys.path.append("../training")
try:
    from clt_module import ESM2ActivationCollector
except ImportError:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# Shape suffixes:
# B: Batch Size 
# L: Total number of LM layers
# T: Sequence length of protein (variable)
# D: CLT Latent dim (d_hidden)
# H: Embedding Dimension of LM (d_model)
# S: B * T
# ──────────────────────────────────────────────────────────────────────────────

class PLTLightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.num_layers = args.num_layers 
        
        self.plt = PerLayerTranscoder(
            num_layers=self.num_layers,
            d_model=args.d_model,
            d_hidden=args.d_hidden,
            k=args.k,
            auxk=args.auxk,
            batch_size=args.batch_size,
            dead_steps_threshold=args.dead_steps_threshold,
        )
        
        self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm_model = ESM2(
            num_layers=args.num_layers,
            embed_dim=args.d_model,
            attention_heads=20,
            alphabet=self.alphabet,
            token_dropout=False
        )
        self._load_esm_weights(args.esm2_weight)
        self.esm_model.eval()
        for param in self.esm_model.parameters(): param.requires_grad = False
        
        self.collector = ESM2ActivationCollector(self.esm_model, self.alphabet)
        self.collector.register_hooks()
        
    def _load_esm_weights(self, esm_pretrained):
        if not os.path.exists(esm_pretrained): raise FileNotFoundError(f"Weights not found: {esm_pretrained}")
        data = torch.load(esm_pretrained, map_location="cpu", weights_only=False)
        if "model" in data: data = data["model"]
        ckpt = {k.replace("encoder.sentence_encoder.", "").replace("encoder.", ""): v for k, v in data.items()}
        self.esm_model.load_state_dict(ckpt, strict=False)

    def tokenize(self, seqs):
        data = [("protein", seq) for seq in seqs]
        _, _, batch_tokens = self.batch_converter(data)
        return batch_tokens.to(self.device)

    def training_step(self, batch, batch_idx):
        seqs = batch["Sequence"]
        batch_size = len(seqs)
        
        tokens_BT = self.tokenize(seqs)
        x_stack_trajectory_SLH, x_mlp_stack_trajectory_SLH, mask_S = self.collector.collect(tokens_BT) # (B*T, L, H), (B*T, L, H), (B*T)
        
        # PLT Forward
        recons_stack_SLH, auxk_stack_SLH, dead_mask_stack_LD = self.plt(x_stack_trajectory_SLH)
        
        total_loss = 0
        total_mse = 0
        total_aux = 0
        total_nmse = 0
        auxk_coef = 1.0 / 32.0 
        
        valid_indices = torch.nonzero(mask_S).squeeze()
        
        for l in range(self.num_layers):
            # Target: Layers 1 to L
            true_state_SH = x_mlp_stack_trajectory_SLH[:, l, :]
            pred_state_SH = recons_stack_SLH[:, l, :]
            
            true_masked = true_state_SH[valid_indices]
            pred_masked = pred_state_SH[valid_indices]
            
            mse = F.mse_loss(pred_masked, true_masked)
            target_var = torch.var(true_masked) + 1e-8
            nmse = mse / target_var
            
            total_loss += nmse
            total_mse += mse
            total_nmse += nmse
            
            if auxk_stack_SLH is not None:
                residual = (true_masked - pred_masked).detach()
                aux_out_masked = auxk_stack_SLH[:, l, :][valid_indices]
                
                aux_loss = F.mse_loss(aux_out_masked, residual)
                total_aux += aux_loss
                total_loss += (aux_loss * auxk_coef)
                
                self.log(f"train/aux_loss_layer_{l}", aux_loss, batch_size=batch_size)
                
            self.log(f"train/mse_layer_{l}", mse, batch_size=batch_size)
            self.log(f"train/nmse_layer_{l}", nmse, batch_size=batch_size)
            self.log(f"train/dead_neurons_{l}", dead_mask_stack_LD[l].sum().float(), batch_size=batch_size)

        avg_nmse = total_nmse / self.num_layers
        self.log("train/loss", total_loss, batch_size=batch_size)
        self.log("train/avg_nmse", avg_nmse, batch_size=batch_size)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        seqs = batch["Sequence"]
        batch_size = len(seqs)
        
        tokens_BT = self.tokenize(seqs)
        x_stack_trajectory_SLH, x_mlp_stack_trajectory_SLH, mask_S = self.collector.collect(tokens_BT)
        
        recons_stack_SLH, _, _ = self.plt(x_stack_trajectory_SLH)
        
        total_loss = 0
        total_nmse = 0
        
        valid_indices = torch.nonzero(mask_S).squeeze()
        
        for l in range(self.num_layers):
            true_state_SH = x_mlp_stack_trajectory_SLH[:, l, :]
            pred_state_SH = recons_stack_SLH[:, l, :]
            
            true_masked = true_state_SH[valid_indices]
            pred_masked = pred_state_SH[valid_indices]
            
            mse = F.mse_loss(pred_masked, true_masked)
            target_var = torch.var(true_masked) + 1e-8
            nmse = mse / target_var
            
            total_loss += nmse
            total_nmse += nmse
            
            self.log(f"val/nmse_layer_{l}", nmse, batch_size=batch_size)
            
        avg_nmse = total_nmse / self.num_layers
        self.log("val/loss", total_loss, prog_bar=True, batch_size=batch_size)
        self.log("val/avg_nmse", avg_nmse, batch_size=batch_size)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.lr, weight_decay=1e-5)
