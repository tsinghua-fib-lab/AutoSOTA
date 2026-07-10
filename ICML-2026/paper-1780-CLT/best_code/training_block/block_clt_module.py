import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import esm
from esm.model.esm2 import ESM2
from clt_module import ESM2ActivationCollector
from block_clt_model import BlockCrossLayerTranscoder

# ──────────────────────────────────────────────────────────────────────────────
# Shape suffixes:
# B: Batch Size 
# L: Total number of LM layers
# T: Sequence length of protein (variable)
# D: CLT Latent dim (d_hidden)
# H: Embedding Dimension of LM (d_model)
# S: B * T
# ──────────────────────────────────────────────────────────────────────────────

class CLTLightningModule(pl.LightningModule):
    def __init__(self, args, esm2_weight=None):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.num_layers = args.num_layers 
        
        # 1. Initialize Block CLT
        self.clt = BlockCrossLayerTranscoder(
            num_layers=self.num_layers,
            d_model=args.d_model,
            d_hidden=args.d_hidden,
            k=args.k,
            auxk=args.auxk,
            batch_size=args.batch_size,
            dead_steps_threshold=args.dead_steps_threshold,
            block_size=args.block_size,
        )
        
        # 2. Initialize Tokenizer
        self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # 3. Initialize & Load ESM Model
        self.esm_model = ESM2(
            num_layers=args.num_layers,
            embed_dim=args.d_model,
            attention_heads=20,
            alphabet=self.alphabet,
            token_dropout=False
        )
        esm_weights_path = esm2_weight if esm2_weight is not None else args.esm2_weight
        self._load_esm_weights(esm_weights_path)

        # 4. Freeze ESM
        self.esm_model.eval()
        for param in self.esm_model.parameters():
            param.requires_grad = False

        # 5. Setup Collector
        self.collector = ESM2ActivationCollector(self.esm_model, self.alphabet)
        self.collector.register_hooks()
        
    def _load_esm_weights(self, esm_pretrained: str):
        if not os.path.exists(esm_pretrained):
            raise FileNotFoundError(f"CRITICAL: ESM weights not found at {esm_pretrained}")

        print(f"Loading ESM weights from {esm_pretrained}...")
        data = torch.load(esm_pretrained, map_location="cpu", weights_only=False)
        if "model" in data: data = data["model"]

        ckpt = {}
        for k, v in data.items():
            new_key = k.replace("encoder.sentence_encoder.", "").replace("encoder.", "")
            ckpt[new_key] = v

        missing, unexpected = self.esm_model.load_state_dict(ckpt, strict=False)
        critical_missing = [k for k in missing if "layers" in k]
        if len(critical_missing) > 0:
            raise RuntimeError(f"CRITICAL: Missing layers: {critical_missing[:5]}")
        print("SUCCESS: ESM weights loaded correctly.")

    def tokenize(self, seqs):
        """Helper to tokenize a list of strings."""
        data = [("protein", seq) for seq in seqs]
        _, _, batch_tokens = self.batch_converter(data)
        return batch_tokens.to(self.device)

    def training_step(self, batch, batch_idx):
        seqs = batch["Sequence"]
        batch_size = len(seqs)
        
        # 1. Tokenize & Collect
        tokens_BT = self.tokenize(seqs)
        
        x_stack_trajectory_SLH, x_mlp_stack_trajectory_SLH, mask_S = self.collector.collect(tokens_BT)

        # 2. Run CLT Forward
        recons_stack_SLH, auxk_stack_SLH, dead_mask_stack_LD = self.clt(x_stack_trajectory_SLH)
        
        total_loss = 0
        total_mse = 0
        total_aux = 0
        total_nmse = 0
        auxk_coef = 1.0 / 32.0 
        
        # 3. Identify Valid Tokens (Masking applied here)
        valid_indices = torch.nonzero(mask_S).squeeze()
        
        # 4. Cumulative Reconstruction Loss
        for l in range(self.num_layers):
            #true_state_SH = x_stack_trajectory_SLH[:, l + 1, :]
            true_state_SH = x_mlp_stack_trajectory_SLH[:, l, :]
            pred_state_SH = recons_stack_SLH[:, l, :]
            
            # --- APPLY MASK ---
            true_masked = true_state_SH[valid_indices]
            pred_masked = pred_state_SH[valid_indices]
            # A. Main Reconstruction Loss
            mse = F.mse_loss(pred_masked, true_masked)
            # B. NMSE
            target_var = torch.var(true_masked) + 1e-8
            nmse = mse / target_var
            
            total_loss += nmse
            total_mse += mse
            total_nmse += nmse
            
            # C. AuxK Loss
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
    
    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.lr, weight_decay=1e-5)

    def validation_step(self, batch, batch_idx):
        seqs = batch["Sequence"]
        batch_size = len(seqs)
        
        tokens_BT = self.tokenize(seqs)
        x_stack_trajectory_SLH, x_mlp_stack_trajectory_SLH, mask_S = self.collector.collect(tokens_BT)
        
        recons_stack_SLH, _, _ = self.clt(x_stack_trajectory_SLH)
        
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
