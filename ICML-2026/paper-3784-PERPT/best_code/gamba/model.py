from typing import Dict, Optional, Set, Tuple, Type
import math
import numpy as np
from sequence_models.constants import MSA_PAD, START, STOP
from sequence_models.convolutional import ByteNetBlock, ByteNetLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from torch.distributions import Gamma, Normal


from gamba.constants import MSA_ALPHABET_PLUS, TaskType
from gamba.losses import (
    OAMaskedCrossEntropyLoss,
    GaussianNLLLoss,
    FocalGaussianNLLLoss,
    WeightedGaussianNLLLoss,
    InverseGammaNLLLoss,
    PoissonNLLLoss,
)


OTHER_METRICS_KEY = "other_metrics"


class LogitOnlyModelWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return {"logits": self.module(*args, **kwargs)}


class OrderAgnosticDiffusionModel(nn.Module):
    def __init__(
        self, module: nn.Module, padding_id: int, aux_loss_weight: float = 1.0
    ):
        super().__init__()
        self.module = module
        self.loss_func = OAMaskedCrossEntropyLoss(reweight=True)
        self.padding_id = padding_id
        self.aux_loss_weight = aux_loss_weight

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        mask: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        n_tokens = mask.sum()
        input_mask = src != self.padding_id
        n_seq = torch.tensor(len(src), device=src.device)
        n_processed = input_mask.sum()

        output = self.module(src, input_mask=input_mask.unsqueeze(-1))
        ce_loss, nll_loss = self.loss_func(
            output["logits"], tgt, mask, timestep, input_mask
        )
        aux_loss = output.get("aux_loss", 0.0)

        with torch.no_grad():
            pred_tok = torch.argmax(output["logits"], dim=-1)
            accu = ((pred_tok == tgt) * mask).float().sum() / n_tokens

        ce_loss = ce_loss / n_tokens
        nll_loss = nll_loss / n_tokens
        other_metrics = {
            "nll_loss": nll_loss,
            "accuracy": accu,
        }
        if hasattr(output, "aux_loss"):
            # log the original CE loss and the auxiliary loss
            other_metrics["ce_loss"] = ce_loss
            other_metrics["aux_loss"] = aux_loss

        outputs = {
            "logits": output["logits"],
            "loss": ce_loss + self.aux_loss_weight * aux_loss,
            OTHER_METRICS_KEY: other_metrics,
            "n_tokens": n_tokens,
            "n_seqs": n_seq,
            "n_processed": n_processed,
        }
        return outputs


class ARDiffusionModel(nn.Module):
    def __init__(self, module: nn.Module, aux_loss_weight: float = 1.0):
        super().__init__()
        self.module = module
        self.aux_loss_weight = aux_loss_weight

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> dict:
        n_tokens = (tgt >= 0).sum()
        n_seq = torch.tensor(len(src), device=src.device)
        n_processed = n_tokens - len(tgt)  # -1 token per sequence for the shift

        output = self.module(src)
        ce_loss = F.cross_entropy(
            # flatten into N*L x C
            output["logits"][:, :-1, :].reshape(-1, output["logits"].shape[-1]),
            tgt[:, 1:].flatten(),
            reduction="mean",
        )
        aux_loss = output.get("aux_loss", 0.0)

        # compute the accuracy
        with torch.no_grad():
            pred_tok = torch.argmax(output["logits"][:, :-1, :], dim=-1)
            accu = (
                (pred_tok == tgt[:, 1:]) * (tgt[:, 1:] >= 0)
            ).float().sum() / n_tokens

        other_metrics = {
            "accuracy": accu,
        }
        if hasattr(output, "aux_loss"):
            # log the original CE loss and the auxiliary loss
            other_metrics["ce_loss"] = ce_loss
            other_metrics["aux_loss"] = aux_loss
        outputs = {
            "logits": output["logits"],
            "loss": ce_loss + self.aux_loss_weight * aux_loss,
            OTHER_METRICS_KEY: other_metrics,
            "n_tokens": n_tokens,
            "n_seqs": n_seq,
            "n_processed": n_processed,
        }
        return outputs


def get_flip_inds(seq, lengths):
    # hidden_states: b x ell
    # lengths: b x 1
    device = seq.device
    b, ell = seq.shape[:2]
    xi = torch.arange(b, device=device).repeat_interleave(ell, dim=0).view(b, ell)
    yi = (torch.arange(ell, device=device) + (ell - lengths)) % ell
    return xi, yi


def flip_with_padding(seq, lengths):
    xi, yi = get_flip_inds(seq, lengths)
    qes = seq.flip(dims=(1,))
    return qes[xi, yi]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[0, : x.size(1)]
        return self.dropout(x)


class JambagambaModel(nn.Module):
    def __init__(
        self,
        jambalm: nn.Module,
        d_model: int,
        nhead: int,
        dim_feedfoward: int,
        n_layers: int,
        padding_id: int,
        weights_path="/home/mica/gamba/data_processing/data/240-mammalian/phyloP_weights.pkl",
    ):
        super().__init__()
        self.embedder = ARDiffusionModel(jambalm).module.model

        # need to split d_model into lm head and scaling head 
        self.each_dim = int(d_model / 2)
        self.lm_head = nn.Linear(d_model, jambalm.vocab_size)
        self.scaling_head = nn.Linear(d_model, 2)
       
        #seq_embedding gets full dimensionality, there is no more any value embedding
        self.seq_embedding = nn.Embedding(jambalm.vocab_size, d_model)
        
        # real number loss
        #self.cons_loss_func = GaussianNLLLoss()
        self.cons_loss_func = FocalGaussianNLLLoss()
        #self.cons_loss_func = WeightedGaussianNLLLoss(weights_path=weights_path)
        self.mse_loss_func = nn.MSELoss()


    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> dict:
        # print("shape of tgt: ", tgt.shape)
        # print("shape of src: ", src.shape)
       
        seq_tgt, conservation_tgt = tgt.split(1, dim=1)
        seq_tgt = seq_tgt.squeeze(1).long()
        conservation_tgt = conservation_tgt.squeeze(1)
      
        # 0 is a token not a padding for sequence
        n_tokens = (seq_tgt >= 0).sum()

        
        seq, conservation = src.split(1, dim=1)
        #print("shape of seq: ", seq.shape)
        seq = seq.squeeze(1).long()
        #print("post squeeze & long shape of seq: ", seq.shape)
        conservation = conservation.squeeze(1)
        #print(f"conservation shape:", conservation.shape)
        
        device = src.device

        n_seq = torch.tensor(seq.size(1), device=device)
        n_processed = n_tokens - seq_tgt.size(1)  # -1 token per sequence for the shift

        # embed seq, conservation and gap separately
        emb_seq = self.seq_embedding(seq)
        
        inputs_embeds = emb_seq
       

        #print(f"shape of input_embeds: {inputs_embeds.shape}")
        # need to set the embedded inputs to inputs_embeds to values in the Jamba model
        output = self.embedder(inputs_embeds=inputs_embeds)["last_hidden_state"]
       
        # put the outputs through their respective linear layers
        seq_logits = self.lm_head(output)
        scaling_logits = self.scaling_head(output)
        
        # apply CE loss on the seq_logits
        ce_loss = F.cross_entropy(
            seq_logits[:, :-1, :].reshape(-1, seq_logits.shape[-1]),
            seq_tgt[:, 1:].flatten(),
            reduction="mean",
        )
        #scaling_logits = scaling_logits[:, :-1, :]
        #conservation_tgt = conservation_tgt[:, 1:]
        # apply GaussianNLLLoss from losses.py on the scaling_logits
        gaussian_loss = self.cons_loss_func(
            scaling_logits, conservation_tgt
        )
        #extract mean and variance from scaling logits
        pred = scaling_logits
        tgt = conservation_tgt
        # mask is where tgt is not equal to -100
        mask = tgt != -100

        mean = pred[:, :, 0]
        log_var = pred[:, :, 1]

        # apply the mask to mean, log_var and tgt
        mean = mean[mask]
        log_var = log_var[mask]
        tgt = tgt[mask]

        #check MSE loss on the unmasked portions
        mse_loss = self.mse_loss_func(mean, tgt)
        

        #clip the gaussian loss
        #gaussian_loss = torch.clamp(gaussian_loss, min=0.0, max=1.0)
        # apply PoissonNLLLoss from losses.py on the gap_logits
        # poisson_loss = self.gap_loss_func(gap_logits[:, :-1, :], gap_tgt[:, 1:])

        #print("CE LOSS: ", ce_loss)
        #print("GAUSSIAN LOSS: ", gaussian_loss)
        # check if any loss is NaN
        if math.isnan(ce_loss):
            raise ValueError("CE Loss is NaN")
        if math.isnan(gaussian_loss):
            raise ValueError("Gaussian Loss is NaN")
        # print("POISSON LOSS: ", poisson_loss)
        #print("LOSS:", ce_loss + gaussian_loss)

        # print(f"shape of the scaling_logits: {scaling_logits.shape}")
        # print(f"shape of the seq_logits: {seq_logits.shape}")
        # compute the accuracy
        with torch.no_grad():
            pred_tok_seq = torch.argmax(seq_logits[:, :-1, :], dim=-1)
            seq_accu = (
                (pred_tok_seq == seq_tgt[:, 1:]) * (seq_tgt[:, 1:] >= 0)
            ).float().sum() / n_tokens

        other_metrics = {
            "accuracy": seq_accu,
        }
        if hasattr(output, "aux_loss"):
            # log the original CE loss and the auxiliary loss
            other_metrics["ce_loss"] = ce_loss
        outputs = {
            "seq_logits": seq_logits,
            "scaling_logits": scaling_logits,
            # "gap_logits": gap_logits,
            "loss": ce_loss + gaussian_loss,  # + poisson_loss,
            "cross_entropy_loss": ce_loss,
            "gaussian_loss": gaussian_loss,
            OTHER_METRICS_KEY: other_metrics,
            "accuracy": seq_accu,
            "n_tokens": n_tokens,
            "n_seqs": n_seq,
            "n_processed": n_processed,
            "representation": output,
            "conservation_tgt": conservation_tgt,
            "mse_loss": mse_loss,
        }
        return outputs
    
class JambaGambaNoConsModel(nn.Module):
    def __init__(
        self,
        jambalm: nn.Module,
        d_model: int,
        nhead: int,
        dim_feedfoward: int,
        n_layers: int,
        padding_id: int,

    ):
        super().__init__()
        self.embedder = ARDiffusionModel(jambalm).module.model
        self.lm_head = nn.Linear(d_model, jambalm.vocab_size)
        self.seq_embedding = nn.Embedding(jambalm.vocab_size, d_model)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> dict:
        # Get sequence data only
        seq_tgt = tgt[:, 0].long()  # Take first channel only
        seq = src[:, 0].long()      # Take first channel only
        
        # Count valid tokens (not padding)
        n_tokens = (seq_tgt >= 0).sum()
        n_seq = torch.tensor(seq.size(1), device=src.device)
        n_processed = n_tokens - seq_tgt.size(1)

        # Embed sequences
        emb_seq = self.seq_embedding(seq)
        
        # Get model outputs
        output = self.embedder(inputs_embeds=emb_seq)["last_hidden_state"]
        seq_logits = self.lm_head(output)

        # Calculate CE loss
        ce_loss = F.cross_entropy(
            seq_logits[:, :-1, :].reshape(-1, seq_logits.shape[-1]),
            seq_tgt[:, 1:].flatten(),
            reduction="mean",
        )

        # Calculate accuracy
        with torch.no_grad():
            pred_tok_seq = torch.argmax(seq_logits[:, :-1, :], dim=-1)
            seq_accu = (
                (pred_tok_seq == seq_tgt[:, 1:]) * (seq_tgt[:, 1:] >= 0)
            ).float().sum() / n_tokens

        outputs = {
            "seq_logits": seq_logits,
            "loss": ce_loss,
            "cross_entropy_loss": ce_loss,
            OTHER_METRICS_KEY: {"accuracy": seq_accu},
            "accuracy": seq_accu,
            "n_tokens": n_tokens,
            "n_seqs": n_seq,
            "n_processed": n_processed,
            "representation": output,
        }
        return outputs
        
class JambaGambaNOALMModel(nn.Module):
    def __init__(
        self,
        jambalm: nn.Module,
        d_model: int,
        nhead: int,
        dim_feedfoward: int,
        n_layers: int,
        padding_id: int,
    ):
        super().__init__()
        self.embedder = ARDiffusionModel(jambalm).module.model
        self.scaling_head = nn.Linear(d_model, 2)
       
        #seq_embedding gets full dimensionality, there is no more any value embedding
        self.seq_embedding = nn.Embedding(jambalm.vocab_size, d_model)
        
        # real number loss
        self.cons_loss_func = GaussianNLLLoss()
        self.mse_loss_func = nn.MSELoss()


    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> dict:
        # print("shape of tgt: ", tgt.shape)
        # print("shape of src: ", src.shape)
       
        seq_tgt, conservation_tgt = tgt.split(1, dim=1)
        seq_tgt = seq_tgt.squeeze(1).long()
        conservation_tgt = conservation_tgt.squeeze(1)
      
        # 0 is a token not a padding for sequence
        n_tokens = (seq_tgt >= 0).sum()

        
        seq, conservation = src.split(1, dim=1)
        #print("shape of seq: ", seq.shape)
        seq = seq.squeeze(1).long()
        #print("post squeeze & long shape of seq: ", seq.shape)
        conservation = conservation.squeeze(1)
        #print(f"conservation shape:", conservation.shape)
        
        device = src.device

        n_seq = torch.tensor(seq.size(1), device=device)
        n_processed = n_tokens - seq_tgt.size(1)  # -1 token per sequence for the shift

        # embed seq, conservation and gap separately
        emb_seq = self.seq_embedding(seq)
        
        inputs_embeds = emb_seq
       

        #print(f"shape of input_embeds: {inputs_embeds.shape}")
        # need to set the embedded inputs to inputs_embeds to values in the Jamba model
        output = self.embedder(inputs_embeds=inputs_embeds)["last_hidden_state"]
       
        # put the outputs through their respective linear layers
        scaling_logits = self.scaling_head(output)
        
        #scaling_logits = scaling_logits[:, :-1, :]
        #conservation_tgt = conservation_tgt[:, 1:]
        # apply GaussianNLLLoss from losses.py on the scaling_logits
        gaussian_loss = self.cons_loss_func(
            scaling_logits, conservation_tgt
        )
        #extract mean and variance from scaling logits
        pred = scaling_logits
        tgt = conservation_tgt
        # mask is where tgt is not equal to -100
        mask = tgt != -100

        mean = pred[:, :, 0]
        log_var = pred[:, :, 1]

        # apply the mask to mean, log_var and tgt
        mean = mean[mask]
        log_var = log_var[mask]
        tgt = tgt[mask]

        #check MSE loss on the unmasked portions
        mse_loss = self.mse_loss_func(mean, tgt)
        

        #clip the gaussian loss
        #gaussian_loss = torch.clamp(gaussian_loss, min=0.0, max=1.0)
        # apply PoissonNLLLoss from losses.py on the gap_logits
        # poisson_loss = self.gap_loss_func(gap_logits[:, :-1, :], gap_tgt[:, 1:])

        #print("CE LOSS: ", ce_loss)
        #print("GAUSSIAN LOSS: ", gaussian_loss)
        # check if any loss is NaN
        if math.isnan(mse_loss):
            raise ValueError("MSE Loss is NaN")
        if math.isnan(gaussian_loss):
            raise ValueError("Gaussian Loss is NaN")
        # print("POISSON LOSS: ", poisson_loss)
        #print("LOSS:", ce_loss + gaussian_loss)
        other_metrics = {}
        outputs = {
            "scaling_logits": scaling_logits,
            # "gap_logits": gap_logits,
            "loss": gaussian_loss,  # + poisson_loss,
            "gaussian_loss": gaussian_loss,
            OTHER_METRICS_KEY: other_metrics,
            "n_tokens": n_tokens,
            "n_seqs": n_seq,
            "n_processed": n_processed,
            "representation": output,
            "conservation_tgt": conservation_tgt,
            "mse_loss": mse_loss,
        }
        return outputs
    

class JambaGambaModelWithDegeneracies(nn.Module):
    def __init__(
        self,
        jambalm: nn.Module,
        d_model: int,
        nhead: int,
        dim_feedfoward: int,
        n_layers: int,
        padding_id: int,
    ):
        super(JambaGambaModelWithDegeneracies, self).__init__()
        self.embedder = ARDiffusionModel(jambalm).module.model

        # need to split d_model into lm head, scaling head
        self.each_dim = int(d_model / 2)
        self.lm_head = nn.Linear(d_model, jambalm.vocab_size)
        self.scaling_head = nn.Linear(d_model, 2)
       

        #seq_embedding gets full dimensionality, there is no more any value embedding
        self.seq_embedding = nn.Embedding(jambalm.vocab_size, d_model)
        

        # real number loss
        self.cons_loss_func = GaussianNLLLoss()

     

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> dict:
        # Split the target tensor into sequence and conservation targets
        seq_tgt, conservation_tgt, degeneracies_tgt = tgt.split(1, dim=1)
        seq_tgt = seq_tgt.squeeze(1).long()
        conservation_tgt = conservation_tgt.squeeze(1)
        degeneracies_tgt = degeneracies_tgt.squeeze(1)
        n_tokens = (seq_tgt >= 0).sum()

        # Split the source tensor into sequence and conservation inputs
        seq, conservation, degeneracies = src.split(1, dim=1)
        seq = seq.squeeze(1).long()
        conservation = conservation.squeeze(1)
        degeneracies = degeneracies.squeeze(1)
        device = src.device

        n_seq = torch.tensor(seq.size(1), device=device)
        n_processed = n_tokens - seq_tgt.size(1)  # -1 token per sequence for the shift

        # Embed the sequence
        emb_seq = self.seq_embedding(seq)
        inputs_embeds = emb_seq

        # Pass the embedded inputs through the model
        output = self.embedder(inputs_embeds=inputs_embeds)["last_hidden_state"]

        # Generate logits for sequence and scaling
        seq_logits = self.lm_head(output)
        scaling_logits = self.scaling_head(output)

        # Exclude the logits for the first and last tokens for conservation
        scaling_logits = scaling_logits[:, 1:-1]
        conservation_tgt = conservation_tgt[:, 1:-1]
        degeneracies_tgt = degeneracies_tgt[:, 1:-1]

        # Apply cross-entropy loss on the sequence logits
        ce_loss = F.cross_entropy(
            seq_logits[:, :-1, :].reshape(-1, seq_logits.shape[-1]),
            seq_tgt[:, 1:].flatten(),
            reduction="mean",
        )

        # Apply GaussianNLLLoss on the scaling logits
        gaussian_loss = self.cons_loss_func(
            scaling_logits[:, :-1, :], conservation_tgt[:, 1:]
        )

        # Check if any loss is NaN
        if math.isnan(ce_loss):
            raise ValueError("CE Loss is NaN")
        if math.isnan(gaussian_loss):
            raise ValueError("Gaussian Loss is NaN")

        # Compute the accuracy
        with torch.no_grad():
            pred_tok_seq = torch.argmax(seq_logits[:, :-1, :], dim=-1)
            seq_accu = (
                (pred_tok_seq == seq_tgt[:, 1:]) * (seq_tgt[:, 1:] >= 0)
            ).float().sum() / n_tokens

        other_metrics = {
            "accuracy": seq_accu,
        }
        if hasattr(output, "aux_loss"):
            other_metrics["ce_loss"] = ce_loss

        outputs = {
            "seq_logits": seq_logits,
            "scaling_logits": scaling_logits,
            "loss": ce_loss + gaussian_loss,
            "cross_entropy_loss": ce_loss,
            "gaussian_loss": gaussian_loss,
            OTHER_METRICS_KEY: other_metrics,
            "accuracy": seq_accu,
            "n_tokens": n_tokens,
            "n_seqs": n_seq,
            "n_processed": n_processed,
            "representation": output,
            "conservation_tgt": conservation_tgt,
            "degeneracies_tgt": degeneracies_tgt,
        }
        return outputs
    
def _create_bytenet(
    task: TaskType, model_config: dict, pad_id: int
) -> Tuple[ByteNetLM, Set[Type[ByteNetBlock]]]:
    pretrained = model_config.pop("pretrained", False)
    if pretrained:
        raise ValueError("Pretrained models not supported for ByteNet")

    n_tokens = len(MSA_ALPHABET_PLUS)
    d_embed = model_config["d_embed"]
    d_model = model_config["d_model"]
    n_layers = model_config["n_layers"]
    kernel_size = model_config["kernel_size"]
    r = model_config["r"]
    slim = model_config.get("slim", True)
    activation = model_config.get("activation", "gelu")
    dropout = model_config.get("dropout", 0.0)

    return (
        ByteNetLM(
            n_tokens,
            d_embed,
            d_model,
            n_layers,
            kernel_size,
            r,
            causal=task == TaskType.LM,
            padding_idx=pad_id,
            dropout=dropout,
            tie_weights=False,
            final_ln=True,
            slim=slim,
            activation=activation,
        ),
        {ByteNetBlock},
    )


def _get_hf_model(
    model_name: str,
    pad_token_id: int,
    *,
    model_config: Optional[dict] = None,
    pretrained: bool = False,
    trust_remote_code: bool = True,
) -> nn.Module:
    if model_config and pretrained:
        # can't overwrite the config of a pretrained model
        raise ValueError("Cannot specify both model_config and pretrained")
    elif pretrained:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )

        # if we need to change the padding token
        if pad_token_id != model.config.pad_token_id:
            # locate the single embedding module
            embeddings = []
            for layer in model.modules():
                if isinstance(layer, nn.Embedding):
                    embeddings.append(layer)
            if len(embeddings) != 1:
                raise ValueError(f"Expected 1 embedding layer, got {len(embeddings)}")

            # update the padding index
            embeddings[0].padding_idx = pad_token_id
            embeddings[0]._fill_padding_idx_with_zero()
    else:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        for k, v in model_config.items():
            if not hasattr(config, k):
                raise ValueError(f"Unknown config key: {k}")
            setattr(config, k, v)

        # ensure the vocab size is a multiple of 8 to maximize tensor core utilization
        model_config["vocab_size"] = (
            np.ceil(len(MSA_ALPHABET_PLUS) / 8).astype(int).item() * 8
        )
        model_config["pad_token_id"] = MSA_ALPHABET_PLUS.index(
            MSA_PAD
        )  # FIXME: MSA_PAD or pad_token_id (which is mask_id in bytenet
        model_config["bos_token_id"] = MSA_ALPHABET_PLUS.index(START)
        model_config["eos_token_id"] = MSA_ALPHABET_PLUS.index(STOP)

        # merge the updates into the default config
        config = type(config).from_dict({**config.to_dict(), **model_config})
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=trust_remote_code
        )
    return model


def _create_jamba(
    task: TaskType, model_config: dict, pad_id: int, trust_remote_code:bool=True,
) -> Tuple[nn.Module, Set[Type[nn.Module]]]:
    pretrained = model_config.pop("pretrained", False)
    model = _get_hf_model(
        "ai21labs/Jamba-v0.1", pad_id, pretrained=pretrained, model_config=model_config, trust_remote_code=trust_remote_code,
    )
    return model, {type(layer) for layer in model.model.layers}


def create_model(
    task: TaskType, model_type: str, model_config: dict, pad_id: int, trust_remote_code:bool=True,
) -> Tuple[nn.Module, Set[Type[nn.Module]]]:
    if model_type == "bytenet":
        model, blocks = _create_bytenet(task, model_config, pad_id)
    elif model_type == "jamba":
        model, blocks = _create_jamba(task, model_config, pad_id)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # assume all non-HF models only output logits, so we wrap it
    # to make it look like a HF model
    if not isinstance(model, PreTrainedModel):
        model = LogitOnlyModelWrapper(model)

    return model, blocks
