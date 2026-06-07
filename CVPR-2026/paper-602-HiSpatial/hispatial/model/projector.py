import torch
from torch import nn
from transformers import PaliGemmaConfig


class CombinedMultiModalProjector(nn.Module):
    """Projects concatenated SigLIP + depth features into the language model's input space.

    The depth half of the weight matrix is zero-initialized so training starts
    from the pretrained RGB-only behavior.
    """

    def __init__(self, config: PaliGemmaConfig, original_projector=None):
        super().__init__()

        # Determine input dimensions
        siglip_dim = config.vision_config.hidden_size
        depth_encoder_dim = config.vision_config.hidden_size
        combined_dim = siglip_dim + depth_encoder_dim

        # Create a new linear layer
        self.linear = nn.Linear(combined_dim, config.vision_config.projection_dim, bias=True)

        # Zero initialization
        if original_projector is not None:
            with torch.no_grad():
                # Copy weights for the SigLIP part
                self.linear.weight[:, :siglip_dim] = original_projector.linear.weight
                self.linear.weight[:, siglip_dim:] = 0.0
                # Copy bias
                self.linear.bias = nn.Parameter(original_projector.linear.bias.clone())
        else:
            # Standard initialization if no original projector
            std = config.text_config.initializer_range
            self.linear.weight.data[:, :siglip_dim].normal_(mean=0.0, std=std)
            self.linear.weight.data[:, siglip_dim:] = 0.0
            if self.linear.bias is not None:
                self.linear.bias.data.zero_()

    def forward(self, combined_features):
        hidden_states = self.linear(combined_features)
        return hidden_states
