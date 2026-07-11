

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Normal, Categorical
from typing import Tuple, Optional, List
from nltk import Nonterminal
from sigs.encoder import Encoder
from sigs.decoder import Decoder
from sigs.stack import Stack
from sigs.grammar import GCFG, S, get_mask
import sys

class GrammarVAE(nn.Module):
    """Grammar Variational Autoencoder with complete VAE operations"""
    def __init__(self, config: dict):
        super().__init__()
        # Extract configurations
        enc_config = config['model']['encoder']
        dec_config = config['model']['decoder']
        shared_config = config['model']['shared']
        
        # Initialize encoder
        self.encoder = Encoder(
            input_dim=shared_config['output_size'],
            hidden_dim=enc_config['hidden_size'],
            z_dim=shared_config['z_dim'],
            max_length=shared_config['max_length'],
            conv_sizes=enc_config['conv_sizes'],
            kernel_sizes=enc_config['kernel_sizes'],
            use_batch_norm=enc_config['use_batch_norm']
        )
        
        # Initialize decoder
        self.decoder = Decoder(
            latent_rep_size=shared_config['z_dim'],
            hidden_size=dec_config['hidden_size'],
            output_size=shared_config['output_size'],
            max_length=shared_config['max_length'],
            rnn_type=dec_config['rnn_type'],
            num_layers=dec_config['num_layers']
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = shared_config['max_length']
        self.tightness = config['training']['tightness']
        

        
    def sample(self, mu: torch.Tensor, logvar: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
        """
        Reparameterization trick with multiple samples
        """
        batch_size, latent_dim = mu.shape


        # Expand mu and compute std = exp(0.5·logvar)
        mu     = mu.unsqueeze(1)                       # → (B,1,D)
        std    = (0.5 * logvar).exp().unsqueeze(1)     # → (B,1,D)
        # Sample epsilon
        eps    = torch.randn(batch_size, num_samples, latent_dim, device=mu.device)
        # Reparameterization
        z      = mu + eps * std * self.tightness
        return z


    
    def kl_divergence(self, mu, logvar):
        # Print intermediate values
        per_dim_kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        # print(f"Per-dim KL shape: {per_dim_kl.shape}, mean: {per_dim_kl.mean().item()}")
        kl_sum = per_dim_kl.sum(dim=1).mean().item()
        kl_mean = per_dim_kl.mean(dim=1).mean().item()
        # print(f"KL sum: {kl_sum}, KL mean: {kl_mean}, ratio: {kl_sum/kl_mean if kl_mean > 0 else 'inf'}")
        
        # Your actual implementation
        kld = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).mean(dim=1)
        return kld.mean()


    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Complete forward pass through the VAE
        """
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # Encode
        mu, logvar = self.encoder(x)
        
        # Sample latent vector
        z = self.sample(mu, logvar)
        
        # Decode
        logits = self.decoder(z)
        
        return logits, mu, logvar

    def generate(self, z: torch.Tensor, sample: bool = False) -> List:
        try:
            # print("Entering generate...")
            sys.stdout.flush()
            # print(f"Input z shape: {z.shape}")
            sys.stdout.flush()
            # Log the process ID for debugging
            import os
            # print(f"Process ID: {os.getpid()} at decoder call.")
            sys.stdout.flush()
            stack = Stack(grammar=GCFG, start_symbol=S)
            # print(f"Initial stack initialized with contents: {stack}")
            # print(f"Initial stack: {stack.contents()}")
            sys.stdout.flush()
            # print("Before decoder forward pass...")
            sys.stdout.flush()
            
            try:
                # print("Calling decoder...")
                sys.stdout.flush()
                # print(f"Devices: z on {z.device}, decoder on {self.device}")
                assert z.device == self.device, f"Mismatch: z on {z.device}, decoder on {self.device}"
                # print(f"decoder: {self.decoder}")
                sys.stdout.flush()
                logits = self.decoder(z).squeeze()
                # print("Decoder call successful.")
                sys.stdout.flush()
            except Exception as e:
                print(f"Decoder failed with exception: {e}")
                import traceback
                traceback.print_exc()  # Print full traceback for debugging
                sys.stdout.flush()
                raise

            # print(f"Logits shape: {logits.shape}")
            sys.stdout.flush()
            
            rules = []
            t = 0

            while stack.nonempty and t < self.max_length:
                # print(f"Step {t}, stack: {stack.contents()}")
                alpha = stack.pop()
                # print(f"Popped symbol: {alpha}")
                sys.stdout.flush()
                mask = get_mask(alpha, stack.grammar, as_variable=True).to(z.device)
                # print(f"Mask: {mask}")
                sys.stdout.flush()

                probs = mask * logits[t].exp()
                probs = probs / probs.sum()
                # print(f"Probabilities: {probs}")
                sys.stdout.flush()

                if sample:
                    m = Categorical(probs)
                    i = m.sample()
                else:
                    _, i = probs.max(-1)
                # print(f"Selected rule index: {i}")
                sys.stdout.flush()

                rule = stack.grammar.productions()[i.item()]
                rules.append(rule)
                # print(f"Selected rule: {rule}")
                sys.stdout.flush()

                for symbol in reversed(rule.rhs()):
                    if isinstance(symbol, Nonterminal):
                        stack.push(symbol)
                t += 1

            # print("Exiting generate.")
            sys.stdout.flush()
            return rules
        except Exception as e:
            print(f"Error gen: {e}")
            sys.stdout.flush()
            return None

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space
        """
        x = x.to(self.device)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to logits
        """
        return self.decoder(z)

    def to(self, device):
        """
        Ensure proper device movement and sanity‐check our internal pointer.
        """
        # Let the base class move all parameters/buffers first
        module = super().to(device)
        # Now update and verify our internal device attribute
        module.device = device

        # Sanity check: pick one parameter and assert it’s on the same device
        example_param = next(module.parameters())
        assert example_param.device == device, (
            f"Device mismatch: internal={device}, param={example_param.device}"
        )

        return module