import os
# Environment variables must be set before importing any huggingface-related libraries
if not os.getenv('HF_ENDPOINT'):
    hf_mirror = os.getenv('HF_MIRROR', 'https://hf-mirror.com')
    os.environ['HF_ENDPOINT'] = hf_mirror
    os.environ['HF_HUB_ENDPOINT'] = hf_mirror
    print(f"🔧 Hugging Face mirror configured before import: {hf_mirror}")

from typing import List, Dict, Optional
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import copy
import requests
import time
import json

# sentence_transformers is lazily imported only inside Encoder.__init__ (to avoid unnecessary dependency loading)
# from sentence_transformers import SentenceTransformer


def _infer_epn_dims_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> List[int]:
    """
    Infer EPN dims from state_dict weights without relying on saved hyperparameters.
    """
    dims: List[int] = []
    mlp_weights: List[tuple] = []
    for key, tensor in state_dict.items():
        if key.startswith("mlp.") and key.endswith(".weight"):
            # key format: mlp.{idx}.weight
            try:
                layer_idx = int(key.split(".")[1])
            except (IndexError, ValueError):
                continue
            mlp_weights.append((layer_idx, tensor))

    mlp_weights.sort(key=lambda x: x[0])
    if mlp_weights:
        dims.append(int(mlp_weights[0][1].shape[1]))
        for _, w in mlp_weights:
            dims.append(int(w.shape[0]))

    output_w = state_dict.get("output_layer.weight")
    if output_w is not None:
        if not dims:
            dims.append(int(output_w.shape[1]))
        out_dim = int(output_w.shape[0])
        if dims[-1] != out_dim:
            dims.append(out_dim)

    if len(dims) < 2:
        raise ValueError("Cannot infer valid EPN dims from state_dict.")
    return dims

class EPN(nn.Module):
    """
    Edge Prediction Network - head network of the Critic.
    Extends a standard multilayer perceptron with a lightweight residual block to improve expressiveness
    without significantly increasing the number of parameters.
    """

    def __init__(self, dims: List[int], dropout: float = 0.0, temperature: float = 5.0):
        """
        Args:
            dims: List of layer dimensions.
            dropout: Optional dropout ratio, default 0 (for small-data scenarios).
            temperature: Tanh scaling factor controlling the output range differences, default 5.0.
        """
        super().__init__()
        self.temperature = temperature
        hidden_dims = dims[:-1]
        self.output_layer = nn.Linear(hidden_dims[-1], dims[-1])
        # Use standard initialization and let PyTorch choose suitable defaults.
        # Avoid custom init here to reduce instability and constant-like outputs.

        mlp_layers: List[nn.Module] = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            mlp_layers.append(nn.ReLU())
            if dropout > 0:
                mlp_layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()

        # Lightweight residual block: Linear -> GELU -> Linear, then add back to the input
        last_hidden_dim = hidden_dims[-1]
        self.residual_block = nn.Sequential(
            nn.Linear(last_hidden_dim, last_hidden_dim),
            nn.GELU(),
            nn.Linear(last_hidden_dim, last_hidden_dim),
        )

        if dropout > 0:
            self.residual_dropout = nn.Dropout(dropout)
        else:
            self.residual_dropout = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, dims[0]].
        Returns:
            Output tensor of shape [batch_size, 1].
        """
        hidden = self.mlp(x)
        if hidden.shape[-1] != self.residual_block[0].in_features:
            raise ValueError(
                "EPN residual block input dimension does not match hidden layer output; "
                "please double-check the dims configuration."
            )
        residual = self.residual_block(hidden)
        residual = self.residual_dropout(residual)
        hidden = hidden + residual
        # Output layer uses linear output (no tanh saturation); value range/sign are learned from loss/data.
        # This avoids over-compressing outputs on small datasets and improves negative-value discrimination.
        output = self.output_layer(hidden)
        return output


def test_huggingface_connection(model_name="all-MiniLM-L6-v2"):
    """
    Test Hugging Face connectivity and configure mirror endpoints.

    Returns:
        bool: Whether connectivity test succeeded.
    """
    print("🔍 Testing Hugging Face connectivity...")
    
    # Ensure environment variables are set
    if not os.getenv('HF_ENDPOINT'):
        hf_mirror = os.getenv('HF_MIRROR', 'https://hf-mirror.com')
        os.environ['HF_ENDPOINT'] = hf_mirror
        os.environ['HF_HUB_ENDPOINT'] = hf_mirror
        print(f"   ✅ Mirror endpoint set: {hf_mirror}")
    else:
        print(f"   ✅ Using configured mirror: {os.getenv('HF_ENDPOINT')}")
    
    # Try configuring huggingface_hub directly (when available)
    try:
        import huggingface_hub
        # Force the hub endpoint
        current_endpoint = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')
        huggingface_hub.constants.ENDPOINT = current_endpoint
        print(f"   ✅ huggingface_hub endpoint set to: {current_endpoint}")
    except (ImportError, AttributeError):
        pass
    
    # Test connectivity to mirror and official endpoints
    test_urls = [
        ("https://hf-mirror.com", "mirror"),
        ("https://huggingface.co", "official"),
    ]
    
    accessible_url = None
    for url, desc in test_urls:
        try:
            print(f"   Testing connectivity to {desc}: {url}...", end=" ")
            response = requests.get(f"{url}/api/models", timeout=5)
            if response.status_code == 200:
                print("✅ reachable")
                accessible_url = url
                break
        except requests.exceptions.SSLError as e:
            print(f"❌ SSL error: {str(e)[:50]}")
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Connection error: {str(e)[:50]}")
        except requests.exceptions.Timeout:
            print("❌ Timeout")
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
    
    if accessible_url:
        print(f"   ✅ Found reachable endpoint: {accessible_url}")
        if accessible_url != os.getenv('HF_ENDPOINT'):
            os.environ['HF_ENDPOINT'] = accessible_url
            os.environ['HF_HUB_ENDPOINT'] = accessible_url
            # Reconfigure huggingface_hub for the new endpoint
            try:
                import huggingface_hub
                huggingface_hub.constants.ENDPOINT = accessible_url
            except (ImportError, AttributeError):
                pass
            print(f"   🔧 Updated mirror configuration to: {accessible_url}")
        return True
    else:
        print("   ⚠️  All test endpoints are unreachable; will rely on cache or keep trying...")
        return False


class Encoder:
    """Text encoder using a pretrained Sentence Transformer."""
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Lazily import sentence_transformers to avoid unnecessary imports and hub version issues
        from sentence_transformers import SentenceTransformer
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use locally cached model when available to avoid HuggingFace connectivity issues
        _local_model_dir = "/autosota_cache/hf/models/all-MiniLM-L6-v2"
        _model_to_load = _local_model_dir if os.path.isdir(_local_model_dir) else model_name
        
        print(f"📥 Start loading model from: {_model_to_load}")
        try:
            self.model = SentenceTransformer(_model_to_load, device=self.device)
            print(f"✅ Model loaded successfully")
        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ Model load failed")
            print(f"   Error details: {error_msg[:300]}")
            raise

    def run(self, text: str):
        """Encode a single piece of text and return its embedding."""
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            device=self.device,
            normalize_embeddings=True
        )
        return embedding

class Critics:
    """
    Critics - intrinsic edge-value predictor.
    
    Core functionalities:
    1. Predict the intrinsic value of an edge (its contribution to the correctness of the final answer).
    2. Support a self-locking mechanism (auto-lock when Critic loss falls below a threshold).
    3. Once locked, it can work independently to save training cost.
    
    Input structure (revised version):
        - node1_info: Source-node information.
        - node2_info: Target-node information.
        - question: Original question text.
        
    Note: edge_logit is NOT used as input!
        - edge_logit is the Actor's decision parameter (how much the Actor wants to select this edge).
        - The Critic should evaluate the *intrinsic value* of the edge (how helpful this edge is by itself).
        - edge_logit should instead be used in the Actor's loss function (policy gradient).
    
    Output:
        - Intrinsic edge-value prediction in [0, 1], representing the edge's contribution.
    """
    
    # Default save/load directory
    DEFAULT_SAVE_DIR = "Pretrained_Critics_Data"
    
    def __init__(self, 
                 epn_dims: List[int], 
                 model_name: str = "all-MiniLM-L6-v2", 
                 lock_threshold: float = 0.01,
                 temperature: float = 5.0,
                 dropout: float = 0.0):
        """
        Args:
            epn_dims: EPN network dimensions.
            model_name: Encoder model name.
            lock_threshold: Lock threshold (auto-lock when Critic loss drops below this; recommend 0.01 ~ 0.05).
            temperature: tanh scaling factor controlling output range (larger → greater separation between predictions).
            dropout: Dropout ratio for EPN network (default 0.0, use 0.1-0.3 to reduce overfitting).
        """
        # Initialize Encoder and EPN
        self.encoder = Encoder(model_name)
        self.device = self.encoder.device
        self.epn = EPN(epn_dims, dropout=dropout, temperature=temperature).to(self.device)
        
        # Save encoder_model_name for later save/load
        self._encoder_model_name = model_name

        # Freeze Encoder; train only EPN
        for param in self.encoder.model.parameters():
            param.requires_grad = False
        for param in self.epn.parameters():
            param.requires_grad = True
        
        # Self-locking related attributes
        self.lock_threshold = lock_threshold
        self.is_locked = False
        self.lock_confidence = 0.0
        
        # Independent parameter copies after locking
        self.locked_encoder = None
        self.locked_epn = None

    def run_differentiated(self,
                          in_node_description: str,
                          in_node_history: str,
                          query: str,
                          out_node_description: str,
                          out_node_history: str,
                          use_locked: bool = False) -> torch.Tensor:
        """
        Predict the intrinsic value of a single edge using differentiated 5-part inputs.
        
        Args:
            in_node_description: Description of the input node (mainly in_node_role).
            in_node_history: History text of the input node.
            query: Query / question text.
            out_node_description: Description of the output node.
            out_node_history: History text of the output node.
            
        Returns:
            Edge intrinsic value prediction tensor [1].
        """
        # Encode five inputs
        text_inputs = [in_node_description, in_node_history, query, out_node_description, out_node_history]
        
        # Automatically choose which model to use
        if use_locked or (self.is_locked and not use_locked):
            if self.locked_encoder is None or self.locked_epn is None:
                raise RuntimeError("Attempted to use locked model, but it is not locked yet")
            encoder = self.locked_encoder
            epn = self.locked_epn
        else:
            encoder = self.encoder
            epn = self.epn
        
        encoded_tensors = []
        for text in text_inputs:
            vec = encoder.model.encode(
                text,
                convert_to_tensor=True,
                device=self.device,
                normalize_embeddings=True
            )
            encoded_tensors.append(vec)
        
        # Concatenate into shape [1, 5*384]
        concatenated_vector = torch.cat(encoded_tensors, dim=-1)
        
        # Forward pass through EPN
        if encoder == self.encoder and epn == self.epn:
            concatenated_vector = concatenated_vector.detach()
            output = epn(concatenated_vector)
        else:
            with torch.no_grad():
                output = epn(concatenated_vector)
        
        return output  # [1]

    def run_batch(self, 
                  node1_list: List[str], 
                  node2_list: List[str], 
                  question_list: List[str],
                  use_locked: bool = False) -> torch.Tensor:
        """
        Batch prediction of intrinsic edge values (unified interface, auto-selects model, backward compatible).

        Args:
            node1_list: List of source node texts.
            node2_list: List of target node texts.
            question_list: List of question texts.
            use_locked: Whether to force using the locked model (default False = auto).

        Returns:
            Tensor of edge-value predictions [N, 1].
        """
        N = len(node1_list)
        assert len(node2_list) == N and len(question_list) == N, \
            "Batch inputs must have same length"

        # Automatically choose which model to use
        # Priority: use_locked arg > current locked status > original model
        if use_locked or (self.is_locked and not use_locked):
            if self.locked_encoder is None or self.locked_epn is None:
                raise RuntimeError("Attempted to use locked model, but it is not locked yet")
            encoder = self.locked_encoder
            epn = self.locked_epn
        else:
            encoder = self.encoder
            epn = self.epn

        # Batch-encode inputs (excluding edge_logit)
        with torch.no_grad():
            n1 = encoder.model.encode(node1_list, convert_to_tensor=True, 
                                      device=self.device, normalize_embeddings=True)
            n2 = encoder.model.encode(node2_list, convert_to_tensor=True, 
                                      device=self.device, normalize_embeddings=True)
            questions = encoder.model.encode(question_list, convert_to_tensor=True, 
                                             device=self.device, normalize_embeddings=True)

        # Concatenate into shape [N, 3*384]
        concatenated = torch.cat([n1, n2, questions], dim=-1)
        
        # If using the original model (training mode), keep gradient flow
        if encoder == self.encoder and epn == self.epn:
            concatenated = concatenated.detach()  # No gradients for Encoder; EPN remains trainable
            output = epn(concatenated)
        else:
            # Use locked model with no gradients
            with torch.no_grad():
                output = epn(concatenated)
        
        return output  # [N, 1]
    
    def run_batch_differentiated(self,
                                in_node_description_list: List[str],
                                in_node_history_list: List[str],
                                query_list: List[str],
                                out_node_description_list: List[str],
                                out_node_history_list: List[str],
                                use_locked: bool = False) -> torch.Tensor:
        """
        Batch prediction of intrinsic edge values (differentiated 5-part inputs).

        Args:
            in_node_description_list: List of input-node descriptions (mainly in_node_role).
            in_node_history_list: List of input-node histories.
            query_list: List of queries/questions.
            out_node_description_list: List of output-node descriptions.
            out_node_history_list: List of output-node histories.

        Returns:
            Tensor of edge-value predictions [N, 1].
        """
        N = len(in_node_description_list)
        assert (len(in_node_history_list) == N and len(query_list) == N and
                len(out_node_description_list) == N and len(out_node_history_list) == N), \
            "Batch inputs must have same length"

        # Automatically choose which model to use
        if use_locked or (self.is_locked and not use_locked):
            if self.locked_encoder is None or self.locked_epn is None:
                raise RuntimeError("Attempted to use locked model, but it is not locked yet")
            encoder = self.locked_encoder
            epn = self.locked_epn
        else:
            encoder = self.encoder
            epn = self.epn

        # Batch-encode the five inputs
        with torch.no_grad():
            in_desc = encoder.model.encode(in_node_description_list, convert_to_tensor=True,
                                           device=self.device, normalize_embeddings=True)
            in_history = encoder.model.encode(in_node_history_list, convert_to_tensor=True,
                                             device=self.device, normalize_embeddings=True)
            queries = encoder.model.encode(query_list, convert_to_tensor=True,
                                             device=self.device, normalize_embeddings=True)
            out_desc = encoder.model.encode(out_node_description_list, convert_to_tensor=True,
                                           device=self.device, normalize_embeddings=True)
            out_history = encoder.model.encode(out_node_history_list, convert_to_tensor=True,
                                             device=self.device, normalize_embeddings=True)

        # Concatenate into shape [N, 5*384]
        concatenated = torch.cat([in_desc, in_history, queries, out_desc, out_history], dim=-1)
        
        # If using the original model (training mode), keep gradient flow
        if encoder == self.encoder and epn == self.epn:
            concatenated = concatenated.detach()  # No gradients for Encoder; EPN remains trainable
            output = epn(concatenated)
        else:
            # Use locked model with no gradients
            with torch.no_grad():
                output = epn(concatenated)
        
        return output  # [N, 1]
    
    # ==================== Self-locking related methods ====================
    
    def should_lock(self, recent_loss: float) -> bool:
        """
        Decide whether the Critic should be locked (based on recent loss).

        Args:
            recent_loss: Most recent Critic loss value.

        Returns:
            bool: True if the model should be locked.
        """
        if recent_loss <= self.lock_threshold and not self.is_locked:
            return True
        return False
    
    def lock_critic(self) -> None:
        """
        Lock the Critic.
        Create independent parameter copies for subsequent simulated training.
        """
        if self.is_locked:
            return
        # Deep-copy Encoder and EPN
        self.locked_encoder = copy.deepcopy(self.encoder)
        self.locked_epn = copy.deepcopy(self.epn)
        
        # Freeze parameters of the locked copies
        for param in self.locked_encoder.model.parameters():
            param.requires_grad = False
        for param in self.locked_epn.parameters():
            param.requires_grad = False

        # Freeze trainable EPN so the optimizer cannot update it after lock; forward uses locked_epn when is_locked.
        for param in self.epn.parameters():
            param.requires_grad = False
            
        self.is_locked = True
        self.lock_confidence = 1.0
    
    def predict_with_locked_model(self, 
                                   node1_list: List[str],
                                   node2_list: List[str],
                                   question_list: List[str]) -> torch.Tensor:
        """
        Predict with the locked model (a convenience wrapper for run_batch(use_locked=True)).
        
        This is a semantic wrapper that internally calls run_batch(use_locked=True)
        so that calling code can clearly express the intention to use the locked model.
        
        Args:
            node1_list: List of source-node information strings.
            node2_list: List of target-node information strings.
            question_list: List of question strings.
            
        Returns:
            Tensor of intrinsic edge-value predictions [N, 1].
        """
        return self.run_batch(
            node1_list=node1_list,
            node2_list=node2_list,
            question_list=question_list,
            use_locked=True,  # Force using the locked model
        )
    
    def unlock_critic(self) -> None:
        """
        Unlock the Critic so it can receive new training samples again.
        """
        self.is_locked = False
        self.lock_confidence = 0.0
        for param in self.epn.parameters():
            param.requires_grad = True
        # Keep locked copies so they can still be reused later if needed
    
    def reset_lock(self) -> None:
        """Completely reset lock state (including deleting locked copies)."""
        self.is_locked = False
        self.lock_confidence = 0.0
        self.locked_encoder = None
        self.locked_epn = None
        for param in self.epn.parameters():
            param.requires_grad = True
    
    def get_lock_status(self) -> Dict[str, any]:
        """
        Get lock status information.

        Returns:
            Dict: Dictionary describing current lock status.
        """
        return {
            'is_locked': self.is_locked,
            'lock_confidence': self.lock_confidence,
            'lock_threshold': self.lock_threshold,
            'has_locked_encoder': self.locked_encoder is not None,
            'has_locked_epn': self.locked_epn is not None
        }
    
    # ==================== Training-related methods ====================
    
    def compute_loss(self, 
                     predictions: torch.Tensor, 
                     targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss function.

        Args:
            predictions: Predicted values [N, 1].
            targets: Ground-truth labels [N, 1] (0 or 1, indicating whether the graph with this edge is correct).

        Returns:
            Loss tensor.
        """
        # Use Binary Cross Entropy Loss
        loss = F.binary_cross_entropy(predictions, targets)
        return loss
    
    def evaluate_accuracy(self, 
                         predictions: torch.Tensor, 
                         targets: torch.Tensor,
                         threshold: float = 0.5) -> float:
        """
        Evaluate prediction accuracy.
        
        Args:
            predictions: Predicted values [N, 1].
            targets: Ground-truth labels [N, 1].
            threshold: Classification threshold.
            
        Returns:
            Accuracy in [0, 1].
        """
        pred_labels = (predictions > threshold).float()
        accuracy = (pred_labels == targets).float().mean().item()
        return accuracy
    
    # ==================== Save and load interfaces ====================
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the Critic model to a file.

        Args:
            filepath: Path to save the model (suggest .pt or .pth extension).
                      If None, uses default path Pretrained_Critics_Data/critics_model.pt.
                      If a directory is given, it will be created.

        Returns:
            The actual file path where the model was saved.
        """
        import os
        
        # If no path is specified, use the default
        if filepath is None:
            os.makedirs(self.DEFAULT_SAVE_DIR, exist_ok=True)
            filepath = f"{self.DEFAULT_SAVE_DIR}/critics_model.pt"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        print(f"💾 Saving Critic model to: {filepath}")
        
        # Prepare data to be saved
        save_dict = {
            # EPN model parameters
            'epn_state_dict': self.epn.state_dict(),
            
            # Configuration info
            'encoder_model_name': getattr(self, '_encoder_model_name', 'all-MiniLM-L6-v2'),
            
            # Lock-related state
            'is_locked': self.is_locked,
            'lock_confidence': self.lock_confidence,
            
            # Parameters of the locked model (if present)
            'locked_epn_state_dict': None,
        }
        
        # Save locked EPN parameters if they exist
        if self.locked_epn is not None:
            save_dict['locked_epn_state_dict'] = self.locked_epn.state_dict()
        
        # Save to file
        torch.save(save_dict, filepath)
        print(f"✅ Critic model saved to: {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: Optional[str] = None, encoder_model_name: Optional[str] = None):
        """
        Load a Critic model from file.

        Args:
            filepath: Model file path (if None, use default Pretrained_Critics_Data/critics_model.pt).
            encoder_model_name: Encoder model name (used if not provided in the saved file).
                               Default: "all-MiniLM-L6-v2".

        Returns:
            Loaded Critics instance.
        """
        import os
        
        # If no path is specified, use the default
        if filepath is None:
            default_path = f"{cls.DEFAULT_SAVE_DIR}/critics_model.pt"
            if not os.path.exists(default_path):
                raise FileNotFoundError(
                    f"Default path does not exist: {default_path}; "
                    f"please specify filepath or save a model first."
                )
            filepath = default_path
        
        print(f"📥 Loading Critic model from file: {filepath}")
        
        # Load the saved data
        save_dict = torch.load(filepath, map_location='cpu')
        
        # Infer EPN dims from state dict to avoid relying on saved architecture hyperparameters
        epn_state = save_dict.get('epn_state_dict')
        if epn_state is None:
            raise KeyError("Missing 'epn_state_dict' in checkpoint; cannot load Critics.")
        epn_dims = _infer_epn_dims_from_state_dict(epn_state)
        encoder_model_name = save_dict.get('encoder_model_name', encoder_model_name) or 'all-MiniLM-L6-v2'
        lock_threshold = save_dict.get('lock_threshold', 0.01)
        
        # Create a new Critics instance
        critics = cls(
            epn_dims=epn_dims,
            model_name=encoder_model_name,
            lock_threshold=lock_threshold
        )
        
        # Save encoder_model_name for later use
        critics._encoder_model_name = encoder_model_name
        
        # Load EPN parameters
        critics.epn.load_state_dict(epn_state)
        critics.epn.to(critics.device)
        
        # Restore locked status
        critics.is_locked = save_dict.get('is_locked', False)
        critics.lock_confidence = save_dict.get('lock_confidence', 0.0)
        
        # If locked-model parameters exist, load them
        if 'locked_epn_state_dict' in save_dict and save_dict['locked_epn_state_dict'] is not None:
            locked_epn_state = save_dict['locked_epn_state_dict']
            locked_epn_dims = _infer_epn_dims_from_state_dict(locked_epn_state)
            # Create a locked EPN copy
            critics.locked_epn = EPN(locked_epn_dims).to(critics.device)
            critics.locked_epn.load_state_dict(locked_epn_state)
            
            # Create a locked Encoder copy (deep copy)
            critics.locked_encoder = copy.deepcopy(critics.encoder)
            
            # Freeze parameters of the locked model
            for param in critics.locked_encoder.model.parameters():
                param.requires_grad = False
            for param in critics.locked_epn.parameters():
                param.requires_grad = False
        
        print(f"✅ Critic model loaded from: {filepath}")
        return critics
    
    def upload(self, filepath: str, remote_path: Optional[str] = None) -> str:
        """
        Upload Critic model to remote storage.

        Args:
            filepath: Local model file path (will be saved first if missing).
            remote_path: Remote path (optional; defaults to filepath).

        Returns:
            Remote path string.
        """
        import os
        
        # If local file is missing, save it first
        if not os.path.exists(filepath):
            self.save(filepath)
        
        # This can be extended to upload to cloud storage (e.g., HuggingFace Hub, S3, etc.)
        # Currently we just return the local path
        print(f"📤 Model is ready for upload: {filepath}")
        print("   Hint: can be extended to upload to HuggingFace Hub or other cloud storage")
        
        return filepath
    
    @classmethod
    def download(cls, remote_path: str, local_path: Optional[str] = None, encoder_model_name: Optional[str] = None):
        """
        Download a Critic model from remote storage.

        Args:
            remote_path: Remote model path.
            local_path: Local path to save to (optional; defaults to remote_path).
            encoder_model_name: Encoder model name (optional).

        Returns:
            Downloaded-and-loaded Critics instance.
        """
        import os
        
        # If no local path is given, use the default directory
        if local_path is None:
            os.makedirs(cls.DEFAULT_SAVE_DIR, exist_ok=True)
            local_path = f"{cls.DEFAULT_SAVE_DIR}/{os.path.basename(remote_path)}"
        
        # This can be extended to download from cloud storage (e.g., HuggingFace Hub, S3, etc.)
        # Currently only local-path loading is supported
        print(f"📥 Downloading model from remote path: {remote_path}")
        print(f"   Local save path: {local_path}")
        print(f"   Note: extend this to download from HuggingFace Hub or other storage")
        
        # Normalize paths so they can be compared correctly
        remote_path_abs = os.path.abspath(remote_path)
        local_path_abs = os.path.abspath(local_path)
        
        # If the remote path already refers to a local file, copy it
        if os.path.exists(remote_path):
            import shutil
            # Check whether the two paths refer to the same file
            try:
                if os.path.samefile(remote_path_abs, local_path_abs):
                    print(f"ℹ️  Source and destination are the same; skipping copy")
                else:
                    shutil.copy2(remote_path, local_path)
                    print(f"✅ File copied to: {local_path}")
            except OSError:
                # If samefile fails (file may not exist), attempt to copy
                shutil.copy2(remote_path, local_path)
                print(f"✅ File copied to: {local_path}")
        else:
            # If the file does not exist, try to load it (might be a remote path; needs extension)
            if not os.path.exists(local_path):
                raise FileNotFoundError(
                    f"Model file not found: remote_path={remote_path}, local_path={local_path}"
                )
        
        return cls.load(local_path, encoder_model_name=encoder_model_name)
