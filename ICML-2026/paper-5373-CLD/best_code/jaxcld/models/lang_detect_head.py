import os, time, torch, types, pickle
import torch.nn as nn
import numpy as np
try:
    from safetensors.torch import load_file as _safetensors_load_file
except Exception:  # optional dependency
    _safetensors_load_file = None
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from abc import ABC, abstractmethod

dtype = torch.float16

class LangDetectHead(ABC):
    def __init__(self, head):
        self.head = head

    @abstractmethod
    def load(filepath, asr_model):
        """Load detection head from filepath."""
        pass

    @abstractmethod
    def predict(self, hidden):
        """Runs classification on the hidden layer"""
        pass

class NNLangDetectHeadModule(nn.Module):
    def __init__(self, d_model: int, n_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes),
        )
    
    def forward(self, hidden_states):
        pooled = hidden_states.mean(dim=1)  # (B, D)
        return self.classifier(pooled)      # (B, 2)
    

class NNLangDetectHead(LangDetectHead):
    @staticmethod
    def load(filepath, asr_model):
        state = None
        if str(filepath).endswith((".pkl", ".pickle")):
            with open(filepath, "rb") as f:
                state = pickle.load(f)
        else:
            if _safetensors_load_file is None:
                raise ImportError(
                    "safetensors is not available, so only .pkl/.pickle NN heads can be loaded in this environment."
                )
            state = _safetensors_load_file(filepath)

        # Normalize state entries to torch tensors
        classifier_state = {}
        for k, v in state.items():
            if not k.startswith("classifier."):
                continue
            kk = k.replace("classifier.", "")
            if isinstance(v, torch.Tensor):
                classifier_state[kk] = v
            else:
                classifier_state[kk] = torch.as_tensor(v)
        
        n_classes = 2
        languages = getattr(asr_model, "config", {}).get("languages")
        if isinstance(languages, (list, tuple)) and len(languages) > 0:
            n_classes = len(languages)

        head = NNLangDetectHeadModule(asr_model.get_dimensions(), n_classes=n_classes)
        head.classifier.load_state_dict(classifier_state)
        
        # Move head to match encoder's last layer device
        head = head.to(asr_model.get_device(), dtype=dtype)
        return NNLangDetectHead(head)
    
    def predict(self, hidden):
        logits = self.head(hidden)
        return logits.argmax(dim=-1).tolist()
    
class CVXNNLangDetectHead(LangDetectHead):
    @staticmethod
    def load(filepath, asr_model):
        with open(filepath, 'rb') as f:
            head = pickle.load(f)
        
        return CVXNNLangDetectHead(head)
    
    def predict(self, hidden):
        # `hidden` may be a raw encoder tensor (B, T, D) from Whisper, or an
        # already-pooled numpy array (B, D) from the MMS callsite in asr_model.
        if isinstance(hidden, torch.Tensor):
            if hidden.ndim == 3:
                if getattr(self.head, 'pooling_mode', None) == 'statistics':
                    mean_pooled = hidden.mean(dim=1)
                    std_pooled = hidden.std(dim=1)
                    hidden = torch.cat([mean_pooled, std_pooled], dim=1)
                else:
                    hidden = hidden.mean(dim=1)
            pooled = hidden.detach().to("cpu").float().numpy()
        else:
            pooled = np.asarray(hidden, dtype=np.float32)
            if pooled.ndim == 3:
                pooled = pooled.mean(axis=1)
        # theta1/theta2 stored by ADMM are batched per-class weights:
        # - theta1: (C, d, m)
        # - theta2: (C, m)
        # so we must use stacked_predict to obtain (B, C) logits.
        logits = self.head.stacked_predict(pooled, self.head.theta1, self.head.theta2)
        logits = np.asarray(logits)
        if logits.ndim == 1:
            # Degenerate case (C==1): return all zeros
            return [0 for _ in range(int(logits.shape[0]))]
        return logits.argmax(axis=1).tolist()


class SklearnLangDetectHead(LangDetectHead):
    """
    sklearn-based language detection head.

    Designed to work with both:
    - Whisper: `hidden` is a torch.Tensor of shape (B, T, D)
    - MMS: `hidden` may be a pooled numpy array of shape (B, D)

    The saved artifact is expected to be a pickled sklearn estimator/pipeline
    exposing `.predict(X)` where X is a numpy array shaped (B, D).
    """

    @staticmethod
    def load(filepath, asr_model):
        if not str(filepath).endswith((".pkl", ".pickle")):
            raise ValueError("SklearnLangDetectHead only supports loading from .pkl/.pickle artifacts")
        with open(filepath, "rb") as f:
            head = pickle.load(f)
        if not hasattr(head, "predict"):
            raise ValueError("Loaded sklearn head does not have a .predict method")
        return SklearnLangDetectHead(head)

    def _to_pooled_numpy(self, hidden):
        # Case 1: torch.Tensor from Whisper encoder hidden states (B, T, D) or pooled (B, D)
        if isinstance(hidden, torch.Tensor):
            if hidden.ndim == 3:
                pooled = hidden.mean(dim=1)
            elif hidden.ndim == 2:
                pooled = hidden
            else:
                raise ValueError(f"Unexpected torch.Tensor hidden shape: {tuple(hidden.shape)}")
            return pooled.detach().to("cpu").float().numpy()

        # Case 2: numpy array already pooled (B, D)
        arr = np.asarray(hidden)
        if arr.ndim != 2:
            raise ValueError(f"Expected pooled features (B, D); got shape {arr.shape}")
        return arr.astype(np.float32, copy=False)

    def predict(self, hidden):
        X = self._to_pooled_numpy(hidden)
        pred = self.head.predict(X)
        pred = np.asarray(pred)
        # Ensure we always return a list[int] for compatibility with Whisper + MMS callsites.
        if pred.ndim != 1:
            pred = pred.reshape(-1)
        return pred.astype(int).tolist()


class SVMLangDetectHead(SklearnLangDetectHead):
    """Backward-compatible alias for sklearn-based SVM heads."""

    @staticmethod
    def load(filepath, asr_model):
        head = SklearnLangDetectHead.load(filepath, asr_model)
        return SVMLangDetectHead(head.head)

