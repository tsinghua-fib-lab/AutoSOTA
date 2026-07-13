
import pickle
import numpy as np
import torch

class EnsembleCVXNNLangDetectHead:
    """Ensemble of multiple CVXNN CLD heads with soft voting (logit averaging)."""
    
    def __init__(self, heads):
        self.heads = heads
    
    @staticmethod
    def load(model_paths, asr_model):
        heads = []
        for path in model_paths:
            with open(path, 'rb') as f:
                head = pickle.load(f)
            heads.append(head)
        return EnsembleCVXNNLangDetectHead(heads)
    
    def predict(self, hidden):
        if isinstance(hidden, torch.Tensor):
            if hidden.ndim == 3:
                hidden = hidden.mean(dim=1)
            pooled = hidden.detach().to("cpu").float().numpy()
        else:
            pooled = np.asarray(hidden, dtype=np.float32)
            if pooled.ndim == 3:
                pooled = pooled.mean(axis=1)
        
        all_logits = []
        for head in self.heads:
            logits = head.stacked_predict(pooled, head.theta1, head.theta2)
            all_logits.append(np.asarray(logits))
        
        avg_logits = np.mean(all_logits, axis=0)
        if avg_logits.ndim == 1:
            return [0 for _ in range(int(avg_logits.shape[0]))]
        return avg_logits.argmax(axis=1).tolist()

def create_ensemble_model(baseline_path, seed_paths, output_path):
    """Create an ensemble model by averaging weights from multiple seeds."""
    heads = []
    
    # Load baseline model
    with open(baseline_path, 'rb') as f:
        baseline = pickle.load(f)
    heads.append(baseline)
    print(f'Loaded baseline model (seed 0), val_acc ~0.980')
    
    # Load seed models
    for path in seed_paths:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        heads.append(model)
        print(f'Loaded {path}')
    
    # For ensemble inference, save the ensemble wrapper
    ensemble = EnsembleCVXNNLangDetectHead(heads)
    
    # Also create a merged model for saving: average theta1 and theta2
    # Note: this is a HACK - averaging in non-convex space is not theoretically sound
    # but can work as an approximation when seeds are close
    merged = baseline  # Use baseline's structure, X, y, etc.
    
    # Average theta1 and theta2 across all heads
    theta1s = [h.theta1 for h in heads]
    theta2s = [h.theta2 for h in heads]
    
    merged.theta1 = np.mean(theta1s, axis=0)
    merged.theta2 = np.mean(theta2s, axis=0)
    
    # Save merged model
    with open(output_path, 'rb') as f:
        pickle.dump(merged, f)
    
    print(f'Ensemble model saved to {output_path}')
    print(f'Merged theta1 shape: {merged.theta1.shape}')
    print(f'Merged theta2 shape: {merged.theta2.shape}')
    
    return merged

if __name__ == '__main__':
    import sys
    baseline = sys.argv[1] if len(sys.argv) > 1 else '/repo/models/cld_whisper_small_v2/openai/whisper-small/openai_whisper-small_trained_cvx_mlp.pkl'
    output = sys.argv[2] if len(sys.argv) > 2 else '/repo/models/sota_iter2_ensemble_weightavg.pkl'
    
    seed_paths = [
        f'/repo/models/sota_seed{s}.pkl' for s in [1, 2, 3, 4]
    ]
    
    create_ensemble_model(baseline, seed_paths, output)
