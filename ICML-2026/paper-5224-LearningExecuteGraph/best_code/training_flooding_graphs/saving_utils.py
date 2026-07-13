import json
from pathlib import Path
from datetime import datetime
import numpy as np

def ensure_dir(path: str):
    """Create a directory if it doesn't exist."""
    Path(path).mkdir(exist_ok=True)

def get_timestamp():
    """Return a formatted timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def to_serializable(obj):
    """Recursively convert NumPy arrays and other non-serializable types to serializable forms."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj

def save_json(data, filepath: str):
    """Save dictionary data to a JSON file with indentation."""
    ensure_dir(str(Path(filepath).parent))
    with open(filepath, "w") as f:
        json.dump(to_serializable(data), f, indent=2)
        

def save_test_samples(samples, file_path, input_shape=None, output_shape=None):
    """Save test samples to file."""
    serializable = []
    for x_t, x_e, y_t in samples:
        serializable.append({
            'original_features': x_t.tolist() if hasattr(x_t, 'tolist') else x_t,
            'encoded_features': x_e.tolist() if hasattr(x_e, 'tolist') else x_e,
            'target': y_t.tolist() if hasattr(y_t, 'tolist') else y_t
        })

    filepath = f"test_samples/test_samples_{file_path}.json"
    save_json({
        'test_samples': serializable,
        'metadata': {
            'total_samples': len(samples),
            'input_shape': input_shape,
            'output_shape': output_shape
        }
    }, filepath)
    print(f"[AUTO-SAVE] Saved {len(samples)} test samples to {filepath}")

def save_problem_cases(problem_cases, file_path, stopped_early, t_idx, total_trees):
    """Save problematic cases to file."""
    filepath = f"failure_cases/problem_cases_{file_path}.json"
    save_json({
        'problem_cases': problem_cases,
        'metadata': {
            'total_cases': len(problem_cases),
            'stopped_early': stopped_early,
            'iterations_completed': t_idx + 1,
            'total_trees': total_trees
        }
    }, filepath)
    print(f"[AUTO-SAVE] Saved {len(problem_cases)} problem cases to {filepath}")


def save_test_cases(cases, file_path, n=None, D=None, l=None):
    """Save test samples to file."""
    serializable = []
    for X_t, X_e, Y_t in cases:
        serializable.append({
            'original_inputs': X_t.tolist() if hasattr(X_t, 'tolist') else X_t,
            'encoded_inputs': X_e.tolist() if hasattr(X_e, 'tolist') else X_e,
            'targets': Y_t.tolist() if hasattr(Y_t, 'tolist') else Y_t
        })

    filepath = f"test_samples/test_cases_{file_path}.json"
    save_json({
        'test_cases': serializable,
        'metadata': {
            'total_samples': len(cases),
            'n': n,
            'D': D,
            'l': l,
        }
    }, filepath)
    print(f"[AUTO-SAVE] Saved {len(cases)} test samples to {filepath}")