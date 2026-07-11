"""
Consolidated utilities for mathematical expression processing, model operations, 
file management, and numerical computations.
"""

import re
import os
import pickle
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, FrozenSet
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import yaml
import symengine as se
import sympy as sp
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans

# Import your existing modules
from sigs.grammar import GCFG, S, get_mask
from sigs.stack import Stack
from sigs.model import GrammarVAE
from sigs.training import GrammarVAEModel


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class MathClass(Enum):
    SPATIOTEMPORAL_3D = auto()
    SPATIOTEMPORAL_2D = auto()
    SPATIAL_2D        = auto()
    SPATIAL_1D        = auto()
    TEMPORAL_1D       = auto()
    CONSTANT          = auto()
    OTHER             = auto()
    ERROR             = auto()


@dataclass
class ExpressionFlags:
    has_x: bool = False
    has_t: bool = False
    has_y: bool = False
    parse_error: bool = False
    math_class: MathClass = MathClass.ERROR

    def as_dict(self):
        return {
            'has_x': self.has_x,
            'has_t': self.has_t,
            'has_y': self.has_y,
            'parse_error': self.parse_error,
            'math_class': self.math_class
        }


# =============================================================================
# EXPRESSION UTILITIES
# =============================================================================

class ExpressionUtils:
    """Utilities for expression processing, validation, and parsing."""
    
    @staticmethod
    def validate_expression(expr: str) -> bool:
        """Check if expression contains invalid characters (S, T, or D)."""
        if not expr or not isinstance(expr, str):
            return False
        invalid_chars = ['S', 'T', 'D']
        return not any(c in expr for c in invalid_chars)
    
    
    
    @staticmethod
    def standardize_expression(expr: str) -> str:
        """Standardize expression format for evaluation."""
        if not expr or not isinstance(expr, str):
            return expr
        
        # Common replacements
        expr = expr.replace('^', '**')
        expr = expr.replace('-x/x', '-1')
        expr = expr.replace('pi2', 'pi**2')
        
        return expr
    
    @staticmethod
    def construct_expression(logits, max_length, device, seed=None):
        """Construct expression from VAE logits using grammar rules."""
        if seed is not None:
            random.seed(seed)
        
        # Ensure logits are on the correct device
        if hasattr(logits, 'to'):
            logits = logits.to(device)
        
        stack = Stack(grammar=GCFG, start_symbol=S)
        rules, t = [], 0

        while stack.nonempty and t < max_length:
            alpha = stack.pop()
            mask = get_mask(alpha, stack.grammar, as_variable=True).to(device)
            step = logits[0, t] if logits.ndim == 3 else logits[t]
            # Ensure step is on the same device as mask
            if hasattr(step, 'to'):
                step = step.to(device)
            probs = (mask * step.exp()).clamp(min=1e-9)
            probs = probs / probs.sum()
            idx = int(probs.argmax())
            rule = stack.grammar.productions()[idx]
            rules.append(rule)
            for sym in reversed(rule.rhs()):
                if isinstance(sym, type(S)):
                    stack.push(sym)
            t += 1

        expr = "S"
        for rule in rules:
            expr = expr.replace(
                rule.lhs().symbol(),
                " ".join(str(r) for r in rule.rhs()),
                1
            )
        return re.sub(r'\s+', '', expr)
    

    @staticmethod
    @lru_cache(maxsize=1024)
    def classify_expression(symbols: FrozenSet[str]) -> MathClass:
        """Classify expression based on variables present."""
        # 1) Lowercase everything, so 'S'/'T' → 's'/'t'
        raw = {s.lower() for s in symbols}

        # 2) Keep only the real variables we care about
        syms = raw & {'x', 'y', 't'}

        has_x = 'x' in syms
        has_y = 'y' in syms
        has_t = 't' in syms

        # Now "other" is always False because syms contains nothing but x,y,t
        other = False  

        if not syms:
            return MathClass.CONSTANT
        if has_x and has_t and has_y:
            return MathClass.SPATIOTEMPORAL_3D
        if (has_x or has_y) and has_t:
            return MathClass.SPATIOTEMPORAL_2D
        if has_x and has_y:
            return MathClass.SPATIAL_2D
        if has_x or has_y:
            return MathClass.SPATIAL_1D
        if has_t:
            return MathClass.TEMPORAL_1D
        return MathClass.OTHER



    
    @staticmethod
    def parse_expression(expr: str) -> ExpressionFlags:
        """Parse expression and extract flags."""
        if not expr or not isinstance(expr, str) or expr.lower() == 'nan':
            return ExpressionFlags()
        
        try:
            e = ExpressionUtils.standardize_expression(expr)
            tree = se.sympify(e)
            syms = frozenset(str(s) for s in tree.free_symbols)
            cls = ExpressionUtils.classify_expression(syms)
            return ExpressionFlags(
                has_x=('x' in syms),
                has_t=('t' in syms),
                has_y=('y' in syms),
                math_class=cls
            )
        except Exception:
            return ExpressionFlags(parse_error=True)
    
    @staticmethod
    def first_constant(expr: str) -> Optional[float]:
        """Find the first floating-point constant in the expression."""
        if not expr:
            return None
        
        # Regex for scientific or decimal/integer form
        m = re.search(r'([+-]?\d+(\.\d*)?(?:[eE][+-]?\d+)?)', expr)
        if not m:
            return None
        return float(m.group(1))
    
    @staticmethod
    def filter_by_first_const(expressions: List[str], min_val: float = 3e-1) -> List[Tuple[int, str]]:
        """Filter expressions by minimum absolute value of first constant."""
        filtered = []
        for i, expr in enumerate(expressions):
            c = ExpressionUtils.first_constant(expr)
            if c is None:
                continue
            if abs(c) >= min_val:
                filtered.append((i, expr))
        return filtered
    
    @staticmethod
    def find_amplitude_constant(expr: str) -> Optional[float]:
        """Find amplitude constants (often in scientific notation after multiplication)."""
        # Look for scientific notation numbers specifically
        pattern = r'([+-]?\d+\.?\d*[eE][+-]?\d+)'
        matches = re.findall(pattern, expr)
        if matches:
            return float(matches[0])  # First scientific notation number
        
        # Fallback to first constant after sin/cos
        pattern = r'(?:sin|cos)\([^)]+\)\s*\*\s*\(?\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)'
        m = re.search(pattern, expr)
        if m:
            return float(m.group(1))
        
        return None

    @staticmethod
    def filter_by_amplitude(expressions: List[str], min_val: float = 1e-2) -> List[Tuple[int, str]]:
        """Filter expressions by minimum amplitude."""
        filtered = []
        for i, expr in enumerate(expressions):
            amp = ExpressionUtils.find_amplitude_constant(expr)
            if amp is not None and abs(amp) >= min_val:
                filtered.append((i, expr))
        return filtered
    
    @staticmethod
    def negate_and_flip_expressions(expressions):
        """
        Transform list of expressions from 'a - b' to '-a + b'
        """
        transformed = []
        for expr in expressions:
            # Simple approach: wrap first part in -() and change - to +
            if ' - ' in expr:
                parts = expr.split(' - ', 1)  # Split on first ' - '
                new_expr = f"-({parts[0]}) + ({parts[1]})"
            else:
                new_expr = f"-({expr})"
            transformed.append(new_expr)
        return transformed

# =============================================================================
# MODEL UTILITIES
# =============================================================================

class ModelUtils:
    """Utilities for model operations and inference."""
    
    @staticmethod
    def load_config(config_path: str) -> dict:
        """Load YAML configuration file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def load_checkpoint(checkpoint_path: str, config: dict):
        """Load model from PyTorch Lightning checkpoint for inference."""
        print("Loading model checkpoint for inference...")
        model = GrammarVAEModel.load_from_checkpoint(checkpoint_path, config=config)
        model.eval()
        print("Checkpoint loaded successfully.")
        return model
    
    @staticmethod
    def decode_latent_vectors(z_array, model, device, batch_size=256, seed=None):
        """Decode latent vectors to expressions."""
        ds = TensorDataset(torch.from_numpy(z_array).float())
        loader = DataLoader(ds, batch_size=batch_size, pin_memory=True)
        decoded = []
        model.eval()
        
        with torch.no_grad():
            for (z_batch,) in loader:
                z_batch = z_batch.to(device)  # Ensure z_batch is on correct device
                logits = model.model.decoder(z_batch.unsqueeze(1)).mean(dim=1)
                # Keep logits on GPU for construct_expression
                for lg in logits:  # Don't move to CPU here
                    decoded.append(
                        ExpressionUtils.construct_expression(
                            lg.unsqueeze(0),
                            model.model.max_length,
                            device,
                            seed
                        )
                    )
        return decoded
    
    @staticmethod
    def ensure_decoded(model, data_tensor, decoded_pkl: Path, batch_size=256, seed=42):
        """Ensure decoded.pkl exists or create it."""
        p = Path(decoded_pkl)
        if p.exists():
            print("decoded.pkl found; skipping decode/embed")
            return p

        print("Running decode/embed ...")
        model.eval()
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        N = len(data_tensor)
        all_mu, all_sig, all_expr = [], [], []

        loader = DataLoader(
            TensorDataset(data_tensor, torch.zeros(N, 0)),
            batch_size=batch_size, pin_memory=True
        )
        
        with torch.no_grad():
            for (x_batch, _) in loader:
                x = x_batch.to(model.device)
                mu, sig = model.model.encoder(x)
                z = model.model.sample(mu, sig, 1).squeeze(1)

                all_mu.append(mu.cpu().numpy())
                all_sig.append(sig.cpu().numpy())
                all_expr.extend(
                    ModelUtils.decode_latent_vectors(
                        z.cpu().numpy(), model, model.device,
                        batch_size, seed
                    )
                )

        mus = np.vstack(all_mu)
        sigs = np.vstack(all_sig)
        with open(p, 'wb') as f:
            pickle.dump({
                'expressions': all_expr,
                'latent_mus': mus,
                'latent_sigmas': sigs
            }, f)
        print(f"Wrote {p.name}")
        return p


# =============================================================================
# FILE UTILITIES
# =============================================================================

class FileUtils:
    """Utilities for file management and persistence."""
    
    @staticmethod
    def ensure_flags(decoded_pkl: Path, flags_pkl: Path):
        """Ensure expression_flags.pkl exists or create it."""
        p = Path(flags_pkl)
        if p.exists():
            print("expression_flags.pkl found; skipping flag creation")
            return p

        print("Running flag creation ...")
        import pandas as pd
        data = pickle.load(open(decoded_pkl, 'rb'))
        df = pd.DataFrame({
            'expression': data['expressions'],
            'mu': list(data['latent_mus']),
            'sigma': list(data['latent_sigmas'])
        })
        
        flags = list(ThreadPoolExecutor().map(ExpressionUtils.parse_expression, df['expression']))
        fdf = pd.DataFrame([f.as_dict() for f in flags])
        df = pd.concat([df, fdf], axis=1)

        df.to_pickle(p)
        print(f"Wrote {p.name}")
        return p
    
    @staticmethod
    def ensure_clusters(flags_pkl: Path, clusters_pkl: Path):
        """Ensure math_class_clusters.pkl exists or create it."""
        p = Path(clusters_pkl)
        if p.exists():
            print("math_class_clusters.pkl found; skipping clustering")
            return p

        print("Running class clustering ...")
        df = pickle.load(open(flags_pkl, 'rb'))
        grouped = defaultdict(lambda: {'vectors': [], 'expressions': []})
        
        for vec, expr, cls in zip(df['mu'], df['expression'], df['math_class']):
            grouped[cls.name]['vectors'].append(vec)
            grouped[cls.name]['expressions'].append(expr)
        grouped.default_factory = None
        

        with open(p, 'wb') as f:
            pickle.dump(grouped, f)
        print(f"Wrote {p.name}")
        return p
    
    @staticmethod
    def load_pickle(file_path: str):
        """Safely load pickle file."""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle file {file_path}: {e}")
            return None
    
    @staticmethod
    def save_pickle(data, file_path: str):
        """Safely save pickle file."""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved {file_path}")
        except Exception as e:
            print(f"Error saving pickle file {file_path}: {e}")


# =============================================================================
# NUMERICAL UTILITIES
# =============================================================================

class NumericalUtils:
    """Utilities for numerical computations and validations."""
    
    @staticmethod
    def is_number(value) -> bool:
        """Check if a value is numeric."""
        if isinstance(value, (int, float)):
            return True

        if isinstance(value, se.Basic) and not value.free_symbols:
            try:
                numeric_value = value.evalf()
                if numeric_value.is_real:
                    return True
                elif value == se.Integer(0):
                    return True
            except (TypeError, ValueError):
                pass

        return False
    
    @staticmethod
 
    def safe_array_operation(arr, operation='rmse', default_value=1e6):
        """Safely perform array operations with NaN/Inf handling."""
        try:
            arr = np.asarray(arr)
            arr = np.nan_to_num(arr, nan=default_value, posinf=default_value, neginf=default_value)
            
            if operation == 'mean':
                return np.mean(arr)
            
            # Handle complex arrays
            if np.iscomplexobj(arr):  # ← Fixed: use 'arr'
                print("Complex array detected; taking absolute values")
                arr = np.abs(arr)  # ← Fixed: use 'arr'
                
            arr = np.asarray(arr, dtype=float)  # ← Fixed: use 'arr'
            
            if operation == 'rmse':
                finite_mask = np.isfinite(arr)  # ← Fixed: use 'arr'
                if not np.any(finite_mask):
                    return 100.0
                    
                finite_data = arr[finite_mask]  # ← Fixed: use 'arr'
                rmse = np.sqrt(np.mean(finite_data**2))
                
                # Ensure result is real and finite
                if np.iscomplexobj(rmse):
                    rmse = np.abs(rmse)
                    
                return float(rmse) if np.isfinite(rmse) else 100.0
                
            elif operation == 'max':
                return np.max(np.abs(arr))  # ← Fixed: use 'arr'
            else:
                return arr  # ← Fixed: use 'arr'
        except Exception:
            return default_value
    
    
    @staticmethod
    def calculate_rmse(predicted, actual):
        """Calculate RMSE between predicted and actual values."""
        try:
            predicted = np.asarray(predicted)
            actual = np.asarray(actual)
            
            if predicted.shape != actual.shape:
                return 100.0
                
            diff = predicted - actual
            rmse = np.sqrt(np.mean(np.square(diff)))
            return float(rmse) if np.isfinite(rmse) else 100.0
        except Exception:
            return 100.0
    
    @staticmethod
    def safe_evaluate_expression(expr, meshes, param_map, symbols):
        """Safely evaluate symbolic expression on meshes - completely silent."""
        try:
            # Prepare expression
            if isinstance(expr, (sp.Basic, se.Basic)):
                expr_prepared = expr
            else:
                expr_prepared = sp.sympify(str(expr))
            
            # Substitute parameters
            param_map_sym = {sp.Symbol(k): v for k, v in param_map.items()}
            expr_prepared = expr_prepared.subs(param_map_sym)
            
            # Create numerical function
            func = sp.lambdify(list(symbols.values()), expr_prepared, modules=['numpy'])
            
            # Ensure meshes are numpy arrays
            numeric_meshes = [np.asarray(mesh) for mesh in meshes]
            
            # Evaluate with error handling
            with np.errstate(all='ignore'):
                vals = func(*numeric_meshes)
            
            # Handle NaN/Inf
            vals = np.nan_to_num(vals, nan=1e6, posinf=1e6, neginf=1e6)
            return vals
            
        except (NameError, TypeError, AttributeError):
            # Silently handle invalid expressions (e.g., 'y' not defined)
            return np.full_like(meshes[0], 100.0)
            
        except Exception:
            # For any other errors, return penalty array silently
            return np.full_like(meshes[0], 100.0)


# =============================================================================
# CLUSTERING UTILITIES
# =============================================================================

class ClusteringUtils:
    """Utilities for clustering operations."""
    
    @staticmethod
    def create_subclusters(vectors, expressions, n_clusters=5, random_state=42):
        """Create subclusters using k-means."""
        vectors = np.array(vectors)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(vectors)
        
        subclusters = {}
        for i in range(n_clusters):
            mask = cluster_labels == i
            subclusters[i] = {
                'vectors': vectors[mask],
                'expressions': [expr for j, expr in enumerate(expressions) if mask[j]],
                'centroid': kmeans.cluster_centers_[i],
                'expression_indices': np.where(mask)[0]
            }
        
        return subclusters
    
    @staticmethod
    def setup_random_seeds(seed):
        """Setup random seeds for reproducibility."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)


# =============================================================================
# MESH GENERATION UTILITIES
# =============================================================================

class MeshUtils:
    """Utilities for mesh generation and management."""
    
    @staticmethod
    def generate_meshes(mesh_config):
        """Generate mesh grids for PDE evaluation."""
        try:
            axes = []
            for dim_name, details in mesh_config.items():
                start = float(details.get("start", 0.0))
                end = float(details.get("end", 1.0))
                points = int(details.get("points", 100))
                
                # Ensure reasonable values
                points = max(min(points, 500), 10)  # Limit between 10 and 500 points
                
                # Generate linear space
                axis = np.linspace(start, end, points)
                axes.append(axis)
                
            # Create mesh grid
            return np.meshgrid(*axes, indexing='ij')
        
        except Exception as e:
            print(f"Error generating meshes: {e}")
            # Return simple default meshes
            return [np.linspace(0, 1, 50), np.linspace(0, 1, 50)]
    
    @staticmethod
    def setup_problem_symbols_meshes(problem):
        """Set up symbols and meshes for a PDE problem."""
        # Extract dimensions and create symbols
        dimensions = list(problem["mesh"].keys())
        symbols = {dim: se.Symbol(dim) for dim in dimensions}
        
        # Generate meshes
        meshes = MeshUtils.generate_meshes(problem["mesh"])
        
        return symbols, meshes