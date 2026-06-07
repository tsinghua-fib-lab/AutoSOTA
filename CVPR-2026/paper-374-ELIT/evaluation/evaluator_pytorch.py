#!/usr/bin/env python3
"""
PyTorch evaluator matching TensorFlow reference implementation.
Uses pytorch-fid for TF-compatible InceptionV3 weights.
"""

import argparse
import json
import os
import numpy as np
import torch
from scipy import linalg
from tqdm.auto import tqdm
import warnings


class FIDStatistics:
    """Statistics for FID computation."""
    
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    
    def frechet_distance(self, other, eps=1e-6):
        """
        Compute the Frechet distance between two sets of statistics.
        Matches the reference TensorFlow implementation.
        """
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma
        
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        assert mu1.shape == mu2.shape, f"Mean vectors have different lengths: {mu1.shape}, {mu2.shape}"
        assert sigma1.shape == sigma2.shape, f"Covariances have different dimensions: {sigma1.shape}, {sigma2.shape}"
        
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            warnings.warn(f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates")
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class PyTorchEvaluator:
    """
    PyTorch evaluator using pytorch-fid's InceptionV3 (TF-compatible weights).
    
    For sFID, the TF reference evaluator (OpenAI guided-diffusion) extracts
    spatial features from `mixed_6/conv:0`, which is the output of the 1x1 conv
    in the first branch of TF's mixed_6 inception module (= PyTorch Mixed_6d).
    It then takes the first 7 channels → [N, 7, 17, 17] → flatten to [N, 2023].
    
    In pytorch-fid's InceptionV3, Mixed_6d is blocks[2][6]. The equivalent of
    TF's `mixed_6/conv:0` is `blocks[2][6].branch1x1.bn` output (post-BN,
    pre-ReLU), since the TF frozen graph fuses BN into the conv weights.
    """
    
    def __init__(self, device='cuda', batch_size=64):
        self.device = device
        self.batch_size = batch_size
        
        # Import pytorch_fid's InceptionV3 model with TF weights
        from pytorch_fid.inception import InceptionV3
        
        # Single model for both pool3 features (2048-dim) and spatial features
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx], normalize_input=False).to(device)
        self.model.eval()
        
        # Register forward hook on Mixed_6d.branch1x1.bn to capture spatial
        # features matching TF's `mixed_6/conv:0` (post-BN, pre-ReLU).
        # In pytorch-fid's InceptionV3:
        #   blocks[2] = [FIDInceptionA x3, InceptionB, FIDInceptionC x4]
        #   blocks[2][6] = FIDInceptionC (= TF mixed_6 = PyTorch Mixed_6d)
        self._spatial_features = None
        mixed_6d = self.model.blocks[2][6]
        mixed_6d.branch1x1.bn.register_forward_hook(self._capture_spatial)
        
        print(f"Initialized PyTorch evaluator on {device}")
    
    def _capture_spatial(self, module, input, output):
        """Hook to capture spatial features from Mixed_6d.branch1x1 (post-BN, pre-ReLU)."""
        self._spatial_features = output
    
    def preprocess(self, images):
        """
        Preprocess images to match InceptionV3 input format.
        Input: numpy array in [0, 255]
        Output: torch tensor in [-1, 1]
        """
        # Ensure float32
        if images.dtype != np.float32:
            images = images.astype(np.float32)
        
        # Convert NHWC to NCHW if needed
        if images.ndim == 4 and images.shape[-1] == 3:
            images = np.transpose(images, (0, 3, 1, 2))
        
        # To tensor
        images = torch.from_numpy(images).to(self.device)
        
        # Normalize to [0, 1]
        images = images / 255.0
        
        # Resize to 299x299 (bilinear)
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = torch.nn.functional.interpolate(
                images, size=(299, 299), mode='bilinear', align_corners=False
            )
        
        # Normalize to [-1, 1]
        images = 2 * images - 1
        
        return images
    
    def compute_activations(self, images):
        """
        Compute pool3 and spatial activations.
        
        Pool3 features (2048-dim) are used for FID.
        Spatial features (2023-dim = 7 × 17 × 17) are used for sFID,
        extracted from Mixed_6d.branch1x1 via a forward hook to match
        the TF reference evaluator's `mixed_6/conv:0`.
        
        Returns: (pool_features, spatial_features)
        """
        pool_acts = []
        spatial_acts = []
        
        num_batches = (len(images) + self.batch_size - 1) // self.batch_size
        
        with torch.no_grad():
            for i in tqdm(range(0, len(images), self.batch_size), total=num_batches, desc="Computing activations"):
                batch = images[i:i + self.batch_size]
                batch_tensor = self.preprocess(batch)
                
                # Forward pass — also triggers the spatial feature hook
                self._spatial_features = None
                pool_feat = self.model(batch_tensor)[0]
                
                # Pool3 features (2048-dim, for FID)
                if pool_feat.dim() == 4:
                    pool_feat = torch.nn.functional.adaptive_avg_pool2d(pool_feat, (1, 1))
                    pool_feat = pool_feat.squeeze(-1).squeeze(-1)
                pool_acts.append(pool_feat.cpu().numpy())
                
                # Spatial features from Mixed_6d.branch1x1.bn hook
                # Shape: [N, 192, 17, 17] — take first 7 channels to match
                # TF reference: spatial = mixed_6/conv:0[..., :7]
                assert self._spatial_features is not None, (
                    "Spatial feature hook did not fire — check model structure"
                )
                spatial_feat = self._spatial_features[:, :7, :, :]  # [N, 7, 17, 17]
                spatial_feat = spatial_feat.reshape(spatial_feat.shape[0], -1)  # [N, 2023]
                spatial_acts.append(spatial_feat.cpu().numpy())
        
        pool_acts = np.concatenate(pool_acts, axis=0)
        spatial_acts = np.concatenate(spatial_acts, axis=0)
        
        return pool_acts, spatial_acts
    
    def compute_statistics(self, activations):
        """Compute mean and covariance efficiently for large datasets."""
        print(f"Computing statistics for activations of shape {activations.shape}")
        
        n_samples, n_features = activations.shape
        
        # Compute mean
        mu = np.mean(activations, axis=0, dtype=np.float64)
        
        # For spatial features with many dimensions, compute covariance in chunks
        # to avoid memory issues
        if n_features > 1500 and n_samples > 10000:
            print(f"Using chunked covariance computation for large matrix")
            # Center the data
            centered = activations.astype(np.float64) - mu
            
            # Compute covariance using chunked matrix multiplication
            # sigma = (1/(n-1)) * X^T @ X where X is centered
            chunk_size = 10000
            sigma = np.zeros((n_features, n_features), dtype=np.float64)
            
            for i in range(0, n_samples, chunk_size):
                chunk = centered[i:i + chunk_size]
                sigma += np.dot(chunk.T, chunk)
                print(f"  Processed {min(i + chunk_size, n_samples)}/{n_samples} samples")
            
            sigma /= (n_samples - 1)
        else:
            # Standard covariance computation for smaller matrices
            activations = activations.astype(np.float64)
            sigma = np.cov(activations, rowvar=False)
        
        print(f"Statistics computed: mu shape {mu.shape}, sigma shape {sigma.shape}")
        return FIDStatistics(mu, sigma)
    
    def compute_inception_score(self, pool_features, split_size=5000):
        """
        Compute Inception Score from pool3 features.
        """
        print("Computing Inception Score...")
        
        # We need softmax outputs, which requires going through the classifier
        # pytorch-fid doesn't expose this, so we'll compute it manually
        from pytorch_fid.inception import InceptionV3
        
        # Get model that outputs logits
        model_logits = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], 
                                    normalize_input=False, use_fid_inception=True).to(self.device)
        model_logits.eval()
        
        # The pool3 features are already computed, but we need to go through
        # the final FC layer to get softmax. Since we already have the features,
        # we need to reload images or use a different approach.
        # For now, return a placeholder that needs images
        print("Warning: IS computation requires re-processing images through full network")
        return 0.0
    
    def compute_inception_score_from_pool_features(self, pool_features, split_size=5000):
        """
        Compute Inception Score - need to recompute through same model.
        This is a limitation: we need the images, not just features.
        """
        print("Warning: IS computation needs images, not just features. Returning placeholder.")
        return 0.0
    
    def compute_inception_score_from_images(self, images, split_size=5000):
        """
        Compute Inception Score from raw images using torchvision Inception.
        """
        print("Computing Inception Score from images...")
        
        # Load torchvision InceptionV3 with classifier
        from torchvision.models import inception_v3, Inception_V3_Weights
        
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).to(self.device)
        model.eval()
        
        # Get softmax predictions
        preds = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(images), self.batch_size), desc="Computing IS"):
                batch = images[i:i + self.batch_size]
                batch_tensor = self.preprocess(batch)
                
                # Forward through full model to get logits
                logits = model(batch_tensor)
                # Apply softmax
                probs = torch.nn.functional.softmax(logits, dim=1)
                preds.append(probs.cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        
        # Compute IS following reference implementation
        scores = []
        for i in range(0, len(preds), split_size):
            part = preds[i:i + split_size]
            # p(y) = mean over samples
            py = np.mean(part, axis=0)
            # KL(p(y|x) || p(y))
            kl = part * (np.log(part + 1e-10) - np.log(np.expand_dims(py, 0) + 1e-10))
            kl = np.mean(np.sum(kl, axis=1))
            scores.append(np.exp(kl))
        
        return float(np.mean(scores))
    
    def read_statistics(self, npz_path, activations):
        """Read or compute statistics."""
        data = np.load(npz_path)
        
        # Check if precomputed
        if 'mu' in data:
            print("Using precomputed statistics")
            return (
                FIDStatistics(data['mu'], data['sigma']),
                FIDStatistics(data['mu_s'], data['sigma_s'])
            )
        
        print("Computing statistics from activations")
        pool_stats = self.compute_statistics(activations[0])
        spatial_stats = self.compute_statistics(activations[1])
        
        return pool_stats, spatial_stats
    
    def compute_prec_recall(self, ref_features, sample_features, k=3):
        """
        Compute precision and recall using k-NN.
        Simplified version - for exact match use TensorFlow evaluator.
        """
        print(f"Computing Precision and Recall (k={k})...")
        
        # Set OpenBLAS thread limit to avoid memory issues
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        
        from sklearn.neighbors import NearestNeighbors
        
        # Use smaller batches to avoid memory issues
        print("  Computing reference manifold...")
        # Subsample if too large to avoid memory issues
        max_samples = 10000
        if len(ref_features) > max_samples:
            print(f"  Subsampling reference from {len(ref_features)} to {max_samples}")
            ref_indices = np.random.choice(len(ref_features), max_samples, replace=False)
            ref_features_sub = ref_features[ref_indices]
        else:
            ref_features_sub = ref_features
            ref_indices = np.arange(len(ref_features))
        
        if len(sample_features) > max_samples:
            print(f"  Subsampling samples from {len(sample_features)} to {max_samples}")
            sample_indices = np.random.choice(len(sample_features), max_samples, replace=False)
            sample_features_sub = sample_features[sample_indices]
        else:
            sample_features_sub = sample_features
            sample_indices = np.arange(len(sample_features))
        
        # Compute k-NN with reduced data
        nbrs_ref = NearestNeighbors(n_neighbors=min(k+1, len(ref_features_sub)), 
                                     algorithm='ball_tree', n_jobs=1).fit(ref_features_sub)
        distances_ref, _ = nbrs_ref.kneighbors(ref_features_sub)
        radii_ref = distances_ref[:, -1]
        
        nbrs_sample = NearestNeighbors(n_neighbors=min(k+1, len(sample_features_sub)), 
                                        algorithm='ball_tree', n_jobs=1).fit(sample_features_sub)
        distances_sample, _ = nbrs_sample.kneighbors(sample_features_sub)
        radii_sample = distances_sample[:, -1]
        
        # Precision: fraction of sample features within reference manifold
        print("  Computing precision...")
        distances_to_ref, closest_ref_indices = nbrs_ref.kneighbors(sample_features_sub)
        precision = np.mean(distances_to_ref[:, 0] <= radii_ref[closest_ref_indices[:, 0]])
        
        # Recall: fraction of reference features within sample manifold
        print("  Computing recall...")
        distances_to_sample, closest_sample_indices = nbrs_sample.kneighbors(ref_features_sub)
        recall = np.mean(distances_to_sample[:, 0] <= radii_sample[closest_sample_indices[:, 0]])
        
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
        return float(precision), float(recall)


def main():
    parser = argparse.ArgumentParser(description='PyTorch FID/sFID/IS evaluator')
    parser.add_argument('ref_batch', help='Path to reference batch NPZ')
    parser.add_argument('sample_batch', help='Path to sample batch NPZ')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default=None, help='output JSON file')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PyTorch Evaluator (TF-compatible)")
    print("=" * 70)
    
    evaluator = PyTorchEvaluator(device=args.device, batch_size=args.batch_size)
    
    # Load data
    print("\nLoading reference batch...")
    ref_data = np.load(args.ref_batch)
    ref_images = ref_data['arr_0'] if 'arr_0' in ref_data else ref_data[list(ref_data.keys())[0]]
    print(f"Reference shape: {ref_images.shape}")
    
    print("\nLoading sample batch...")
    sample_data = np.load(args.sample_batch)
    sample_images = sample_data['arr_0'] if 'arr_0' in sample_data else sample_data[list(sample_data.keys())[0]]
    print(f"Sample shape: {sample_images.shape}")
    
    # Compute activations
    print("\nComputing reference activations...")
    ref_acts = evaluator.compute_activations(ref_images)
    
    print("\nComputing sample activations...")
    sample_acts = evaluator.compute_activations(sample_images)
    
    # Compute statistics
    print("\nComputing reference statistics...")
    ref_stats, ref_stats_spatial = evaluator.read_statistics(args.ref_batch, ref_acts)
    
    print("\nComputing sample statistics...")
    sample_stats, sample_stats_spatial = evaluator.read_statistics(args.sample_batch, sample_acts)
    
    # Compute metrics
    print("\n" + "=" * 70)
    print("Computing evaluations...")
    print("=" * 70)
    
    fid = sample_stats.frechet_distance(ref_stats)
    print(f"FID: {fid}")
    inception_score = evaluator.compute_inception_score_from_images(sample_images)
    print(f"Inception Score: {inception_score}")
    
    # Compute precision and recall
    precision, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    
    print("\n" + "=" * 70)
    print("NOTE: sFID may not match TensorFlow reference exactly due to")
    print("different InceptionV3 implementations. For exact metrics, use:")
    print("https://github.com/openai/guided-diffusion/tree/main/evaluations")
    print("=" * 70)
    
    results = {
        'is': float(inception_score),
        'fid': float(fid),
        'precision': float(precision),
        'recall': float(recall)
    }
    
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return results


if __name__ == '__main__':
    main()
