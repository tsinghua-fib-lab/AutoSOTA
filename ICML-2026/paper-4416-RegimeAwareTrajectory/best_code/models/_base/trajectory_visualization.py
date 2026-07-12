import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from typing import Optional, List, Tuple, Union, Dict
import io
from PIL import Image

# ===================== Publication-Quality Style Configuration =====================

def setup_publication_style():
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        
        # Figure settings
        'figure.figsize': (6, 4),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        
        # Axes settings
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'grid.linestyle': '--',
        
        # Line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        
        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,
        
        # Tick settings
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        
        # Other
        'text.usetex': False,  # Set True if LaTeX is available
        'mathtext.fontset': 'stix',
    })


# ===================== Color Palettes =====================

class ColorPalette:
    """
    Professional color palettes for scientific visualization.
    Colors are chosen for:
    - Color-blind friendliness
    - Print compatibility
    - Visual distinction
    """
    
    # Primary palette (for main elements)
    GROUND_TRUTH = '#2E86AB'      # Deep blue - solid, trustworthy
    PREDICTION = '#E94F37'         # Coral red - attention-grabbing
    INPUT = '#1B4332'              # Dark green - context/history
    UNCERTAINTY = '#E94F37'        # Same as prediction, with alpha
    
    # Extended palette (for multiple series)
    SERIES_COLORS = [
        '#2E86AB',  # Blue
        '#E94F37',  # Red
        '#F6AE2D',  # Yellow/Gold
        '#33A853',  # Green
        '#8E44AD',  # Purple
        '#E67E22',  # Orange
        '#1ABC9C',  # Teal
        '#E91E63',  # Pink
    ]
    
    # Background and grid
    BACKGROUND = '#FFFFFF'
    GRID = '#E0E0E0'
    SPLIT_LINE = '#7F7F7F'
    
    # Uncertainty bands
    UNCERTAINTY_ALPHA = 0.2
    
    @classmethod
    def get_series_color(cls, idx: int) -> str:
        """Get color for series by index (cycles through palette)."""
        return cls.SERIES_COLORS[idx % len(cls.SERIES_COLORS)]


# ===================== Main Visualization Functions =====================

def plot_trajectory_prediction(
    input_seq: Union[torch.Tensor, np.ndarray],
    target_seq: Union[torch.Tensor, np.ndarray],
    pred_seq: Union[torch.Tensor, np.ndarray],
    uncertainty: Optional[Union[torch.Tensor, np.ndarray]] = None,
    sample_idx: int = 0,
    feature_idx: int = 0,
    title: Optional[str] = None,
    xlabel: str = 'Time Step',
    ylabel: str = 'Value',
    figsize: Tuple[float, float] = (8, 4),
    show_split_line: bool = True,
    confidence_level: float = 0.95,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a single trajectory with input, ground truth, and prediction.
    
    Args:
        input_seq: Input sequence [B, L_in, D] or [L_in, D] or [L_in]
        target_seq: Target sequence [B, L_out, D] or [L_out, D] or [L_out]
        pred_seq: Predicted sequence [B, L_out, D] or [L_out, D] or [L_out]
        uncertainty: Optional uncertainty estimates (same shape as pred_seq)
        sample_idx: Which sample to plot (if batch dimension exists)
        feature_idx: Which feature to plot (if multiple features)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        show_split_line: Whether to show vertical line at input/prediction split
        confidence_level: Confidence level for uncertainty bands
        save_path: Path to save figure (optional)
        
    Returns:
        matplotlib Figure object
    """
    setup_publication_style()
    
    # Convert to numpy and extract single series
    input_np = _to_numpy_1d(input_seq, sample_idx, feature_idx)
    target_np = _to_numpy_1d(target_seq, sample_idx, feature_idx)
    pred_np = _to_numpy_1d(pred_seq, sample_idx, feature_idx)
    
    if uncertainty is not None:
        unc_np = _to_numpy_1d(uncertainty, sample_idx, feature_idx)
    else:
        unc_np = None
    
    # Create time indices
    L_in = len(input_np)
    L_out = len(target_np)
    t_input = np.arange(L_in)
    t_output = np.arange(L_in, L_in + L_out)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot input sequence (solid, darker)
    ax.plot(t_input, input_np, 
            color=ColorPalette.INPUT, 
            linestyle='-', 
            linewidth=2.0,
            label='Input (History)', 
            zorder=3)
    
    # Plot ground truth (solid line)
    ax.plot(t_output, target_np, 
            color=ColorPalette.GROUND_TRUTH, 
            linestyle='-', 
            linewidth=2.0,
            label='Ground Truth', 
            zorder=3)
    
    # Plot prediction (dashed line)
    ax.plot(t_output, pred_np, 
            color=ColorPalette.PREDICTION, 
            linestyle='--', 
            linewidth=2.0,
            dashes=(5, 3),  # Custom dash pattern
            label='Prediction', 
            zorder=4)
    
    # Plot uncertainty band if available
    if unc_np is not None:
        # Assuming uncertainty is standard deviation
        z_score = _get_z_score(confidence_level)
        lower = pred_np - z_score * unc_np
        upper = pred_np + z_score * unc_np
        
        ax.fill_between(t_output, lower, upper, 
                        color=ColorPalette.UNCERTAINTY, 
                        alpha=ColorPalette.UNCERTAINTY_ALPHA,
                        label=f'{int(confidence_level*100)}% CI',
                        zorder=2)
    
    # Add vertical split line
    if show_split_line:
        ax.axvline(x=L_in - 0.5, 
                   color=ColorPalette.SPLIT_LINE, 
                   linestyle=':', 
                   linewidth=1.5, 
                   alpha=0.7,
                   zorder=1)
        
        # Add subtle background shading
        ax.axvspan(-0.5, L_in - 0.5, 
                   alpha=0.05, 
                   color='gray',
                   zorder=0)
    
    # Styling
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if title:
        ax.set_title(title, fontweight='medium')
    
    # Legend
    ax.legend(loc='best', ncol=1)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_trajectory_grid(
    input_seqs: Union[torch.Tensor, np.ndarray],
    target_seqs: Union[torch.Tensor, np.ndarray],
    pred_seqs: Union[torch.Tensor, np.ndarray],
    uncertainties: Optional[Union[torch.Tensor, np.ndarray]] = None,
    n_samples: int = 4,
    n_cols: int = 2,
    feature_idx: int = 0,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    sample_indices: Optional[List[int]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a grid of trajectory predictions.
    
    Args:
        input_seqs: Input sequences [B, L_in, D]
        target_seqs: Target sequences [B, L_out, D]
        pred_seqs: Predicted sequences [B, L_out, D]
        uncertainties: Optional uncertainty estimates [B, L_out, D]
        n_samples: Number of samples to plot
        n_cols: Number of columns in grid
        feature_idx: Which feature to plot
        title: Overall figure title
        figsize: Figure size (auto-calculated if None)
        sample_indices: Specific sample indices to plot (random if None)
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    setup_publication_style()
    
    # Convert to numpy
    input_np = _to_numpy(input_seqs)
    target_np = _to_numpy(target_seqs)
    pred_np = _to_numpy(pred_seqs)
    unc_np = _to_numpy(uncertainties) if uncertainties is not None else None
    
    B = input_np.shape[0]
    n_samples = min(n_samples, B)
    
    # Select sample indices
    if sample_indices is None:
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(B, n_samples, replace=False)
        # sample_indices = list(range(min(n_samples, B)))
    
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # Calculate figure size
    if figsize is None:
        figsize = (4 * n_cols, 2.5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    for idx, sample_idx in enumerate(sample_indices):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Extract data
        inp = _extract_feature(input_np[sample_idx], feature_idx)
        tgt = _extract_feature(target_np[sample_idx], feature_idx)
        prd = _extract_feature(pred_np[sample_idx], feature_idx)
        unc = _extract_feature(unc_np[sample_idx], feature_idx) if unc_np is not None else None
        
        # Time indices
        L_in, L_out = len(inp), len(tgt)
        t_input = np.arange(L_in)
        t_output = np.arange(L_in, L_in + L_out)
        
        # Plot
        ax.plot(t_input, inp, color=ColorPalette.INPUT, 
                linestyle='-', linewidth=1.5, label='Input')
        ax.plot(t_output, tgt, color=ColorPalette.GROUND_TRUTH, 
                linestyle='-', linewidth=1.5, label='GT')
        ax.plot(t_output, prd, color=ColorPalette.PREDICTION, 
                linestyle='--', linewidth=1.5, dashes=(4, 2), label='Pred')
        
        if unc is not None:
            lower = prd - 1.96 * unc
            upper = prd + 1.96 * unc
            ax.fill_between(t_output, lower, upper, 
                           color=ColorPalette.UNCERTAINTY, 
                           alpha=ColorPalette.UNCERTAINTY_ALPHA)
        
        # Split line
        ax.axvline(x=L_in - 0.5, color=ColorPalette.SPLIT_LINE, 
                   linestyle=':', linewidth=1.0, alpha=0.6)
        
        # Compute MSE for this sample
        mse = np.mean((tgt - prd) ** 2)
        ax.set_title(f'Sample {sample_idx} (MSE: {mse:.4f})', fontsize=9)
        
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Only show legend for first plot
        if idx == 0:
            ax.legend(loc='upper right', fontsize=7, ncol=1)
    
    # Hide empty subplots
    for idx in range(n_samples, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    # Overall title
    if title:
        fig.suptitle(title, fontsize=12, fontweight='medium', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_trajectory_comparison(
    input_seq: Union[torch.Tensor, np.ndarray],
    target_seq: Union[torch.Tensor, np.ndarray],
    pred_seq: Union[torch.Tensor, np.ndarray],
    uncertainty: Optional[Union[torch.Tensor, np.ndarray]] = None,
    sample_idx: int = 0,
    feature_idx: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot trajectory with additional error subplot.
    Two-panel layout: main trajectory and prediction error.
    
    Args:
        input_seq, target_seq, pred_seq: Sequences to plot
        uncertainty: Optional uncertainty estimates
        sample_idx, feature_idx: Which sample/feature to plot
        metrics: Optional dict of metrics to display
        title: Overall figure title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    setup_publication_style()
    
    # Convert to numpy
    input_np = _to_numpy_1d(input_seq, sample_idx, feature_idx)
    target_np = _to_numpy_1d(target_seq, sample_idx, feature_idx)
    pred_np = _to_numpy_1d(pred_seq, sample_idx, feature_idx)
    unc_np = _to_numpy_1d(uncertainty, sample_idx, feature_idx) if uncertainty is not None else None
    
    # Time indices
    L_in, L_out = len(input_np), len(target_np)
    t_input = np.arange(L_in)
    t_output = np.arange(L_in, L_in + L_out)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                    height_ratios=[3, 1],
                                    sharex=True)
    
    # ===== Top panel: Trajectory =====
    ax1.plot(t_input, input_np, color=ColorPalette.INPUT, 
             linestyle='-', linewidth=2.0, label='Input')
    ax1.plot(t_output, target_np, color=ColorPalette.GROUND_TRUTH, 
             linestyle='-', linewidth=2.0, label='Ground Truth')
    ax1.plot(t_output, pred_np, color=ColorPalette.PREDICTION, 
             linestyle='--', linewidth=2.0, dashes=(5, 3), label='Prediction')
    
    if unc_np is not None:
        lower = pred_np - 1.96 * unc_np
        upper = pred_np + 1.96 * unc_np
        ax1.fill_between(t_output, lower, upper, 
                        color=ColorPalette.UNCERTAINTY, 
                        alpha=ColorPalette.UNCERTAINTY_ALPHA,
                        label='95% CI')
    
    ax1.axvline(x=L_in - 0.5, color=ColorPalette.SPLIT_LINE, 
                linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.axvspan(-0.5, L_in - 0.5, alpha=0.05, color='gray')
    
    ax1.set_ylabel('Value')
    ax1.legend(loc='upper right', ncol=2)
    
    # Add metrics text box
    if metrics:
        metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ===== Bottom panel: Error =====
    error = target_np - pred_np
    ax2.bar(t_output, error, color=ColorPalette.PREDICTION, alpha=0.7, width=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axvline(x=L_in - 0.5, color=ColorPalette.SPLIT_LINE, 
                linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Error')
    
    # Title
    if title:
        fig.suptitle(title, fontsize=12, fontweight='medium')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ===================== Logger Integration =====================

def fig_to_image(fig: plt.Figure) -> Image.Image:
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf)
    return image


def fig_to_tensor(fig: plt.Figure) -> torch.Tensor:
    """Convert matplotlib figure to torch tensor for TensorBoard."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf)
    
    # Convert to tensor [C, H, W]
    img_array = np.array(image)
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
    tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    return tensor


class TrajectoryVisualizer:
    """
    Helper class for logging trajectory visualizations during training.
    Supports TensorBoard and WandB.
    """
    
    def __init__(
        self,
        n_samples: int = 4,
        n_cols: int = 2,
        feature_idx: int = 0,
        log_every_n_epochs: int = 1,
    ):
        """
        Args:
            n_samples: Number of samples to visualize
            n_cols: Number of columns in grid
            feature_idx: Which feature to plot
            log_every_n_epochs: How often to log visualizations
        """
        self.n_samples = n_samples
        self.n_cols = n_cols
        self.feature_idx = feature_idx
        self.log_every_n_epochs = log_every_n_epochs
        self._sample_indices = None
    
    def should_log(self, epoch: int) -> bool:
        """Check if should log this epoch."""
        return epoch % self.log_every_n_epochs == 0
    
    def create_visualization(
        self,
        input_seqs: torch.Tensor,
        target_seqs: torch.Tensor,
        pred_seqs: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
    ) -> plt.Figure:
        """
        Create visualization figure.
        
        Args:
            input_seqs: [B, L_in, D]
            target_seqs: [B, L_out, D]
            pred_seqs: [B, L_out, D]
            uncertainties: [B, L_out, D] (optional)
            epoch: Current epoch for title
            
        Returns:
            matplotlib Figure
        """
        # Keep same samples across epochs for consistency
        B = input_seqs.shape[0]
        if self._sample_indices is None or len(self._sample_indices) > B:
            n = min(self.n_samples, B)
            self._sample_indices = list(range(n))
        
        title = f'Validation Predictions'
        if epoch is not None:
            title += f' (Epoch {epoch})'
        
        fig = plot_trajectory_grid(
            input_seqs=input_seqs,
            target_seqs=target_seqs,
            pred_seqs=pred_seqs,
            uncertainties=uncertainties,
            n_samples=len(self._sample_indices),
            n_cols=self.n_cols,
            feature_idx=self.feature_idx,
            title=title,
            sample_indices=self._sample_indices,
        )
        
        return fig
    
    def log_to_tensorboard(
        self,
        writer,  # torch.utils.tensorboard.SummaryWriter
        fig: plt.Figure,
        global_step: int,
        tag: str = 'val/trajectory_predictions',
    ):
        """Log figure to TensorBoard."""
        img_tensor = fig_to_tensor(fig)
        writer.add_image(tag, img_tensor, global_step)
        plt.close(fig)
    
    def log_to_wandb(
        self,
        fig: plt.Figure,
        key: str = 'val/trajectory_predictions',
    ) -> Dict:
        """Return dict for WandB logging."""
        import wandb
        return {key: wandb.Image(fig)}


# ===================== Helper Functions =====================

def _to_numpy(x: Union[torch.Tensor, np.ndarray, None]) -> Optional[np.ndarray]:
    """Convert tensor to numpy array."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def _to_numpy_1d(
    x: Union[torch.Tensor, np.ndarray],
    sample_idx: int = 0,
    feature_idx: int = 0,
) -> np.ndarray:
    """Convert tensor to 1D numpy array, extracting specified sample and feature."""
    x_np = _to_numpy(x)
    
    if x_np.ndim == 1:
        return x_np
    elif x_np.ndim == 2:
        if x_np.shape[1] == 1:
            return x_np[:, 0]
        return x_np[:, feature_idx]
    elif x_np.ndim == 3:
        if x_np.shape[2] == 1:
            return x_np[sample_idx, :, 0]
        return x_np[sample_idx, :, feature_idx]
    else:
        raise ValueError(f"Unexpected tensor dimension: {x_np.ndim}")


def _extract_feature(x: np.ndarray, feature_idx: int = 0) -> np.ndarray:
    """Extract single feature from array."""
    if x.ndim == 1:
        return x
    elif x.ndim == 2:
        if x.shape[1] == 1:
            return x[:, 0]
        return x[:, feature_idx]
    else:
        raise ValueError(f"Unexpected array dimension: {x.ndim}")


def _get_z_score(confidence_level: float) -> float:
    """Get z-score for confidence level."""
    from scipy import stats
    return stats.norm.ppf((1 + confidence_level) / 2)
