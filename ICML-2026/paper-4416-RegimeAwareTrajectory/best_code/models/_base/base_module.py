from typing import Optional
import pytorch_lightning as pl

class BaseModule(pl.LightningModule):
    def __init__(
        self,
        # Visualization params
        log_val_plots: bool = True,
        plot_n_samples: int = 4,
        plot_every_n_epochs: int = 5,
        save_predictions_to_file_path: Optional[str] = None,

        **kwargs,
    ):
        super().__init__(**kwargs)
        self.log_val_plots = log_val_plots
        self.plot_n_samples = plot_n_samples
        self.plot_every_n_epochs = plot_every_n_epochs
        self._val_vis_data = None
        self._test_vis_data = None
        self.test_metrics = {'past': [], 'pred': [], 'true': [], 'ucty': []}
        self.save_predictions_to_file_path = save_predictions_to_file_path
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Training step is not implemented.")
    
    def val_test_step(self, batch, batch_idx, prefix: str):
        raise NotImplementedError(f"{prefix} step is not implemented.")

    def validation_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, prefix='val')
    
    def test_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, prefix='test')

    def configure_optimizers(self):
        raise NotImplementedError("Optimizer configuration is not implemented.")

    def on_validation_epoch_end(self):
        """Log validation visualizations at end of epoch."""
        if not self.log_val_plots:
            return

        # Check if should plot this epoch
        if (self.current_epoch + 1) % self.plot_every_n_epochs != 0:
            self._val_vis_data = None
            return
        
        if self._val_vis_data is None:
            return
        
        self.plot_trajectory_predictions(
            x_input=self._val_vis_data['input'],
            x_target=self._val_vis_data['target'],
            predictions=self._val_vis_data['pred'],
            uncertainties=self._val_vis_data['uncertainty'] if 'uncertainty' in self._val_vis_data else None,
            prefix='val'
        )
        self._val_vis_data = None

    def plot_trajectory_predictions(self, x_input, x_target, predictions, uncertainties=None, prefix='test'):
        """Plot trajectory predictions for given inputs."""
        try:
            from models._base.trajectory_visualization import (
                plot_trajectory_grid, 
                fig_to_tensor
            )
            import matplotlib.pyplot as plt
            
            fig = plot_trajectory_grid(
                input_seqs=x_input.detach().cpu(),
                target_seqs=x_target.detach().cpu(),
                pred_seqs=predictions.detach().cpu(),
                uncertainties=uncertainties.detach().cpu() if uncertainties is not None else None,
                n_samples=self.plot_n_samples,
                n_cols=2,
                title=f'{prefix.capitalize()} Predictions',
            )
            
            # Log to logger (WandB)
            if self.logger is not None:
                try:
                    if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log'):
                        import wandb
                        self.logger.experiment.log({
                            f'{prefix}/trajectory_predictions': wandb.Image(fig),
                        })
                except Exception as e:
                    print(f"Warning: Failed to log to wandb: {e}")
            
            plt.close(fig)
            
        except ImportError:
            print("Warning: trajectory_visualization module not found. Skipping visualization.")
        except Exception as e:
            print(f"Warning: {prefix.capitalize()} visualization failed: {e}")

    def on_test_epoch_end(self):
        """Log test visualizations and save to file."""
        self._save_predictions_to_file(self.save_predictions_to_file_path)
        
         # ===== Log test visualizations =====
        if not self.log_val_plots or self._test_vis_data is None:
            return
        self.plot_trajectory_predictions(
            x_input=self._test_vis_data['input'],
            x_target=self._test_vis_data['target'],
            predictions=self._test_vis_data['pred'],
            uncertainties=self._test_vis_data['uncertainty'] if 'uncertainty' in self._test_vis_data else None,
            prefix='test'
        )
        self._test_vis_data = None

    def _save_predictions_to_file(self, save_path: str):
        """Save predictions and targets to file."""
        if not save_path:
            print("Warning: No save path provided for predictions.")
            return
        
        import os
        import numpy as np
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save test metrics
        np.savez_compressed(
            save_path + '/test_predictions.npz',
            inputs=np.concatenate(self.test_metrics['past'], axis=0),
            predictions=np.concatenate(self.test_metrics['pred'], axis=0),
            targets=np.concatenate(self.test_metrics['true'], axis=0),
            uncertainties=np.concatenate(self.test_metrics['ucty'], axis=0) if self.test_metrics['ucty'] else None,
        )
        print(f"Test predictions saved to: {save_path}")
