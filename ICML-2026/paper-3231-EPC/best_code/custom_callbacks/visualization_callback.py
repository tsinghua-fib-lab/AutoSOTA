import torch
from lightning import Callback


class VisualizationCallback(Callback):
    viz_order = None  # order in which the attributes must be visualized

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        self.preprocess(trainer, pl_module, outputs, batch)
        self.visualize(batch, outputs)

    def preprocess(
        self, trainer, pl_module, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ):
        """Optional preprocessing step"""
        pass

    def visualize(
        self,
        batch: dict[str, torch.Tensor],
        predictions: dict[str, torch.Tensor],
    ):
        """Visualizes every single item in the batch, and creates a slider for the user"""
        import ipywidgets as widgets
        from IPython.display import display

        # Create an interactive slider to navigate through the batch predictions
        batch_size = next(iter(batch.values())).size(0)
        slider = widgets.IntSlider(value=0, min=0, max=batch_size - 1, description="Item:")

        # Function to update the displayed item based on the slider value
        def update_item(idx):
            self.visualize_single_item(batch, predictions, idx)

        # Connect the slider to the update function
        display(widgets.interact(update_item, idx=slider))

    @torch.no_grad()
    def visualize_single_item(
        self,
        batch: dict[str, torch.Tensor],
        predictions: dict[str, torch.Tensor],
        idx: int = 0,
    ):
        """Visualizes the results for the 'idx'-th item in the batch"""
        import matplotlib.pyplot as plt

        # Visualize keys (always in the same order)
        viz_order = self.viz_order if self.viz_order is not None else sorted(list(batch.keys()))
        for key in viz_order:
            pred = predictions[key][idx].detach().cpu() if key in predictions else None
            self.visualize_key(key, batch[key][idx].cpu(), pred)

        plt.show()

    def visualize_key(self, key: str, ground_truth: torch.Tensor, pred: torch.Tensor):
        """
        Implements the look-up table for every key visualization.
        Expects visualization functions to be implemented for every key.

        """
        getattr(self, "visualize_" + key)(ground_truth, pred)
