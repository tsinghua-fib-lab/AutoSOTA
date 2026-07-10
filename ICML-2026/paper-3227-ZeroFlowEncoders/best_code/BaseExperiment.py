import os
import time
from abc import ABC, abstractmethod
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output

class BaseExperiment(ABC):
    def __init__(self, config, seed=3):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if seed:
            torch.manual_seed(seed)

        # Initialize abstract components
        self.loader = self._init_loader()
        self.models, self.optimizer = self._init_models()
        
        # vis_batch
        self.vis_batch, _, _ = self.loader.get_batch()

        # History containers
        self.history = {} # generic dict for 'loss', 'penalty', 'accuracy', etc.

    # --- ABSTRACT METHODS (You MUST implement these) ---
    
    @abstractmethod
    def _init_loader(self):
        """Return your dataloader instance."""
        pass

    @abstractmethod
    def _init_models(self):
        """Return (dict_of_models, optimizer)."""
        pass

    @abstractmethod
    def _forward_step(self, batch):
        """
        Perform one training step.
        Args:
            batch: output from your loader
        Returns:
            loss (Tensor): The value to backpropagate.
            metrics (dict): A dictionary of values to log (e.g. {'penalty': 0.1})
        """
        pass
    
    @abstractmethod
    def _visualize_live(self, epoch, metrics):
        """Draw plots during training (e.g. reconstructions)."""
        pass

    @abstractmethod
    def _visualize_test(self):
        """Draw final plots on the test set."""
        pass

    @abstractmethod
    def _bench(self):
        """perform benchmarking"""
        pass

    # --- COMMON METHODS (Reuse these) ---

    def train(self, num_epochs=5000, viz_interval=10):
        print(f"Starting experiment on {self.device}...")
        
        try:
            # load the latest checkpoint
            start_epoch = self.resume(f"./data/{self.__class__.__name__}_checkpoint.pt") if os.path.exists(f"./data/{self.__class__.__name__}_checkpoint.pt") else 0
            print(f"Resumed from epoch {start_epoch}")
            
            for epoch in range(start_epoch, num_epochs):
                # 1. Get Data (Generic)
                batch = self.loader.get_batch()
                
                # 2. Compute Loss (Specific)
                loss, metrics = self._forward_step(batch)
                
                # 3. Optimize (Generic)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 4. Log History (Generic)
                for k, v in metrics.items():
                    self._update_history(k, v)
                    
                # 5. Visualize (Specific)
                if epoch % viz_interval == 0:
                    self._visualize_live(epoch, metrics)

                if epoch % 1000 == 0:
                    self.dump(f"./data/{self.__class__.__name__}_checkpoint.pt", epoch)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Exiting gracefully.")


    def summary(self):
        """Automatically plots all keys stored in self.history"""
        plt.figure(figsize=(12, 4))
        keys = list(self.history.keys())
        for i, key in enumerate(keys):
            plt.subplot(1, len(keys), i + 1)
            plt.plot(self.history[key])
            plt.title(f"{key.capitalize()} History")
        plt.show()

    def _update_history(self, key, value):
        if key not in self.history:
            self.history[key] = []
        self.history[key].append(value)

    def test(self):
        """Run final visualization on test set."""
        self._visualize_test()

    def bench(self):
        """Run benchmarking."""
        return self._bench()
    
    def dump(self, path, epoch):
        """Save model checkpoints."""
        state = {
            'epoch': epoch,
            'models': {k: v.state_dict() for k, v in self.models.items()},
            'optimizer': self.optimizer.state_dict(),
            'history': self.history
        }
        
        # Save Scheduler if it exists
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
            
        torch.save(state, path)
        print(f"Checkpoint saved to {path} (Epoch {epoch})")
        
    def resume(self, path):
        """Load model checkpoints and return the start epoch."""
        # Add map_location to ensure it loads on the correct device (CPU or GPU)
        device = next(self.models['encoder'].parameters()).device
        state = torch.load(path, map_location=device)
        
        # 1. Load Models
        for k, v in self.models.items():
            v.load_state_dict(state['models'][k])
            
        # 2. Load Optimizer
        self.optimizer.load_state_dict(state['optimizer'])
        
        # 3. Load Scheduler (Critical for LR continuity)
        if 'scheduler' in state and hasattr(self, 'scheduler') and self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler'])
            
        # 4. Restore History
        self.history = state['history']
        
        start_epoch = state['epoch'] + 1
        print(f"Checkpoint loaded. Resuming from Epoch {start_epoch}")
        
        return start_epoch