# %%
from torchvision import transforms
from datasets import PatchedCIFARLoader
from utils.logistic import acc_classifier, fine_tune
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output

from models.nnunet import DynamicEncoder as Encoder, DynamicUNetVectorField as VectorField
# from models.nnRGB import Encoder, VectorField
from utils.mine import MINE
from datasets.MNISTLoader import FastMNISTLoader
from datasets.FastFashionDataset import FastFashionLoader
from BaseExperiment import BaseExperiment

# %%
from CIFAR10 import CIFARFlowExperiment
class PatchedCIFARFlowExperiment(CIFARFlowExperiment):
    def _init_loader(self):
        self.img_size = self.config.get('img_size', 32)
        from datasets.PatchedCIFARLoader import PatchedCIFARLoader
        return PatchedCIFARLoader(patched=True, 
                                  batch_size=self.config['batch_size'], 
                                  device=self.device)

# %% run the experiment
config = {
    'img_size': 32,
    'batch_size': 256,
    'enc_dim': 512,
    'lr': 1e-3,
    'num_test_samples': 5000
}
import os
if os.path.exists(f'data/PatchedCIFARFlowExperiment_checkpoint.pt'):
    os.remove(f'data/PatchedCIFARFlowExperiment_checkpoint.pt')
experiment = PatchedCIFARFlowExperiment(config, seed = 5)
def main():
    experiment.train(num_epochs=5_000, viz_interval=100)

    experiment.summary()
    experiment.test()

    # experiment.bench()
    xtrain = experiment.loader.data
    ytrain = experiment.loader.labels
    xtest = experiment.loader.test_data
    ytest = experiment.loader.test_labels

    acc = acc_classifier(experiment.models['encoder'], 
                            xtrain, ytrain, 
                            xtest, ytest,
                            batch_size=512, epochs=100)
    print(f"Classifier Accuracy: {acc * 100:.2f}%")
        
# %%
if __name__ == "__main__":
    main()
# %%
