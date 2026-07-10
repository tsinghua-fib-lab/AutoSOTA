# %%
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torch import nn
from torch.nn import functional as F
# Assumed imports based on your script context
from models.nn import Encoder, VectorField
from utils.mine import MINE
from datasets.MNISTLoader import FastMNISTLoader
from MNIST import MNISTFlowExperiment
from utils.simclr import SimCLRLoss

class MNISTSimCLRExperiment(MNISTFlowExperiment):
    def _init_models(self):
        self.simclr_loss = SimCLRLoss(temperature=0.1)
        return super()._init_models()   
    
    def _forward_step(self, batch):
        v1, v2, _ = batch
        z1 = self.models['encoder'](v1)
        z2 = self.models['encoder'](v2)

        loss = self.simclr_loss(z1, z2)
        return loss, {'total_loss': loss.item()}
    
    def _visualize_live(self, epoch, metrics):
        print(f"Epoch {epoch}: Loss = {metrics['total_loss']:.4f}")
        
    def _visualize_test(self):
        testx = self.loader.test_data
        testlabels = self.loader.test_labels
        fashion_mnist_labels = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]

        for labels in [0, 5, 7, 9]: 
            idxs = (testlabels == labels).nonzero().squeeze()
            Xenc = self.models['encoder'](testx[idxs].to(self.device)).detach().cpu().numpy()
            plt.scatter(Xenc[:, 0], Xenc[:, 1], alpha=0.5, label=fashion_mnist_labels[labels])
        plt.legend()
        plt.title("Encoded Xbench Visualization")
        plt.xlabel("enc_dim 1")
        plt.ylabel("enc_dim 2")
        plt.show()
    
# %% run the experiment
config = {
    'batch_size': 256,
    'enc_dim': 2,
    'lr': 1e-3,
    'num_test_samples': 10000
}
experiment = MNISTSimCLRExperiment(config, seed = 4)
experiment.train(num_epochs=10000, viz_interval=100)
experiment.summary()

experiment.test()


experiment.bench()
# %%

testx = experiment.loader.test_data
testlabels = experiment.loader.test_labels
fashion_mnist_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

for labels in [0, 1, 2, 3, 9]: 
    idxs = (testlabels == labels).nonzero().squeeze()
    Xenc = experiment.models['encoder'](testx[idxs].to(experiment.device)).detach().cpu().numpy()
    plt.scatter(Xenc[:, 0], Xenc[:, 1], alpha=0.5, label=fashion_mnist_labels[labels], s = 5)
plt.legend()
plt.title("Encoded Xbench Visualization")
plt.xlabel("enc_dim 1")
plt.ylabel("enc_dim 2")
plt.show()

# %%
trainx = experiment.loader.data
trainlabels = experiment.loader.labels

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

label1 = 1
label2 = 3  

# extract tshirt and dress
idxs = ((trainlabels == label1) | (trainlabels == label2)).nonzero().squeeze()
trainx_sub = trainx[idxs]
trainlabels_sub = trainlabels[idxs]
# encode
train_enc = experiment.models['encoder'](trainx_sub.to(experiment.device)).detach().cpu().numpy()
trainlabels_np = (trainlabels_sub == label2).cpu().numpy().astype(int) # dress=1, tshirt=0

clf = LogisticRegression().fit(train_enc, trainlabels_np)
testx_sub = testx[((testlabels == label1) | (testlabels == label2)).nonzero().squeeze()]
testlabels_sub = testlabels[((testlabels == label1) | (testlabels == label2)).nonzero().squeeze()]
test_enc = experiment.models['encoder'](testx_sub.to(experiment.device)).detach().cpu().numpy()
testlabels_np = (testlabels_sub == label2).cpu().numpy().astype(int) # dress=1, tshirt=0
preds = clf.predict(test_enc)
acc = accuracy_score(testlabels_np, preds)  
print(f"Classification Accuracy on {fashion_mnist_labels[label1]} vs {fashion_mnist_labels[label2]}: {acc*100:.2f}%")
# %%
