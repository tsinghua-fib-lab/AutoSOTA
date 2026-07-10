# %%
from torchvision import transforms
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
from MNIST import MNISTFlowExperiment
class CIFARFlowExperiment(MNISTFlowExperiment):
    def _init_loader(self):
        self.img_size = self.config.get('img_size', 32)
        from datasets.CIFAR10Loader import CIFAR10Loader
        cache_path=f'./data/cifar10_cache_{self.img_size}x{self.img_size}.pt'
        return CIFAR10Loader(batch_size=self.config['batch_size'], 
                             img_size=self.img_size, device=self.device, cache_path=cache_path)
    
    def _init_models(self):
        enc_dim = self.config['enc_dim']

        self.enc = Encoder(enc_dim, img_size=self.img_size).to(self.device)
        self.vf = VectorField(enc_dim, img_size=self.img_size).to(self.device)
        
        opt = torch.optim.AdamW(
            list(self.enc.parameters()) + list(self.vf.parameters()), 
            lr=self.config['lr']
        )
        return {'encoder': self.enc, 'vectorfield': self.vf}, opt

    def _visualize_live(self, epoch, metrics):
        img_size = self.img_size
        clear_output(wait=True)
        self.models['encoder'].eval() 
        self.models['vectorfield'].eval()
        
        with torch.no_grad():
            z = self.models['encoder'](self.vis_batch)
            recon = self.models['vectorfield'].decode(z).cpu().numpy()
        
            clear_output(wait=True)
            plt.figure(figsize=(12, 4))
            for imgidx in range(10):
                plt.subplot(2, 10, imgidx + 1)
                plt.imshow(recon[imgidx].reshape(3, img_size, img_size).transpose(1, 2, 0))
                plt.axis('off')
                plt.subplot(2, 10, imgidx + 11)
                plt.imshow(self.vis_batch[imgidx].cpu().numpy().reshape(3, img_size, img_size).transpose(1, 2, 0))
                plt.axis('off')
            plt.suptitle(f"Epoch {epoch}: Loss {metrics['total_loss']:.4f}, Penalty {metrics['penalty']:.4f}, Recon {metrics['recon']:.4f}, Constrastive {metrics['contrastive']:.4f}")
            plt.show() 

        
        self.models['encoder'].train()
        self.models['vectorfield'].train()

    def _visualize_test(self):
        # randomly pick three labels to visualize
        test_labels = self.loader.test_labels
        test_data = self.loader.test_data
        img_size = self.img_size
        
        for labels in [0, 7]:
            idxs = (test_labels == labels).nonzero().squeeze()
            Xenc = self.models['encoder'](test_data[idxs].to(self.device)).detach().cpu().numpy()
            plt.scatter(Xenc[:, 0], Xenc[:, 1], alpha=0.5, label=f"{self.loader.get_label_text(labels)}")
        plt.legend()
        plt.title("Encoded Xbench Visualization")
        plt.xlabel("enc_dim 1")
        plt.ylabel("enc_dim 2")
        plt.show()

    def _bench(self):
        xtrain = self.loader.data
        ytrain = self.loader.labels
        xtest = self.loader.test_data
        ytest = self.loader.test_labels
            
        acc = acc_classifier(self.models['encoder'], 
                                xtrain, ytrain, 
                                xtest, ytest,
                                batch_size=512, epochs=100)
        print(f"Classifier Accuracy: {acc * 100:.2f}%")

# %% run the experiment
config = {
    'img_size': 32,
    'batch_size': 256,
    'enc_dim': 256,
    'lr': 1e-4,
    'num_test_samples': 5000
}

experiment = CIFARFlowExperiment(config, seed = 5)
def main():
    experiment.train(num_epochs=100000, viz_interval=100)
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

