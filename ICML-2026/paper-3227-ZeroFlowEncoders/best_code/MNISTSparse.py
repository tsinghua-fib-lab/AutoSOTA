# %%
from utils.hsic import hsic
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Assumed imports based on your script context
from models.nnSparse import Encoder
from models.nnSparse import VectorField
# from models.nn import Encoder, VectorField
from utils.mine import MINE
from datasets.MNISTLoader import FastMNISTLoader
from BaseExperiment import BaseExperiment

class MNISTSparseFlowExperiment(BaseExperiment):
    def _init_loader(self):
        # Specific: Uses FastMNISTLoader
        return FastMNISTLoader(batch_size=self.config['batch_size'], device=self.device)

    def _init_models(self):
        # Specific: Uses Encoder/VectorField
        enc_dim = self.config['enc_dim']
        
        self.enc = Encoder(enc_dim).to(self.device)
        self.vf = VectorField(enc_dim).to(self.device)
        
        opt = torch.optim.AdamW(
            list(self.enc.parameters()) + list(self.vf.parameters()), 
            lr=self.config['lr']
        )
        # Store models in a dict for easy access
        return {'encoder': self.enc, 'vectorfield': self.vf}, opt

    def _forward_step(self, batch):
        x, y, _ = batch
        xprime, yprime, _ = self.loader.get_batch() 
        
        t = torch.rand(x.size(0), 1).to(self.device)
        yt = t * yprime + (1 - t) * y
        
        enc_xprime = self.models['encoder'](xprime)
        pred_dydt = self.models['vectorfield'](yt, enc_xprime, x, t)
        
        # Rectified Main Loss
        main_loss = torch.mean((yprime - y - pred_dydt)**2)
        
        # Penalty of the vector field
        enc_x = self.models['encoder'](x)
        pred_static = self.models['vectorfield'](yt, enc_x, x, t)
        penalty = torch.mean(pred_static**2)

        # Reconstruction Loss
        xdec = self.models['vectorfield'].decode(enc_x).view(x.size(0), -1)
        recon_loss = torch.mean(torch.abs(xdec - x))
                
        total_loss = main_loss + 1*penalty + 0*recon_loss
        
        # Return loss and any metrics to log
        return total_loss, {'total_loss': total_loss.item(), 'penalty': penalty.item(), 'recon': recon_loss.item()}

    def _visualize_live(self, epoch, metrics):
        # Specific: MNIST Grayscale Plotting
        # (Assuming you keep a fixed batch for viz)
        if not hasattr(self, 'vis_batch'):
            self.vis_batch, _, _ = self.loader.get_batch()
            
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
                plt.imshow(recon[imgidx].reshape(32, 32), cmap='gray')
                plt.axis('off')
                plt.subplot(2, 10, imgidx + 11)
                plt.imshow(self.vis_batch[imgidx].cpu().numpy().reshape(32, 32), cmap='gray')
                plt.axis('off')
            
            # print metrics in title
            title = []
            for k, v in metrics.items():
                title.append(f"{k}: {v:.4f}")
            plt.suptitle(f"Epoch {epoch}: " + ", ".join(title))
            plt.show() 

        
        self.models['encoder'].train()
        self.models['vectorfield'].train()

    def _visualize_test(self):
        fashion_mnist_labels = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]

        for labels in [1, 0, 2, 4, 5, 9, 7, 8, 3, 6]: 
            idxs = (self.testlabels == labels).nonzero().squeeze()
            Xenc = self.models['encoder'](self.testx[idxs].to(self.device)).detach().cpu().numpy()
            plt.scatter(Xenc[:, 0], Xenc[:, 1], alpha=0.5, label=fashion_mnist_labels[labels])
        plt.legend()
        plt.title("Encoded Xbench Visualization")
        plt.xlabel("enc_dim 1")
        plt.ylabel("enc_dim 2")
        plt.show()

    def _bench(self):
        testx = self.testx
        testy = self.testy
        testlabels = self.testlabels

        Xenc = self.models['encoder'](testx).detach().cpu().numpy()
        y = testlabels.cpu().numpy()

        # train a classfier on encodings and compute cross-val accuracy
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression() 
        scores = cross_val_score(clf, Xenc, y, cv=5)

        print(f"Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

        # MI0 = MINE(testx, testy)[0]
        # Xenc = self.models['encoder'](testx).detach()
        # MI1 = MINE(Xenc, testy)[0]
        # print(f"MI(X, Y) = {MI0:.4f}")
        # print(f"MI(enc(X), Y) = {MI1:.4f}")

        # Xenc = self.models['encoder'](testx).detach()
        # MI1 = MINE(Xenc, testx)[0]
        # print(f"MI(enc(X), X) = {MI1:.4f}")

        # hsic0 = hsic(testx, testy)
        # Xenc = self.models['encoder'](testx).detach()
        # hsic1 = hsic(Xenc, testy)
        # print(f"HSIC(X, Y) = {hsic0:.4f}")
        # print(f"HSIC(enc(X), Y) = {hsic1:.4f}")

        # Xenc = self.models['encoder'](testx).detach()
        # hsic1 = hsic(Xenc, testx)
        # print(f"HSIC(enc(X), X) = {hsic1:.4f}")


# %%
if __name__ == "__main__":
    # %% run the experiment
    config = {
        'batch_size': 128,
        'enc_dim': 128,
        'lr': 1e-3,
        'num_test_samples': 10000
    }
    # %%
    experiment = MNISTSparseFlowExperiment(config, seed = 999)
    experiment.train(num_epochs=10000, viz_interval=100)
    # %%
    experiment.summary()
    # %%
    experiment.test()

    # %%
    experiment.bench()
    # %%
    enc_x = experiment.models['encoder'](experiment.testx.to(experiment.device)).detach().cpu().numpy()
    test_y = experiment.testlabels.cpu().numpy()

    # plt.scatter(enc_x[:, 0], enc_x[:, 1], c=test_y, cmap='tab10', alpha=0.7)
    # plt.title("Encoded Xbench Visualization")
    # plt.xlabel("enc_dim 1")
    # plt.ylabel("enc_dim 2")
    # plt.show()
    
    # run linear classifier on encoded testing set
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(enc_x, test_y)
    print(f"Accuracy: {clf.score(enc_x, test_y):.4f}")
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(experiment.testx.cpu().numpy(), test_y)
    print(f"Accuracy: {clf.score(experiment.testx.cpu().numpy(), test_y):.4f}")

    
    # %%