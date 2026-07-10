# %%
import torch.nn.functional as F
from utils.logistic import acc_classifier, fine_tune
from utils.hsic import hsic
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output

from models.nn import Encoder, VectorField
from utils.mine import MINE
from datasets.MNISTLoader import FastMNISTLoader
from BaseExperiment import BaseExperiment

# %%
class MNISTFlowExperiment(BaseExperiment):
    def _init_loader(self):
        # data loader
        return FastMNISTLoader(batch_size=self.config['batch_size'], device=self.device)

    def _init_models(self):
        enc_dim = self.config['enc_dim']
        
        #encoder
        self.enc = Encoder(enc_dim).to(self.device)
        #vector field
        self.vf = VectorField(enc_dim).to(self.device)
        
        # optimizers
        opt = torch.optim.AdamW(
            list(self.enc.parameters()) + list(self.vf.parameters()), 
            lr=self.config['lr']
        )
        
        return {'encoder': self.enc, 'vectorfield': self.vf}, opt

    def _info_nce_loss(self, features_x, features_y, temperature=0.1):
        """
        Computes the InfoNCE / NT-Xent loss.
        
        Args:
            features_x: Tensor of shape (batch_size, dim)
            features_y: Tensor of shape (batch_size, dim)
            temperature: Scalar scaling factor (default 0.1)
        """
        # 1. Normalize features to use Cosine Similarity
        #    (Dot product of normalized vectors == Cosine Similarity)
        x_norm = F.normalize(features_x, dim=1)
        y_norm = F.normalize(features_y, dim=1)
        
        # 2. Compute the Similarity Matrix
        #    logits[i, j] = cos_sim(x_i, y_j) / temperature
        #    Shape: (batch_size, batch_size)
        logits = torch.matmul(x_norm, y_norm.T) / temperature
        
        # 3. Create Labels
        #    The "correct" match for x_0 is y_0 (index 0), x_1 is y_1 (index 1), etc.
        batch_size = features_x.shape[0]
        labels = torch.arange(batch_size).to(features_x.device)
        
        # 4. Compute Cross Entropy Loss
        #    This tries to maximize the diagonal (xi, yi) 
        #    and minimize the off-diagonal (xi, yj)
        loss = F.cross_entropy(logits, labels)
        
        return loss

    def _forward_step(self, batch):
        # loss 
        y, x, _ = batch
        yprime, xprime, _ = self.loader.get_batch() 
        
        t = torch.rand(y.size(0), 1).to(self.device)
        xt = t * xprime + (1 - t) * x
        
        enc_yprime = self.models['encoder'](yprime)
        pred_dydt = self.models['vectorfield'](xt, enc_yprime, y, t)
        
        # Rectified Main Loss
        main_loss = torch.mean((xprime - x - pred_dydt)**2)
        
        # Penalty of the vector field at t = 0.5
        t = torch.ones_like(t) * .5
        # b=0.05
        # t = torch.exp(-torch.abs(t - 0.5) / b)
        enc_y = self.models['encoder'](y)
        pred_static = self.models['vectorfield'](xt, enc_y, y, t)
        penalty = torch.mean(pred_static**2)
        # pred_static2 = self.models['vectorfield'](xt, enc_y, y, 1-t)
        # penalty = torch.mean((pred_static + pred_static2)**2)
        
        # Reconstruction Loss
        # ydec = self.models['vectorfield'].decode(enc_y).view(y.size(0), -1)
        # recon_loss = torch.mean((ydec - y)**2)
        recon_loss = torch.tensor(0.0).to(self.device)
        
        # # constrastive loss
        enc_x = self.models['encoder'](x)
        contrastive_loss = self._info_nce_loss(enc_y, enc_x)
        # contrastive_loss = torch.tensor(0.0).to(self.device)
        
        total_loss = main_loss + .1*penalty + .1*recon_loss + .1*contrastive_loss
        
        # Return loss and any metrics to log
        return total_loss, {'total_loss': total_loss.item(), 
                    'penalty': penalty.item(), 
                    'recon': recon_loss.item(), 
                    'contrastive': contrastive_loss.item()}

    def _visualize_live(self, epoch, metrics):
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
        testx = self.loader.test_data
        testlabels = self.loader.test_labels
        fashion_mnist_labels = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]

        for labels in [1, 0, 2, 4, 5, 9, 7, 8, 3, 6]: 
            idxs = (testlabels == labels).nonzero().squeeze()
            Xenc = self.models['encoder'](testx[idxs].to(self.device)).detach().cpu().numpy()
            plt.scatter(Xenc[:, 0], Xenc[:, 1], alpha=0.5, label=fashion_mnist_labels[labels])
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
    'batch_size': 256,
    'enc_dim': 2,
    'lr': 1e-3,
    'num_test_samples': 10000,
    'vis_interval': 100
}
experiment = MNISTFlowExperiment(config, seed = 1)
def main():
    experiment.train(num_epochs=5000, viz_interval = 100)
    experiment.summary()
    experiment.test()

# %%
if __name__ == "__main__":
    main()

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
    
    # train a classifier to separate tshirt and dress
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