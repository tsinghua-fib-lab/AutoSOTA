import time
from flcore.clients.clientbase import Client
from torchvision.transforms import transforms
from utils.fedspeed_utils import ESAM
import torch

class clientSAM(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)
        self.rho = args.rho
        
    def train(self):
        base_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5,momentum=self.momentum)
        optimizer = ESAM(self.model.parameters(), base_optimizer, rho=self.rho)
        trainloader = self.load_train_data()
        self.model.train()
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])
        
        start_time = time.time()
        for step in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                if self.dataset != "agnews":
                    x = transform_train(x)
                optimizer.paras = [x, y, self.loss, self.model]
                optimizer.step()

                base_optimizer.step()   

        print(f"client{self.id} train_samples:{self.train_samples} train cost time :{time.time() - start_time}")


