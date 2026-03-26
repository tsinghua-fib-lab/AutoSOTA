import time
from flcore.clients.clientbase import Client
from torchvision.transforms import transforms
import torch

class clientAVG(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)

        
    def train(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,weight_decay=1e-5,momentum=self.momentum)
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
                optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                optimizer.step()

        print(f"client{self.id} train_samples:{self.train_samples} train cost time :{time.time() - start_time}")


