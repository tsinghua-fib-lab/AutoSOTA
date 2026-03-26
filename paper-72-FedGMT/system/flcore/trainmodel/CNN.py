import torch.nn as nn
import torch.nn.functional as F



class SimpleCNN(nn.Module):
    def __init__(self, input_dim = 400, hidden_dims = [120,84], output_dim=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        # self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        # self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])
        
        self.classifier = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        # feature = []
        x = self.pool(F.relu(self.conv1(x)))
        # feature.append(x.reshape(len(x),-1))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        # feature.append(x)
        x = F.relu(self.fc1(x))
        # feature.append(x)
        x = F.relu(self.fc2(x))

        # x = F.relu(self.fc3(x))

        # x = F.relu(self.fc4(x))
        # feature.append(x)
        y = self.classifier(x)
        # feature.append(y)
        return x, y
    

class FedAvgNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(FedAvgNetCIFAR, self).__init__()
        self.conv2d_1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(4096, 512)
        self.fc = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        z = self.relu(self.linear_1(x))
        x = self.fc(z)

       
        return x

