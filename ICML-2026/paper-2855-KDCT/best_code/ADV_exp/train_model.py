import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import os


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock,num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  num_blocks[0], stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512, num_classes) #512

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class ResNet_Fea(nn.Module):
    def __init__(self, ResidualBlock,num_blocks, num_classes=10):
        super(ResNet_Fea, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  num_blocks[0], stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512, num_classes) #512

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        _ = self.fc(out)
        return out

def ResNet18():
    return ResNet(ResidualBlock, [2,2,2,2])

def ResNet34():
    return ResNet(ResidualBlock, [3,4,6,3])

def ResNet18_Fea():
    return ResNet_Fea(ResidualBlock, [2,2,2,2])

def ResNet34_Fea():
    return ResNet(ResidualBlock, [3,4,6,3])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter setting
EPOCH = 150   #Number of times to traverse the dataset
pre_epoch = 0  #Defines the number of times a dataset has been traversed
BATCH_SIZE = 128      #Batch size (batch size)
LR = 0.01        #learning rate 

def parse_train_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--outf', default='./Res18_model', help='folder to output images and model checkpoints')
    return parser.parse_args()


def build_cifar10_loaders(batch_size=128):
    transform_test = transforms.Compose([transforms.ToTensor(),])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return trainloader, testloader

# train
if __name__ == "__main__":
    args = parse_train_args()
    trainloader, testloader = build_cifar10_loaders(BATCH_SIZE)

    # define-ResNet
    #net = ResNet34().to(device)
    net = ResNet18().to(device)

    # Define loss functions and optimizations
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) 

    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    best_acc = 85  #2 initialize best test accuracy
    print("Start Training, Resnet-18!")  
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    # To prepare data
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # Each training batch prints loss and accuracy once
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # Test the accuracy rate after each training epoch
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # The class with the highest score (outputs.data index)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('The test classification accuracy is：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    # Write the results of each test in real time to acc.txt file
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # Record the best test classification accuracy and write it to the BEST_ACC.TXT file
                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)
