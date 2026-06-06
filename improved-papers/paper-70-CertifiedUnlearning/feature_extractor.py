import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import os
from tqdm import tqdm

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# transform = transforms.Compose([
#     transforms.Resize(224),  
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
# ])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# feature_extractor = nn.Sequential(*list(model.children())[:-1])  
# feature_extractor = feature_extractor.to(device)
# feature_extractor.eval()
# def extract_features(dataloader):
#     all_features = []
#     all_labels = []
    
#     with torch.no_grad():
#         for images, labels in tqdm(dataloader, desc="Extracting features"):
#             images = images.to(device)
#             features = feature_extractor(images)
#             features = features.reshape(features.size(0), -1)
            
#             all_features.append(features.cpu())
#             all_labels.append(labels)
    
    
#     all_features = torch.cat(all_features, dim=0)
#     all_labels = torch.cat(all_labels, dim=0)
    
#     return all_features, all_labels


# print("Extracting features from training set...")
# train_features, train_labels = extract_features(trainloader)
# print("Extracting features from test set...")
# test_features, test_labels = extract_features(testloader)


# os.makedirs('cifar10_features', exist_ok=True)


# train_data = {
#     'features': train_features,
#     'labels': train_labels
# }

# test_data = {
#     'features': test_features,
#     'labels': test_labels
# }


# print("Saving features to disk...")
# torch.save(train_data, 'cifar10_features/cifar10_train_features.pt')
# torch.save(test_data, 'cifar10_features/cifar10_test_features.pt')

# print("Feature extraction complete!")
# print(f"Train features shape: {train_features.shape}")
# print(f"Test features shape: {test_features.shape}")



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize(224),  
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  
])


trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
feature_extractor = nn.Sequential(*list(model.children())[:-1])  
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

def extract_features(dataloader):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            features = feature_extractor(images)
            features = features.reshape(features.size(0), -1)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_features, all_labels


print("Extracting features from training set...")
train_features, train_labels = extract_features(trainloader)
print("Extracting features from test set...")
test_features, test_labels = extract_features(testloader)


os.makedirs('cifar100_features', exist_ok=True)


train_data = {
    'features': train_features,
    'labels': train_labels
}

test_data = {
    'features': test_features,
    'labels': test_labels
}


print("Saving features to disk...")
torch.save(train_data, 'cifar100_features/cifar100_train_features.pt')  
torch.save(test_data, 'cifar100_features/cifar100_test_features.pt')

print("Feature extraction complete!")
print(f"Train features shape: {train_features.shape}")
print(f"Test features shape: {test_features.shape}")