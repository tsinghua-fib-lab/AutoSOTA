import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.models as models


from utils import add_artificial_bias, create_balanced_subset, dilate_masks_torch
from utils_fast import add_artificial_bias_fast
# Use fast GPU-based bias function
add_artificial_bias = add_artificial_bias_fast
from x_resnet import xfixup_resnet50

# Compatible sdpa_kernel import (PyTorch 2.1+)
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
except (ImportError, ModuleNotFoundError):
    try:
        from torch.backends.cuda import sdp_kernel, SDPBackend
        # In PyTorch 2.1, sdp_kernel takes booleans, not SDPBackend
        # For compatibility, create a wrapper
        from contextlib import contextmanager
        @contextmanager
        def sdpa_kernel(backend):
            if backend == SDPBackend.MATH:
                yield
            else:
                yield
    except ImportError:
        # Fallback: no-op context manager
        from contextlib import contextmanager
        @contextmanager
        def sdpa_kernel(backend=None):
            yield
        class SDPBackend:
            MATH = "math"



# ----- Paths -----
DATA_DIR = '/tmp/ISIC2020_2'  # Local copy for faster I/O  # e.g., './dataset'
STORE_DIR = '/models/50_bias' # ''(default), '50_bias', 'vit'
XRESNET_PATH = '/models/xfixup_resnet50_model_best.pth.tar'
BATCH_SIZE = 128
NUM_EPOCHS = 20
NUM_WORKERS = 0  # Set to 0 to avoid NFS multiprocessing issues
RUNS = 1  # Reduced from 5 for reproduction
DEVICE = 'cuda:0'
MODE = 'presence_absence_debias' # 'default', 'presence_debias', 'presence_absence_debias'
MODEL = 'xresnet50' # 'xresnet50', 'vit_b_16'
train_wo_bias = False # set true if no training bias should be available
if train_wo_bias:
    print('Training without bias enabled!')

random_seed = 0
torch.manual_seed(random_seed)

# ----- ImageNet Normalization -----
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# ----- Transforms -----
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ]),
}

# ----- Datasets and Loaders -----
image_datasets = {
    x: datasets.ImageFolder(os.path.join(DATA_DIR, x), transform=data_transforms[x])
    for x in ['train', 'val']
}

for x in ['train', 'val']:
    print(x, len(image_datasets[x]))
for x in ['train', 'val']:
    image_datasets[x] = create_balanced_subset(image_datasets[x], targets=image_datasets[x].targets)

for x in ['train', 'val']:
    print(x, len(image_datasets[x]))

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x=='train'), num_workers=NUM_WORKERS)
    for x in ['train', 'val']
}

class_names = image_datasets['train'].dataset.classes  # ['benign', 'malignant']

# ----- Training Loop -----
def train_model(model, dataloaders, criterion, optimizer, attribution_prior_weight, best_acc, num_epochs=NUM_EPOCHS):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_current = 0

    if train_wo_bias:
        class_bias_train = {
            0: 0.0,  # 0% of benign get a patch
            1: 0.0   # 0% of malignant get a patch
        }
    else:
        class_bias_train = {
            0: 1.0,  # 100% of benign get a patch
            1: 0.0   # 0% of malignant get a patch
        }

    class_bias_inverse = {
        0: 0.0,  # 0% of benign get a patch
        1: 1.0   # 100% of malignant get a patch
    }

    class_bias_off = {
        0: 0.0,  # 0% of benign get a patch
        1: 0.0   # 0% of malignant get a patch
    }
    
    print('attribution_prior_weight:', attribution_prior_weight)
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 20)
        #nr_benign = 0
        #nr_malignant = 0
        for phase in ['train_', 'val_trainbias', 'val_inversebias', 'val_nobias']:
            print('Starting', phase)
            model.train() if phase == 'train_' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase.split("_")[0]]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                labels_one_hot = F.one_hot(labels, num_classes=2).float()

                if phase == 'train_':
                    inputs, patch_segmentation = add_artificial_bias(inputs, labels, class_bias_train)
                if phase == 'val_trainbias':
                   inputs, patch_segmentation = add_artificial_bias(inputs, labels, class_bias_train)
                if phase == 'val_inversebias':
                   inputs, patch_segmentation = add_artificial_bias(inputs, labels, class_bias_inverse)
                if phase == 'val_nobias':
                    inputs, patch_segmentation = add_artificial_bias(inputs, labels, class_bias_off)
                    
                patch_segmentation = dilate_masks_torch(patch_segmentation)
                    
                optimizer.zero_grad()

                with sdpa_kernel(SDPBackend.MATH):
                    inputs.requires_grad = True
                    outputs = model(inputs)
                    if MODE == 'presence_debias' or MODE == 'presence_absence_debias':
                        target_outputs = torch.gather(outputs, 1, labels.unsqueeze(-1))
                        gradients = torch.autograd.grad(torch.unbind(target_outputs), inputs, create_graph=True)[0] # set to false if attribution is only used for evaluation
                        gradients = inputs * gradients
                        attribution_inside1 = ((gradients.abs().sum(dim=1, keepdim=True) * patch_segmentation).sum()) / (patch_segmentation.sum() + 1e-5) # normalize by number of active masks
                            
                        loss_attribution_prior1 = attribution_inside1
                        
                        if MODE == 'presence_absence_debias':
                            labels_flipped = 1 - labels
                            target_outputs = torch.gather(outputs, 1, labels_flipped.unsqueeze(-1))
                            gradients = torch.autograd.grad(torch.unbind(target_outputs), inputs, create_graph=True)[0] # set to false if attribution is only used for evaluation
                            gradients = inputs * gradients
                            attribution_inside2 = ((gradients.abs().sum(dim=1, keepdim=True) * patch_segmentation).sum()) / (patch_segmentation.sum() + 1e-5) # normalize by number of active masks
                                
                            loss_attribution_prior2 = attribution_inside2
                        else:
                            loss_attribution_prior2 = loss_attribution_prior1
                        
                        loss_attribution_prior = (loss_attribution_prior1 + loss_attribution_prior2) / 2
                    else:
                        loss_attribution_prior = 0

                loss_classification = criterion(outputs, labels_one_hot)
                loss = loss_classification + attribution_prior_weight*loss_attribution_prior #10000.0*loss_attribution_prior
                preds = torch.argmax(outputs, dim=1)

                if phase == 'train_':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase.split("_")[0]].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase.split("_")[0]].dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Save best model
            if phase == 'val_nobias' and epoch_acc > best_acc_current:
                best_acc_current = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"New best model saved (val acc: {best_acc_current:.4f})")

    print(f"\nBest validation accuracy: {best_acc_current:.4f}")
    model.load_state_dict(best_model_wts)
    return model, best_acc_current

# ----- Train -----
for seed in range(RUNS): # different training runs
    
    best_acc = 0.0

    for attribution_prior_weight in [1, 10, 100, 1000, 10000]:
        torch.manual_seed(seed)

        # ----- Model -----
        if MODEL == 'xresnet50':
            model = xfixup_resnet50()
            checkpoint = torch.load(XRESNET_PATH)
            new_state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                new_key = k.replace("module.", "", 1) if k.startswith("module.") else k
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2) # binary classification

        elif MODEL == 'vit_b_16':
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            num_ftrs = model.heads.head.in_features
            model.heads.head = nn.Linear(num_ftrs, 2) # binary classification


        model = model.to(DEVICE)

        # ----- Loss + Optimizer -----
        pos_weight = torch.ones([2]).to(DEVICE)  # All weights are equal to 1
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

        model, best_acc_current = train_model(model, dataloaders, criterion, optimizer, attribution_prior_weight, best_acc)

        if best_acc_current >= best_acc:
            best_acc = best_acc_current
            best_model_wts = copy.deepcopy(model.state_dict())

        
    model.load_state_dict(best_model_wts)
    if train_wo_bias:
        torch.save(model.state_dict(), os.path.join(STORE_DIR, f'model_no_train_bias_seed{seed}_mode_' + MODE + '.pth'))
    else:
        torch.save(model.state_dict(), os.path.join(STORE_DIR, f'model_seed{seed}_mode_' + MODE + '.pth'))