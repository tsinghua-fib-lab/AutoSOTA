# Script to load the dataloaders. 
from PIL import Image
import torch
import numpy as np
import torchvision
import os
import glob
import json
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import train_test_split

from config.config import DATA_DIR, EMBEDDINGS_DIR  # , GLOBAL_SEED

# torch.manual_seed(GLOBAL_SEED)
# np.random.seed(GLOBAL_SEED)


def data_path(*parts):
    return os.path.join(DATA_DIR, *parts)


def get_dataloaders(dataset_name, batch_size, transformations, only_test=False, valid_split=False, shuffle=True):
    dataset_functions = {
        'mnist': get_mnist_dataloaders,
        'cifar10': get_cifar10_dataloaders,
        'cifar100': get_cifar100_dataloaders,
        'imagenet1kval': get_imagenet1kval_dataloaders,
        'food101': get_food101_dataloaders,
        'flowers102': get_flowers102_dataloaders,
        'oxfordpets': get_oxfordpets_dataloaders,
        'eurosat': get_eurosat_dataloaders,
        'dtd': get_dtd_dataloaders,
        'places365': get_places365_dataloaders,
        'resisc45': get_resisc45_dataloaders,
        'flickr30k': get_flickr30k_dataloaders,  # Added flickr30k support
        'coco': get_coco_dataloaders,  # Added COCO support
        'ham10000': get_ham10000_dataloaders, # Added HAM10000 support
    }
    
    # Find matching dataset function
    dataset_function = None
    for dataset_key, function in dataset_functions.items():
        if dataset_name == dataset_key:
            dataset_function = function
            break

    # If dataset name is for CIFAR-C datasets, handle separately
    if dataset_name.startswith('cifar10-c') or dataset_name.startswith('cifar100-c'):
        corruption, severity = dataset_name.split('-')[2:]        
        assert corruption in [
            "pixelate",
            "gaussian_noise",
            "shot_noise",
            "jpeg_compression",
            "elastic_transform",
            "zoom_blur",
            "contrast",
            "speckle_noise",
            "defocus_blur",
            "impulse_noise",
            "snow",
            "saturate",
            "fog",
            "frost",
            "glass_blur",
            "gaussian_blur",
            "motion_blur",
            "brightness",
            "spatter"
        ]
        assert severity in ['1', '2', '3', '4', '5']
        if dataset_name.startswith('cifar10-c'):
            dataset_function = lambda batch_size, transformations, only_test, valid_split, shuffle: get_cifar10_c_dataloaders(
                corruption, int(severity), batch_size, transformations, only_test, valid_split, shuffle)
        elif dataset_name.startswith('cifar100-c'):
            dataset_function = lambda batch_size, transformations, only_test, valid_split, shuffle: get_cifar100_c_dataloaders(
                corruption, int(severity), batch_size, transformations, only_test, valid_split, shuffle)
    
    if dataset_function is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    train_dataloader, valid_dataloader, test_dataloader = dataset_function(batch_size, transformations, only_test, valid_split, shuffle)

    return train_dataloader, valid_dataloader, test_dataloader

#### Individual datasets

def get_cifar_c_datasets(dataset, corruption, severity, transformations):
    assert dataset in ['cifar10', 'cifar100']

    corrupt_path = data_path(f'{dataset}-c', corruption + '.npy')
    labels_path = data_path(f'{dataset}-c', 'labels.npy')

    # Load all corruptions (shape: [50000 * 15, 32, 32, 3])
    all_data = np.load(corrupt_path)
    print(all_data.shape)
    all_labels = np.load(labels_path)

    # Choose severity block
    num_images = 10000
    start = (severity - 1) * num_images
    end = severity * num_images

    images = all_data[start:end]
    labels = all_labels[start:end]

    # Numpy to pil images
    images = [Image.fromarray(img) for img in images]

    # Apply transforms manually
    images = torch.stack([transformations(img) for img in images])
    labels = torch.tensor(labels)

    return TensorDataset(images, labels)

def get_cifar10_c_dataloaders(corruption, severity, batch_size, transformations, only_test=False, valid_split=False, shuffle=True):
    # for train, provide the normal cifar training set
    if only_test:
        train_dataloader, valid_dataloader = create_dummy_dataloaders(
            batch_size, transformations)
    else:
        train_dataloader, valid_dataloader, _ = get_cifar10_dataloaders(batch_size,
            transformations, only_test=False, valid_split=valid_split,
            shuffle=shuffle)
    # Load the test dataloder
    testset = get_cifar_c_datasets('cifar10', corruption, severity, transformations)
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader, valid_dataloader, test_dataloader

def get_cifar100_c_dataloaders(corruption, severity, batch_size, transformations, only_test=False, valid_split=False, shuffle=True):
    # for train, provide the normal cifar training set
    if only_test:
        train_dataloader, valid_dataloader = create_dummy_dataloaders(
            batch_size, transformations)
    else:
        train_dataloader, valid_dataloader, _ = get_cifar100_dataloaders(batch_size,
            transformations, only_test=False, valid_split=valid_split,
            shuffle=shuffle)
    # Load the test dataloder
    testset = get_cifar_c_datasets('cifar100', corruption, severity, transformations)
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader, valid_dataloader, test_dataloader


def get_mnist_datasets(transformations, only_test=False, valid_split=False):
    """
    Load MNIST datasets with train/valid/test splits.
    
    MNIST has 10 digit classes (0-9) with grayscale 28x28 images.
    Training set: 60,000 images, Test set: 10,000 images
    We split the training set into 80% train and 20% validation.
    
    Note: MNIST images are converted from grayscale to RGB to be compatible with models expecting 3-channel input.
    
    Returns:
        tuple: (trainset, validset, testset) or (None, None, testset) if only_test=True
    """
    # Handle SSL certificate issues that can occur with MNIST download
    import ssl
    import torchvision.transforms as transforms
    ssl._create_default_https_context = ssl._create_unverified_context
    
    print("Loading MNIST dataset...")
    
    # Create MNIST-specific transform that converts grayscale to RGB first
    # MNIST is grayscale, but most models expect RGB (3 channels)
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to tensor and scale to [0,1]
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert grayscale (1,H,W) to RGB (3,H,W)
        transforms.ToPILImage(),  # Convert back to PIL for compatibility with other transforms
        transformations  # Apply the provided transformations
    ])
    
    if only_test:
        testset = torchvision.datasets.MNIST(root=DATA_DIR, train=False,
                                             download=True, transform=mnist_transform)
        return None, None, testset
    
    # Load the training and test sets applying the MNIST-specific transformations
    trainvalidset = torchvision.datasets.MNIST(root=DATA_DIR, train=True,
                                               download=True, transform=mnist_transform)    
    testset = torchvision.datasets.MNIST(root=DATA_DIR, train=False,
                                         download=True, transform=mnist_transform)

    if valid_split:
        # Split train/valid with 80/20 ratio
        trainvalid_ratio = 0.8
        indices = list(range(len(trainvalidset)))
        train_indices, valid_indices = train_test_split(
            indices, 
            test_size=1-trainvalid_ratio, 
            random_state=42, 
            stratify=[trainvalidset[i][1] for i in indices]
        )
        trainset = torch.utils.data.Subset(trainvalidset, train_indices)
        validset = torch.utils.data.Subset(trainvalidset, valid_indices)
        return trainset, validset, testset
    else:
        return trainvalidset, None, testset

def get_mnist_dataloaders(batch_size, transformations, only_test=False, valid_split=False, shuffle=True):
    """
    Load MNIST dataset with train/valid/test splits.
    
    MNIST has 10 digit classes (0-9) with grayscale 28x28 images.
    Training set: 60,000 images, Test set: 10,000 images
    We split the training set into 80% train and 20% validation.
    
    Note: MNIST images are converted from grayscale to RGB to be compatible with models expecting 3-channel input.
    """
    # Get datasets
    trainset, validset, testset = get_mnist_datasets(transformations, only_test, valid_split)
    
    if only_test:
        dummy_train_dataloader, dummy_valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
        print(f"MNIST dataloaders created (train and valid are dummy, test contains actual data)")
        return dummy_train_dataloader, dummy_valid_dataloader, test_dataloader
    
    # Create dataloaders
    if valid_split:
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
        valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=shuffle)
        print(f"MNIST dataloaders created - Train: {len(trainset)}, Valid: {len(validset)}, Test: {len(testset)}")
    else:
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
        _, valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
        print(f"MNIST dataloaders created - Train: {len(trainset)}, Valid: 0, Test: {len(testset)}")

    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
    
    return train_dataloader, valid_dataloader, test_dataloader

def get_cifar10_datasets(transformations, only_test=False, valid_split=False):
    """
    Load CIFAR-10 datasets with train/valid/test splits.
    
    Returns:
        tuple: (trainset, validset, testset) or (None, None, testset) if only_test=True
    """
    if only_test:
        testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                                               download=True, transform=transformations)
        return None, None, testset
    
    # Load the training and test sets applying the normalization
    trainvalidset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                                            download=True, transform=transformations)    
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                                        download=True, transform=transformations)

    if valid_split:
        trainvalid_ratio = 0.8
        indices = list(range(len(trainvalidset)))
        train_indices, valid_indices = train_test_split(
            indices, 
            test_size=1-trainvalid_ratio, 
            random_state=42, 
            stratify=[trainvalidset[i][1] for i in indices]
        )
        trainset = torch.utils.data.Subset(trainvalidset, train_indices)
        validset = torch.utils.data.Subset(trainvalidset, valid_indices)
        return trainset, validset, testset
    else:
        return trainvalidset, None, testset

def get_cifar10_dataloaders(batch_size, transformations, only_test=False, valid_split=False, shuffle=True):
    """
    Load CIFAR-10 dataset with train/valid/test splits.
    
    CIFAR-10 has 10 classes with 32x32 color images.
    Training set: 50,000 images, Test set: 10,000 images
    We split the training set into 80% train and 20% validation.
    """
    # Get datasets
    trainset, validset, testset = get_cifar10_datasets(transformations, only_test, valid_split)
    
    if only_test:
        dummy_train_dataloader, dummy_valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
        print(f"CIFAR-10 dataloaders created (train and valid are dummy, test contains actual data)")
        return dummy_train_dataloader, dummy_valid_dataloader, test_dataloader
    
    # Create dataloaders
    if valid_split:
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
        valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=shuffle)
        print(f"CIFAR-10 dataloaders created - Train: {len(trainset)}, Valid: {len(validset)}, Test: {len(testset)}")
    else:
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
        _, valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
        print(f"CIFAR-10 dataloaders created - Train: {len(trainset)}, Valid: 0, Test: {len(testset)}")

    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
    
    return train_dataloader, valid_dataloader, test_dataloader

def get_cifar100_datasets(transformations, only_test=False, valid_split=False):
    """
    Load CIFAR-100 datasets with train/valid/test splits.
    
    CIFAR-100 has 100 classes with 32x32 color images.
    Training set: 50,000 images, Test set: 10,000 images
    We split the training set into 80% train and 20% validation.
    
    Returns:
        tuple: (trainset, validset, testset) or (None, None, testset) if only_test=True
    """
    if only_test:
        testset = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False,
                                                download=True, transform=transformations)
        return None, None, testset
    
    # Load the training and test sets applying the transformations
    trainvalidset = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True,
                                            download=True, transform=transformations)    
    testset = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False,
                                        download=True, transform=transformations)

    if valid_split:
        trainvalid_ratio = 0.8
        indices = list(range(len(trainvalidset)))
        train_indices, valid_indices = train_test_split(
            indices, 
            test_size=1-trainvalid_ratio, 
            random_state=42, 
            stratify=[trainvalidset[i][1] for i in indices]
        )
        trainset = torch.utils.data.Subset(trainvalidset, train_indices)
        validset = torch.utils.data.Subset(trainvalidset, valid_indices)
        return trainset, validset, testset
    else:
        return trainvalidset, None, testset

def get_cifar100_dataloaders(batch_size, transformations, only_test=False, valid_split=False, shuffle=True):
    """
    Load CIFAR-100 dataset with train/valid/test splits.
    
    CIFAR-100 has 100 classes with 32x32 color images.
    Training set: 50,000 images, Test set: 10,000 images
    We split the training set into 80% train and 20% validation.
    """
    # Get datasets
    trainset, validset, testset = get_cifar100_datasets(transformations, only_test, valid_split)
    
    if only_test:
        dummy_train_dataloader, dummy_valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
        print(f"CIFAR-100 dataloaders created (train and valid are dummy, test contains actual data)")
        return dummy_train_dataloader, dummy_valid_dataloader, test_dataloader
    
    # Create dataloaders
    if valid_split:
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
        valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=shuffle)
        print(f"CIFAR-100 dataloaders created - Train: {len(trainset)}, Valid: {len(validset)}, Test: {len(testset)}")
    else:
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
        _, valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
        print(f"CIFAR-100 dataloaders created - Train: {len(trainset)}, Valid: 0, Test: {len(testset)}")

    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
    
    return train_dataloader, valid_dataloader, test_dataloader

def get_imagenet1kval_datasets(transformations, only_test=False, valid_split=False):
    """
    Load ImageNet-1K validation datasets with stratified train/valid/test splits.
    
    The ImageNet-1K validation set has 1000 classes with 50 images per class.
    We split each class as: 40 train, 5 valid, 5 test (0.8/0.1/0.1 ratio).
    
    Returns:
        tuple: (trainset, validset, testset) or (None, None, testset) if only_test=True
    """
    import torchvision.datasets as datasets
    from torch.utils.data import Subset
    import numpy as np
    
    valdir = data_path('imagenet1k', 'val')
    print(f"Loading ImageNet-1K validation dataset from {valdir}")
    full_dataset = datasets.ImageFolder(valdir, transformations)
    targets = np.array([full_dataset.targets[i] for i in range(len(full_dataset))])
    print(f"Total samples: {len(full_dataset)}")
    print(f"Number of classes: {len(full_dataset.classes)}")
    
    if only_test:
        return None, None, full_dataset
    
    train_indices = []
    valid_indices = []
    test_indices = []
    
    for class_idx in range(len(full_dataset.classes)):
        class_indices = np.where(targets == class_idx)[0]
        if len(class_indices) != 50:
            print(f"Warning: Class {class_idx} has {len(class_indices)} samples instead of 50")
        # np.random.seed(GLOBAL_SEED)
        np.random.shuffle(class_indices)
        
        if valid_split:
            train_indices.extend(class_indices[:40])
            valid_indices.extend(class_indices[40:45])
            test_indices.extend(class_indices[45:50])
        else:
            train_indices.extend(class_indices[:45])
            test_indices.extend(class_indices[45:])

    if valid_split:
        print(f"Split sizes - Train: {len(train_indices)}, Valid: {len(valid_indices)}, Test: {len(test_indices)}")
        trainset = Subset(full_dataset, train_indices)
        validset = Subset(full_dataset, valid_indices)
        testset = Subset(full_dataset, test_indices)
        return trainset, validset, testset
    else:
        print(f"Split sizes - Train: {len(train_indices)}, Test: {len(test_indices)}")
        trainset = Subset(full_dataset, train_indices)
        testset = Subset(full_dataset, test_indices)
        return trainset, None, testset

def get_imagenet1kval_dataloaders(batch_size, transformations, only_test=False, valid_split=False, shuffle=True):
    """
    Load ImageNet-1K validation dataset with stratified train/valid/test splits.
    
    The ImageNet-1K validation set has 1000 classes with 50 images per class.
    We split each class as: 40 train, 5 valid, 5 test (0.8/0.1/0.1 ratio).
    """
    # Get datasets
    trainset, validset, testset = get_imagenet1kval_datasets(transformations, only_test, valid_split)
    #Combine train and valid sets into one split and create dummy for valid and test
    # valid_dataloader, test_dataloader = create_dummy_dataloaders(batch_size, transformations)
    # fullset = torch.utils.data.ConcatDataset([trainset, validset, testset]) if validset else\
    #     torch.utils.data.ConcatDataset([trainset, testset])
    # train_dataloader = DataLoader(fullset, batch_size=batch_size, shuffle=shuffle)


    if only_test:
        dummy_train_dataloader, dummy_valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
        print("ImageNet-1K dataloaders created (train and valid are dummy, test contains actual data)")
        # Number of classes in ImageNet-1K validation set
        print(f"ImageNet-1K test set loaded: {len(testset)} samples, {len(testset.classes)} classes")
        return dummy_train_dataloader, dummy_valid_dataloader, test_dataloader
    
    # Create dataloaders
    if valid_split:
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
        valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=shuffle)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
        print(f"ImageNet-1K dataloaders created - Train: {len(trainset)}, Valid: {len(validset)}, Test: {len(testset)}")
    else:
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
        _, valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
        print(f"ImageNet-1K dataloaders created - Train: {len(trainset)}, Valid: 0, Test: {len(testset)}")
       
    
    return train_dataloader, valid_dataloader, test_dataloader

def get_food101_datasets(transformations, only_test=False, valid_split=False):
    """
    Load Food101 datasets with train/valid/test splits.
    Food101 has 101 food categories with 1000 images each.
    
    Returns:
        tuple: (trainset, validset, testset) or (None, None, testset) if only_test=True
    """
    if only_test:
        print("Loading Food101 dataset (test split only)")
        testset = torchvision.datasets.Food101(
            root=DATA_DIR,
            split='test',
            download=True,
            transform=transformations
        )
        print(f"Food101 test set loaded: {len(testset)} samples, {len(testset.classes)} classes")
        return None, None, testset
    else:
        print("Loading Food101 dataset (train/valid/test splits)")
        trainset_full = torchvision.datasets.Food101(
            root=DATA_DIR,
            split='train',
            download=True,
            transform=transformations
        )
        testset = torchvision.datasets.Food101(
            root=DATA_DIR,
            split='test',
            download=True,
            transform=transformations
        )
        
        if valid_split:
            trainvalid_ratio = 0.8
            indices = list(range(len(trainset_full)))
            train_indices, valid_indices = train_test_split(
                indices,
                test_size=1-trainvalid_ratio,
                random_state=42,
                stratify=[trainset_full[i][1] for i in indices]
            )
            trainset = torch.utils.data.Subset(trainset_full, train_indices)
            validset = torch.utils.data.Subset(trainset_full, valid_indices)
            return trainset, validset, testset
        else:
            return trainset_full, None, testset

def get_food101_dataloaders(batch_size, transformations, only_test=False, valid_split=False, shuffle=True):
    """
    Load Food101 dataset with train/valid/test splits.
    Food101 has 101 food categories with 1000 images each.
    """
    # Get datasets
    trainset, validset, testset = get_food101_datasets(transformations, only_test, valid_split)
    
    if only_test:
        dummy_train_dataloader, dummy_valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
        print("Food101 dataloaders created (train and valid are dummy, test contains actual data)")
        return dummy_train_dataloader, dummy_valid_dataloader, test_dataloader
    else:
        # Create dataloaders
        if valid_split:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
            valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=shuffle)
            print(f"Food101 dataloaders created - Train: {len(trainset)}, Valid: {len(validset)}, Test: {len(testset)}")
        else:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
            _, valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
            print(f"Food101 dataloaders created - Train: {len(trainset)}, Test: {len(testset)} (no separate valid split)")
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
        return train_dataloader, valid_dataloader, test_dataloader

def get_flowers102_datasets(transformations, only_test=False, valid_split=False):
    """
    Load Flowers102 datasets with train/valid/test splits.
    Flowers102 has 102 flower categories with varying numbers of images per class.
    
    Returns:
        tuple: (trainset, validset, testset) or (None, None, testset) if only_test=True
    """
    if only_test:
        print("Loading Flowers102 dataset (test split only)")
        testset = torchvision.datasets.Flowers102(
            root=DATA_DIR,
            split='test',
            download=True,
            transform=transformations
        )
        print(f"Flowers102 test set loaded: {len(testset)} samples, 102 classes")
        return None, None, testset
    else:
        print("Loading Flowers102 dataset (train/valid/test splits)")
        testset = torchvision.datasets.Flowers102(
            root=DATA_DIR,
            split='test',
            download=True,
            transform=transformations
        )
        trainset = torchvision.datasets.Flowers102(
            root=DATA_DIR,
            split='train',
            download=True,
            transform=transformations
        )
        validset = torchvision.datasets.Flowers102(
            root=DATA_DIR,
            split='val',
            download=True,
            transform=transformations
        )
        
        if valid_split:            
            return trainset, validset, testset
        else:
            # Join train and valid sets into one full training set
            trainset_full = torch.utils.data.ConcatDataset([trainset, validset])
            return trainset_full, None, testset

def get_flowers102_dataloaders(batch_size, transformations, only_test=False, valid_split=False, shuffle=True):
    """
    Load Flowers102 dataset with train/valid/test splits.
    Flowers102 has 102 flower categories with varying numbers of images per class.
    """
    # Get datasets
    trainset, validset, testset = get_flowers102_datasets(transformations, only_test, valid_split)
    
    if only_test:
        dummy_train_dataloader, dummy_valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
        print("Flowers102 dataloaders created (train and valid are dummy, test contains actual data)")
        return dummy_train_dataloader, dummy_valid_dataloader, test_dataloader
    else:
        # Create dataloaders
        if valid_split:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
            valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=shuffle)
            print(f"Flowers102 dataloaders created - Train: {len(trainset)}, Valid: {len(validset)}, Test: {len(testset)}")
        else:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
            _, valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
            print(f"Flowers102 dataloaders created - Train: {len(trainset)}, Test: {len(testset)} (no separate valid split)")
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
        return train_dataloader, valid_dataloader, test_dataloader

def get_oxfordpets_datasets(transformations, only_test=False, valid_split=False):
    """
    Load Oxford-IIIT Pet datasets with train/valid/test splits.
    Oxford Pets has 37 pet categories (25 dog breeds + 12 cat breeds) with varying numbers of images per class.
    
    Returns:
        tuple: (trainset, validset, testset) or (None, None, testset) if only_test=True
    """
    if only_test:
        print("Loading Oxford-IIIT Pet dataset (test split only)")
        testset = torchvision.datasets.OxfordIIITPet(
            root=DATA_DIR,
            split='test',
            download=True,
            transform=transformations
        )
        print(f"Oxford Pets test set loaded: {len(testset)} samples, 37 classes")
        return None, None, testset
    else:
        print("Loading Oxford-IIIT Pet dataset (train/valid/test splits)")
        trainset_full = torchvision.datasets.OxfordIIITPet(
            root=DATA_DIR,
            split='trainval',
            download=True,
            transform=transformations
        )
        testset = torchvision.datasets.OxfordIIITPet(
            root=DATA_DIR,
            split='test',
            download=True,
            transform=transformations
        )
        
        if valid_split:
            trainvalid_ratio = 0.8
            indices = list(range(len(trainset_full)))
            train_indices, valid_indices = train_test_split(
                indices,
                test_size=1-trainvalid_ratio,
                random_state=42,
                stratify=[trainset_full[i][1] for i in indices]
            )
            trainset = torch.utils.data.Subset(trainset_full, train_indices)
            validset = torch.utils.data.Subset(trainset_full, valid_indices)
            return trainset, validset, testset
        else:
            return trainset_full, None, testset

def get_oxfordpets_dataloaders(batch_size, transformations, only_test=False, valid_split=False, shuffle=True):
    """
    Load Oxford-IIIT Pet dataset with train/valid/test splits.
    Oxford Pets has 37 pet categories (25 dog breeds + 12 cat breeds) with varying numbers of images per class.
    """
    # Get datasets
    trainset, validset, testset = get_oxfordpets_datasets(transformations, only_test, valid_split)
    
    if only_test:
        dummy_train_dataloader, dummy_valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
        print("Oxford Pets dataloaders created (train and valid are dummy, test contains actual data)")
        return dummy_train_dataloader, dummy_valid_dataloader, test_dataloader
    else:
        # Create dataloaders
        if valid_split:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
            valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=shuffle)
            print(f"Oxford Pets dataloaders created - Train: {len(trainset)}, Valid: {len(validset)}, Test: {len(testset)}")
        else:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
            _, valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
            print(f"Oxford Pets dataloaders created - Train: {len(trainset)}, Test: {len(testset)} (no separate valid split)")
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
        return train_dataloader, valid_dataloader, test_dataloader

def get_eurosat_datasets(transformations, only_test=False, valid_split=False):
    """
    Load EuroSAT datasets with train/valid/test splits.
    EuroSAT has 10 land use/land cover categories with varying numbers of satellite images per class.
    
    Returns:
        tuple: (trainset, validset, testset) or (None, None, testset) if only_test=True
    """
    print("Loading EuroSAT dataset (train/valid/test splits)")
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    fullset = torchvision.datasets.EuroSAT(
        root=DATA_DIR,
        download=True,
        transform=transformations
    )
    # First split: train-test (90% train, 10% test)
    train_indices, test_indices = train_test_split(
        range(len(fullset)),
        test_size=0.1,
        random_state=42,
        stratify=[fullset[i][1] for i in range(len(fullset))]
    )

    # Second split: train-validation (89% train, 11% validation from the train set)
    train_indices, valid_indices = train_test_split(
        train_indices,
        test_size=0.11,
        random_state=42,
        stratify=[fullset[i][1] for i in train_indices]
    )

    trainset = torch.utils.data.Subset(fullset, train_indices)
    validset = torch.utils.data.Subset(fullset, valid_indices)
    testset = torch.utils.data.Subset(fullset, test_indices)
    print(f"EuroSAT dataset loaded: {len(fullset)} samples, {len(fullset.classes)} classes")

    if only_test:
        print(f"EuroSAT test set loaded: {len(testset)} samples, 10 classes")
        return None, None, testset
    else:
        if valid_split:
            return trainset, validset, testset
        else:
            # Combine train and valid sets into one full training set
            trainset_full = torch.utils.data.ConcatDataset([trainset, validset])
            return trainset_full, None, testset

def get_eurosat_dataloaders(batch_size, transformations, only_test=False, valid_split=False, shuffle=True):
    """
    Load EuroSAT dataset with train/valid/test splits.
    EuroSAT has 10 land use/land cover categories with varying numbers of satellite images per class.
    """
    # Get datasets
    trainset, validset, testset = get_eurosat_datasets(transformations, only_test, valid_split)
    
    if only_test:
        dummy_train_dataloader, dummy_valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
        print("EuroSAT dataloaders created (train and valid are dummy, test contains actual data)")
        return dummy_train_dataloader, dummy_valid_dataloader, test_dataloader
    else:
        # Create dataloaders
        if valid_split:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
            valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=shuffle)
            test_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=shuffle)
            print(f"EuroSAT dataloaders created - Train: {len(trainset)}, Valid: {len(validset)}, Test: {len(validset)} (no separate test split)")
        else:
            # Combine train and valid sets into one full training set
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
            _, valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
            test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
            print(f"EuroSAT dataloaders created - Train: {len(trainset)}, Test: {len(testset)} (no separate valid split)")
        return train_dataloader, valid_dataloader, test_dataloader

def get_resisc45_datasets(transformations, only_test=False, valid_split=False):
    """
    Load RESISC45 datasets using split files for train/val/test.
    RESISC45 has 45 scene categories with 700 images per class.
    The split files (resisc45-train.txt, resisc45-val.txt, resisc45-test.txt) in DATA_DIR define the splits.
    
    Returns:
        tuple: (trainset, validset, testset) or (None, None, testset) if only_test=True
    """
    print("Loading RESISC45 dataset using split files")
    try:
        from torchgeo.datasets import RESISC45
    except ImportError:
        raise ImportError("torchgeo is required for RESISC45 dataset. Install it with: pip install torchgeo")
    
    # Load the full dataset (no transforms yet)
    train_dataset = RESISC45(root=DATA_DIR, download=True, split='train')
    val_dataset = RESISC45(root=DATA_DIR, download=True, split='val')
    test_dataset = RESISC45(root=DATA_DIR, download=True, split='test')

    # Wrapper to apply transforms
    class RESISC45Wrapper(torch.utils.data.Dataset):
        def __init__(self, dataset, transforms=None):
            self.dataset = dataset
            self.transforms = transforms
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, i):
            sample = self.dataset[i]
            image = sample['image']
            label = sample['label']
            if self.transforms is not None:
                from torchvision.transforms.functional import to_pil_image
                if isinstance(image, torch.Tensor):
                    if image.max() > 1.0:
                        image = image.float() / 255.0
                    image = to_pil_image(image)
                image = self.transforms(image)
            return image, label
    
    if only_test:
        testset = RESISC45Wrapper(test_dataset, transforms=transformations)
        return None, None, testset
    else:
        trainset = RESISC45Wrapper(train_dataset, transforms=transformations)
        valset = RESISC45Wrapper(val_dataset, transforms=transformations)
        testset = RESISC45Wrapper(test_dataset, transforms=transformations)
        
        if valid_split:
            return trainset, valset, testset
        else:
            return trainset, None, testset

def get_resisc45_dataloaders(batch_size, transformations, only_test=False,  valid_split=False, shuffle=True):
    """
    Load RESISC45 dataset using split files for train/val/test.
    RESISC45 has 45 scene categories with 700 images per class.
    The split files (resisc45-train.txt, resisc45-val.txt, resisc45-test.txt) in DATA_DIR define the splits.
    """
    # Get datasets
    trainset, validset, testset = get_resisc45_datasets(transformations, only_test, valid_split)
    
    if only_test:
        dummy_train_dataloader, dummy_valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
        print("RESISC45 dataloaders created (train and valid are dummy, test contains actual data)")
        return dummy_train_dataloader, dummy_valid_dataloader, test_dataloader
    else:
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
        if valid_split:
            valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=shuffle)
        else:
            _, valid_dataloader = create_dummy_dataloaders(batch_size, transformations)        
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
        print(f"RESISC45 dataloaders created - Train: {len(trainset)}, Valid: {len(validset) if validset else 0}, Test: {len(testset)}")
        return train_dataloader, valid_dataloader, test_dataloader

def get_dtd_datasets(transformations, only_test=False, valid_split=False):
    """
    Load Describable Textures Dataset (DTD) datasets with train/valid/test splits.
    
    Returns:
        tuple: (trainset, validset, testset) or (None, None, testset) if only_test=True
    """
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    print("Loading DTD dataset...")
    trainset = torchvision.datasets.DTD(root=DATA_DIR, split='train', download=True, transform=transformations)
    validset = torchvision.datasets.DTD(root=DATA_DIR, split='val', download=True, transform=transformations)
    testset = torchvision.datasets.DTD(root=DATA_DIR, split='test', download=True, transform=transformations)
    
    if only_test:
        return None, None, testset
    else:
        if valid_split:
            return trainset, validset, testset
        else:
            trainset_full = torch.utils.data.ConcatDataset([trainset, validset])
            return trainset_full, None, testset

def get_dtd_dataloaders(batch_size, transformations, only_test=False, valid_split=False, shuffle=True):
    """
    Load Describable Textures Dataset (DTD) with train/valid/test splits.
    If only_test=True, returns dummy train/valid and real test dataloader.
    If only_test=False, uses official train/val/test splits or full train set with dummy valid depending on valid_split.
    """
    # Get datasets
    trainset, validset, testset = get_dtd_datasets(transformations, only_test, valid_split)
    
    if only_test:
        dummy_train_dataloader, dummy_valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
        print("DTD dataloaders created (train and valid are dummy, test contains actual data)")
        return dummy_train_dataloader, dummy_valid_dataloader, test_dataloader
    else:
        if valid_split:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
            valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=shuffle)
            print(f"DTD dataloaders created - Train: {len(trainset)}, Valid: {len(validset)}, Test: {len(testset)}")
        else:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
            _, valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
            print(f"DTD dataloaders created - Train: {len(trainset)}, Test: {len(testset)} (no separate valid split)")
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
        return train_dataloader, valid_dataloader, test_dataloader

def get_places365_datasets(transformations, only_test=False, valid_split=False):
    """
    Load Places365 datasets with only test split from the validation set.
    If only_test=True, returns dummy train/valid and real test dataloader.
    If only_test=False, splits validation set 80/20 for train/valid and uses valid as test.
    
    Returns:
        tuple: (trainset, validset, testset) or (None, None, testset) if only_test=True
    """
    print("Loading Places365 dataset (validation split)")
    full_dataset = torchvision.datasets.Places365(
        root=DATA_DIR,
        split='val',
        small=True,
        download=False,
        transform=transformations
    )

    # Do a stratified sampling for train/test split
    # Extract labels for stratification
    labels = [full_dataset[i][1] for i in range(len(full_dataset))]

    # Stratified split (e.g., 80% train / 20% test)
    train_idx, test_idx = train_test_split_np(
        range(len(labels)),
        test_size=0.8,# 40 imgs for test, 10 for train per class
        stratify=labels
    )

    # Wrap back into PyTorch Subsets
    trainset = torch.utils.data.Subset(full_dataset, train_idx)
    testset = torch.utils.data.Subset(full_dataset, test_idx)

    if only_test:
        return None, None, testset
    else:
        if valid_split:
            # Again, stratify train split into validset and trainset
            train_idx, valid_idx = train_test_split(
            train_idx,
            test_size=0.1,
            stratify=[labels[i] for i in train_idx],
            random_state=9871  # Change manually for few-shot approach 42 137 823 5619 9871
            )
            trainset = torch.utils.data.Subset(full_dataset, train_idx)
            validset = torch.utils.data.Subset(full_dataset, valid_idx)
            return trainset, validset, validset  # Note: using validset as testset as mentioned in original function
        else:
            return trainset, None, testset

def get_places365_dataloaders(batch_size, transformations, only_test=False, valid_split=False, shuffle=True):
    """
    Load Places365 dataset with train/valid/test splits.
    If only_test=True, returns dummy train/valid and real test dataloader.
    If only_test=False, splits validation set 80/20 for train/valid and uses valid as test.
    """
    # Get datasets
    trainset, validset, testset = get_places365_datasets(transformations, only_test, valid_split)
    
    if only_test:
        dummy_train_dataloader, dummy_valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
        print("Places365 dataloaders created (train and valid are dummy, test contains actual data)")
        return dummy_train_dataloader, dummy_valid_dataloader, test_dataloader
    else:
        # Create dataloaders
        if valid_split:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
            valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=shuffle)
            test_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=shuffle)
            print(f"Places365 dataloaders created - Train: {len(trainset)}, Valid: {len(validset)}, Test: {len(validset)} (no separate test split)")
        else:
            train_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
            _, valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
            test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
            print(f"Places365 dataloaders created - Train: {len(testset)}, Test: {len(testset)} (no separate valid split)")
        return train_dataloader, valid_dataloader, test_dataloader


# Embedding dataloaders
def get_embeddings_dataloaders(dataset_name, model_name, batch_size, class_idx=None, balance_type="undersample", num_workers=4, pin_memory=True):    
    if dataset_name not in ['mnist', 'cifar10', 'cifar100', 'imagenet1kval', 'food101', 'flowers102', 'oxfordpets', 'eurosat', 'dtd', 'places365', 'resisc45']:
        raise NotImplementedError(f'Dataset {dataset_name} not implemented with embedding dataloaders.')

    train_embeddings_path = os.path.join(EMBEDDINGS_DIR, f'{dataset_name}_{model_name}_train.pt')
    train_embeddings_labels_path = os.path.join(EMBEDDINGS_DIR, f'{dataset_name}_{model_name}_train_labels.pt')
    valid_embeddings_path = os.path.join(EMBEDDINGS_DIR, f'{dataset_name}_{model_name}_valid.pt')
    valid_embeddings_labels_path = os.path.join(EMBEDDINGS_DIR, f'{dataset_name}_{model_name}_valid_labels.pt')
    test_embeddings_path = os.path.join(EMBEDDINGS_DIR, f'{dataset_name}_{model_name}_test.pt')
    test_embeddings_labels_path = os.path.join(EMBEDDINGS_DIR, f'{dataset_name}_{model_name}_test_labels.pt')

    # Load the embeddings as tensors with optimized settings
    print('> Loading embeddings into memory...')
    train_embeddings = torch.load(train_embeddings_path, map_location='cpu')
    train_embeddings_labels = torch.load(train_embeddings_labels_path, map_location='cpu')
    valid_embeddings = torch.load(valid_embeddings_path, map_location='cpu')
    valid_embeddings_labels = torch.load(valid_embeddings_labels_path, map_location='cpu')
    test_embeddings = torch.load(test_embeddings_path, map_location='cpu')
    test_embeddings_labels = torch.load(test_embeddings_labels_path, map_location='cpu')

    # Create a mapping of dataset names to their processing functions
    dataset_processors = {
        'mnist': process_mnist_embeddings,
        'cifar10': process_cifar10_embeddings,
        'cifar100': process_cifar100_embeddings,
        'imagenet1kval': process_imagenet1kval_embeddings,
        'food101': process_food101_embeddings,
        'flowers102': process_flowers102_embeddings,
        'oxfordpets': process_oxfordpets_embeddings,
        'eurosat': process_eurosat_embeddings,
        'dtd': process_dtd_embeddings,
        'places365': process_places365_embeddings,
        'resisc45': process_resisc45_embeddings,
    }
    
    # Get the appropriate processing function
    if dataset_name in dataset_processors:
        process_fn = dataset_processors[dataset_name]
        
        # Process all three splits using the same function
        train_embeddings, train_embeddings_labels, _ = process_fn(
            train_embeddings, train_embeddings_labels, class_idx, balance_type)
        
        valid_embeddings, valid_embeddings_labels, _ = process_fn(
            valid_embeddings, valid_embeddings_labels, class_idx, balance_type)
        
        test_embeddings, test_embeddings_labels, _ = process_fn(
            test_embeddings, test_embeddings_labels, class_idx, balance_type)
    else:
        raise NotImplementedError(f'Dataset {dataset_name} not implemented with embedding dataloaders.')
    
    # Load the training and test sets applying the normalization
    trainset = TensorDataset(train_embeddings, train_embeddings_labels)
    validset = TensorDataset(valid_embeddings, valid_embeddings_labels)
    testset = TensorDataset(test_embeddings, test_embeddings_labels)

    # Optimized data loaders with persistent workers and memory pinning
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                                  num_workers=num_workers, pin_memory=pin_memory, 
                                  persistent_workers=True if num_workers > 0 else False)
    valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=False, 
                                  num_workers=num_workers, pin_memory=pin_memory,
                                  persistent_workers=True if num_workers > 0 else False)  
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                                 num_workers=num_workers, pin_memory=pin_memory,
                                 persistent_workers=True if num_workers > 0 else False)
    
    print(f'> DataLoaders created with {num_workers} workers and pin_memory={pin_memory}')
    return train_dataloader, valid_dataloader, test_dataloader

def process_mnist_embeddings(embeddings, embeddings_labels, class_idx=None, balance_type="undersample"):
    """
    Process MNIST embeddings for binary classification of one class vs. the rest.
    
    Args:
        embeddings: Tensor of shape (N, D) with N embeddings of dimension D
        embeddings_labels: Tensor of shape (N,) with original labels (0-9)
        class_idx: Optional integer (0-9) indicating the positive class
        balance_type: How to handle class imbalance ('undersample', 'balanced_loss', or 'none')
        
    Returns:
        embeddings: Processed embeddings (potentially subsampled)
        binary_labels: Binary labels (1 for target class, 0 for others)
        all_indices: All indices from the original data (or subset if undersampled)
    """
    # If no class index is specified, return original labels
    if class_idx is None:
        return embeddings, embeddings_labels, torch.arange(len(embeddings_labels))
    
    # Validate the class index
    if not (0 <= class_idx <= 9):
        raise ValueError(f"Invalid class index: {class_idx}. Must be between 0 and 9.")
    
    # Get MNIST class names for better logging
    mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    print(f"\tConverting MNIST to binary classification (class {class_idx}: digit '{mnist_classes[class_idx]}' vs. rest)")
    
    # Get indices for positive and negative classes
    n = len(embeddings_labels)
    positive_indices = torch.arange(n)[embeddings_labels == class_idx]
    negative_indices = torch.arange(n)[embeddings_labels != class_idx]
    
    # Count samples in each class
    positive_count = len(positive_indices)
    negative_count = len(negative_indices)
    
    print(f"\tPositive samples (digit {class_idx}): {positive_count}")
    print(f"\tNegative samples (other digits): {negative_count}")
    
    if balance_type == "undersample":
        print(f"\tUndersampling the majority class to balance the dataset")
        
        # Find the minority class size
        min_class_size = min(positive_count, negative_count)
        print(f"\t\tMinority class size: {min_class_size}")
        
        # Randomly subsample both classes to the minority size
        positive_indices = positive_indices[torch.randperm(len(positive_indices))][:min_class_size]
        negative_indices = negative_indices[torch.randperm(len(negative_indices))][:min_class_size]
        
        print(f"\t\tAfter undersampling - Positive: {len(positive_indices)}, Negative: {len(negative_indices)}")
        
        # Combine indices
        all_indices = torch.cat([positive_indices, negative_indices])
        
        # Apply undersampling to embeddings and labels
        embeddings = embeddings[all_indices]
        embeddings_labels = embeddings_labels[all_indices]
    elif balance_type == "balanced_loss":
        print(f"\tUsing balanced loss (class weights will be applied during training)")
        all_indices = torch.arange(len(embeddings_labels))
    elif balance_type == "none":
        print(f"\tNo balancing applied")
        all_indices = torch.arange(len(embeddings_labels))
    else:
        raise ValueError(f"Invalid balance_type: {balance_type}. Must be 'undersample', 'balanced_loss', or 'none'.")
    
    # Convert to binary classification: target class = 1, others = 0
    binary_labels = (embeddings_labels == class_idx).long()
    
    return embeddings, binary_labels, all_indices

def process_cifar10_embeddings(embeddings, embeddings_labels, class_idx=None, balance_type="undersample"):
    """
    Process CIFAR-10 embeddings for binary classification of one class vs. the rest.
    
    Args:
        embeddings: Tensor of shape (N, D) with N embeddings of dimension D
        embeddings_labels: Tensor of shape (N,) with original labels (0-9)
        class_idx: Optional integer (0-9) indicating the positive class
        balance_type: How to handle class imbalance ('undersample', 'balanced_loss', or 'none')
        
    Returns:
        embeddings: Processed embeddings (potentially subsampled)
        binary_labels: Binary labels (1 for target class, 0 for others)
        all_indices: All indices from the original data (or subset if undersampled)
    """
    # If no class index is specified, return original labels
    if class_idx is None:
        return embeddings, embeddings_labels, torch.arange(len(embeddings_labels))
    
    # Validate the class index
    if not (0 <= class_idx <= 9):
        raise ValueError(f"Invalid class index: {class_idx}. Must be between 0 and 9.")
    
    # Get CIFAR-10 class names for better logging
    cifar10_classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    print(f"\tConverting CIFAR-10 to binary classification (class {class_idx}: '{cifar10_classes[class_idx]}' vs. rest)")
    
    # Get indices for positive and negative classes
    n = len(embeddings_labels)
    positive_indices = torch.arange(n)[embeddings_labels == class_idx]
    negative_indices = torch.arange(n)[embeddings_labels != class_idx]
    
    # Count samples in each class
    positive_count = len(positive_indices)
    negative_count = len(negative_indices)
    
    print(f"\tPositive samples (class {class_idx}): {positive_count}")
    print(f"\tNegative samples (other classes): {negative_count}")
    
    if balance_type == "undersample":
        print(f"\tUndersampling the majority class to balance the dataset")
        
        # Find the minority class size
        min_class_size = min(positive_count, negative_count)
        print(f"\t\tMinority class size: {min_class_size}")
        
        # Randomly subsample both classes to the minority size
        positive_indices = positive_indices[torch.randperm(len(positive_indices))][:min_class_size]
        negative_indices = negative_indices[torch.randperm(len(negative_indices))][:min_class_size]
        
        print(f"\t\tAfter undersampling - Positive: {len(positive_indices)}, Negative: {len(negative_indices)}")
        
        # Combine indices
        all_indices = torch.cat([positive_indices, negative_indices])
        
        # Apply undersampling to embeddings and labels
        embeddings = embeddings[all_indices]
        embeddings_labels = embeddings_labels[all_indices]
    elif balance_type == "balanced_loss":
        print(f"\tUsing balanced loss (class weights will be applied during training)")
        all_indices = torch.arange(len(embeddings_labels))
    elif balance_type == "none":
        print(f"\tNo balancing applied")
        all_indices = torch.arange(len(embeddings_labels))
    else:
        raise ValueError(f"Invalid balance_type: {balance_type}. Must be 'undersample', 'balanced_loss', or 'none'.")
    
    # Convert to binary classification: target class = 1, others = 0
    binary_labels = (embeddings_labels == class_idx).long()
    
    return embeddings, binary_labels, all_indices

def process_cifar100_embeddings(embeddings, embeddings_labels, class_idx=None, balance_type="undersample"):
    """
    Process CIFAR-100 embeddings for binary classification of one class vs. the rest.
    
    Args:
        embeddings: Tensor of shape (N, D) with N embeddings of dimension D
        embeddings_labels: Tensor of shape (N,) with original labels (0-99)
        class_idx: Optional integer (0-99) indicating the positive class
        balance_type: How to handle class imbalance ('undersample', 'balanced_loss', or 'none')
        
    Returns:
        embeddings: Processed embeddings (potentially subsampled)
        binary_labels: Binary labels (1 for target class, 0 for others)
        all_indices: All indices from the original data (or subset if undersampled)
    """
    # If no class index is specified, return original labels
    if class_idx is None:
        return embeddings, embeddings_labels, torch.arange(len(embeddings_labels))
    
    # Validate the class index
    if not (0 <= class_idx <= 99):
        raise ValueError(f"Invalid class index: {class_idx}. Must be between 0 and 99.")
    
    # Get CIFAR-100 class names for better logging
    cifar100_fine_labels = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]
    
    print(f"\tConverting CIFAR-100 to binary classification (class {class_idx}: '{cifar100_fine_labels[class_idx]}' vs. rest)")
    
    # Get indices for positive and negative classes
    n = len(embeddings_labels)
    positive_indices = torch.arange(n)[embeddings_labels == class_idx]
    negative_indices = torch.arange(n)[embeddings_labels != class_idx]
    
    # Count samples in each class
    positive_count = len(positive_indices)
    negative_count = len(negative_indices)
    
    print(f"\tPositive samples (class {class_idx}): {positive_count}")
    print(f"\tNegative samples (other classes): {negative_count}")
    
    if balance_type == "undersample":
        print(f"\tUndersampling the majority class to balance the dataset")
        
        # Find the minority class size
        min_class_size = min(positive_count, negative_count)
        print(f"\t\tMinority class size: {min_class_size}")
        
        # Randomly subsample both classes to the minority size
        positive_indices = positive_indices[torch.randperm(len(positive_indices))][:min_class_size]
        negative_indices = negative_indices[torch.randperm(len(negative_indices))][:min_class_size]
        
        print(f"\t\tAfter undersampling - Positive: {len(positive_indices)}, Negative: {len(negative_indices)}")
        
        # Combine indices
        all_indices = torch.cat([positive_indices, negative_indices])
        
        # Apply undersampling to embeddings and labels
        embeddings = embeddings[all_indices]
        embeddings_labels = embeddings_labels[all_indices]
    elif balance_type == "balanced_loss":
        print(f"\tUsing balanced loss (class weights will be applied during training)")
        all_indices = torch.arange(len(embeddings_labels))
    elif balance_type == "none":
        print(f"\tNo balancing applied")
        all_indices = torch.arange(len(embeddings_labels))
    else:
        raise ValueError(f"Invalid balance_type: {balance_type}. Must be 'undersample', 'balanced_loss', or 'none'.")
    
    # Convert to binary classification: target class = 1, others = 0
    binary_labels = (embeddings_labels == class_idx).long()
    
    return embeddings, binary_labels, all_indices

def process_imagenet1kval_embeddings(embeddings, embeddings_labels, class_idx=None, balance_type="undersample"):
    """
    Process ImageNet-1K validation embeddings for binary classification of one class vs. the rest.
    
    Args:
        embeddings: Tensor of shape (N, D) with N embeddings of dimension D
        embeddings_labels: Tensor of shape (N,) with original labels (0-999)
        class_idx: Optional integer (0-999) indicating the positive class
        balance_type: How to handle class imbalance ('undersample', 'balanced_loss', or 'none')
        
    Returns:
        embeddings: Processed embeddings (potentially subsampled)
        binary_labels: Binary labels (1 for target class, 0 for others)
        all_indices: All indices from the original data (or subset if undersampled)
    """
    # If no class index is specified, return original labels
    if class_idx is None:
        return embeddings, embeddings_labels, torch.arange(len(embeddings_labels))
    # Raise NotImplementedError if class_idx is not None
    else:
        raise NotImplementedError("ImageNet-1K validation dataset processing is not implemented yet.")

def process_food101_embeddings(embeddings, embeddings_labels, class_idx=None, balance_type="undersample"):
    """
    Process Food101 embeddings for binary classification of one class vs. the rest.
    
    Args:
        embeddings: Tensor of shape (N, D) with N embeddings of dimension D
        embeddings_labels: Tensor of shape (N,) with original labels (0-100)
        class_idx: Optional integer (0-100) indicating the positive class
        balance_type: How to handle class imbalance ('undersample', 'balanced_loss', or 'none')
        
    Returns:
        embeddings: Processed embeddings (potentially subsampled)
        binary_labels: Binary labels (1 for target class, 0 for others)
        all_indices: All indices from the original data (or subset if undersampled)
    """
    # If no class index is specified, return original labels
    if class_idx is None:
        return embeddings, embeddings_labels, torch.arange(len(embeddings_labels))
    # Raise NotImplementedError if class_idx is not None
    else:
        raise NotImplementedError("Food101 dataset processing is not implemented yet.")

def process_flowers102_embeddings(embeddings, embeddings_labels, class_idx=None, balance_type="undersample"):
    """
    Process Flowers102 embeddings for binary classification of one class vs. the rest.
    
    Args:
        embeddings: Tensor of shape (N, D) with N embeddings of dimension D
        embeddings_labels: Tensor of shape (N,) with original labels (0-101)
        class_idx: Optional integer (0-101) indicating the positive class
        balance_type: How to handle class imbalance ('undersample', 'balanced_loss', or 'none')
        
    Returns:
        embeddings: Processed embeddings (potentially subsampled)
        binary_labels: Binary labels (1 for target class, 0 for others)
        all_indices: All indices from the original data (or subset if undersampled)
    """
    # If no class index is specified, return original labels
    if class_idx is None:
        return embeddings, embeddings_labels, torch.arange(len(embeddings_labels))
    # Raise NotImplementedError if class_idx is not None
    else:
        raise NotImplementedError("Flowers102 dataset processing is not implemented yet.")

def process_oxfordpets_embeddings(embeddings, embeddings_labels, class_idx=None, balance_type="undersample"):
    """
    Process Oxford Pets embeddings for binary classification of one class vs. the rest.
    
    Args:
        embeddings: Tensor of shape (N, D) with N embeddings of dimension D
        embeddings_labels: Tensor of shape (N,) with original labels (0-36)
        class_idx: Optional integer (0-36) indicating the positive class
        balance_type: How to handle class imbalance ('undersample', 'balanced_loss', or 'none')
        
    Returns:
        embeddings: Processed embeddings (potentially subsampled)
        binary_labels: Binary labels (1 for target class, 0 for others)
        all_indices: All indices from the original data (or subset if undersampled)
    """
    # If no class index is specified, return original labels
    if class_idx is None:
        return embeddings, embeddings_labels, torch.arange(len(embeddings_labels))
    #Raise NotImplementedError if class_idx is not None
    else:
        raise NotImplementedError("Oxford Pets dataset processing is not implemented yet.")

def process_eurosat_embeddings(embeddings, embeddings_labels, class_idx=None, balance_type="undersample"):
    """
    Process EuroSAT embeddings for binary classification of one class vs. the rest.
    
    Args:
        embeddings: Tensor of shape (N, D) with N embeddings of dimension D
        embeddings_labels: Tensor of shape (N,) with original labels (0-9)
        class_idx: Optional integer (0-9) indicating the positive class
        balance_type: How to handle class imbalance ('undersample', 'balanced_loss', or 'none')
        
    Returns:
        embeddings: Processed embeddings (potentially subsampled)
        binary_labels: Binary labels (1 for target class, 0 for others)
        all_indices: All indices from the original data (or subset if undersampled)
    """
    # If no class index is specified, return original labels
    if class_idx is None:
        return embeddings, embeddings_labels, torch.arange(len(embeddings_labels))
    # Not implemented error if class_idx is not None
    else:
        raise NotImplementedError("EuroSAT dataset processing is not implemented yet.")

def process_resisc45_embeddings(embeddings, embeddings_labels, class_idx=None, balance_type="undersample"):
    """
    Process RESISC45 embeddings for binary classification of one class vs. the rest.
    
    Args:
        embeddings: Tensor of shape (N, D) with N embeddings of dimension D
        embeddings_labels: Tensor of shape (N,) with original labels (0-44)
        class_idx: Optional integer (0-44) indicating the positive class
        balance_type: How to handle class imbalance ('undersample', 'balanced_loss', or 'none')
        
    Returns:
        embeddings: Processed embeddings (potentially subsampled)
        binary_labels: Binary labels (1 for target class, 0 for others)
        all_indices: All indices from the original data (or subset if undersampled)
    """
    # If no class index is specified, return original labels
    if class_idx is None:
        return embeddings, embeddings_labels, torch.arange(len(embeddings_labels))
    
    # Validate the class index
    if not (0 <= class_idx <= 44):
        raise ValueError(f"Invalid class index: {class_idx}. Must be between 0 and 44.")
    # Not implemented error if class_idx is not None
    else:
        raise NotImplementedError("RESISC45 dataset processing is not implemented yet.")

def process_dtd_embeddings(embeddings, embeddings_labels, class_idx=None, balance_type="undersample"):
    """
    Process DTD embeddings for binary classification of one class vs. the rest.
    
    Args:
        embeddings: Tensor of shape (N, D) with N embeddings of dimension D
        embeddings_labels: Tensor of shape (N,) with original labels (0-46)
        class_idx: Optional integer (0-46) indicating the positive class
        balance_type: How to handle class imbalance ('undersample', 'balanced_loss', or 'none')
        
    Returns:
        embeddings: Processed embeddings (potentially subsampled)
        binary_labels: Binary labels (1 for target class, 0 for others)
        all_indices: All indices from the original data (or subset if undersampled)
    """
    # If no class index is specified, return original labels
    if class_idx is None:
        return embeddings, embeddings_labels, torch.arange(len(embeddings_labels))
    # Raise NotImplementedError if class_idx is not None
    else:
        raise NotImplementedError("DTD dataset processing is not implemented yet.")

def process_places365_embeddings(embeddings, embeddings_labels, class_idx=None, balance_type="undersample"):
    """
    Process Places365 embeddings for binary classification of one class vs. the rest.
    
    Args:
        embeddings: Tensor of shape (N, D) with N embeddings of dimension D
        embeddings_labels: Tensor of shape (N,) with original labels (0-364)
        class_idx: Optional integer (0-364) indicating the positive class
        balance_type: How to handle class imbalance ('undersample', 'balanced_loss', or 'none')
        
    Returns:
        embeddings: Processed embeddings (potentially subsampled)
        binary_labels: Binary labels (1 for target class, 0 for others)
        all_indices: All indices from the original data (or subset if undersampled)
    """
    # If no class index is specified, return original labels
    if class_idx is None:
        return embeddings, embeddings_labels, torch.arange(len(embeddings_labels))
    
    # Validate the class index
    if not (0 <= class_idx <= 364):
        raise ValueError(f"Invalid class index: {class_idx}. Must be between 0 and 364.")
    
    print(f"\tConverting Places365 to binary classification (class {class_idx} vs. rest)")
    
    # Get indices for positive and negative classes
    n = len(embeddings_labels)
    positive_indices = torch.arange(n)[embeddings_labels == class_idx]
    negative_indices = torch.arange(n)[embeddings_labels != class_idx]
    
    # Count samples in each class
    positive_count = len(positive_indices)
    negative_count = len(negative_indices)
    
    print(f"\tPositive samples (class {class_idx}): {positive_count}")
    print(f"\tNegative samples (other classes): {negative_count}")
    
    if balance_type == "undersample":
        print(f"\tUndersampling the majority class to balance the dataset")
        
        # Find the minority class size
        min_class_size = min(positive_count, negative_count)
        print(f"\t\tMinority class size: {min_class_size}")
        
        # Randomly subsample both classes to the minority size
        positive_indices = positive_indices[torch.randperm(len(positive_indices))][:min_class_size]
        negative_indices = negative_indices[torch.randperm(len(negative_indices))][:min_class_size]
        
        print(f"\t\tAfter undersampling - Positive: {len(positive_indices)}, Negative: {len(negative_indices)}")
        
        # Combine indices
        all_indices = torch.cat([positive_indices, negative_indices])
        
        # Apply undersampling to embeddings and labels
        embeddings = embeddings[all_indices]
        embeddings_labels = embeddings_labels[all_indices]
    elif balance_type == "balanced_loss":
        print(f"\tUsing balanced loss (class weights will be applied during training)")
        all_indices = torch.arange(len(embeddings_labels))
    elif balance_type == "none":
        print(f"\tNo balancing applied")
        all_indices = torch.arange(len(embeddings_labels))
    else:
        raise ValueError(f"Invalid balance_type: {balance_type}. Must be 'undersample', 'balanced_loss', or 'none'.")
    
    # Convert to binary classification: target class = 1, others = 0
    binary_labels = (embeddings_labels == class_idx).long()
    

    
    return embeddings, binary_labels, all_indices

def get_flickr30k_datasets(transformations, only_test=False, valid_split=False):
    """
    Load Flickr30k dataset using Karpathy splits.
    Test images are defined in data/flickr30k/flickr30k_test_karpathy.txt
    All images are loaded from data/flickr30k/Images.
    Returns:
        tuple: (trainset, validset, testset) or (None, None, testset) if only_test=True
    """
    import glob
    from PIL import Image
    from torch.utils.data import Dataset
    import torch
    import numpy as np
    import os
    from utils.flickr30 import load_captions

    class Flickr30kImageDataset(Dataset):
        def __init__(self, image_paths, image_to_captions, transform=None):
            self.image_paths = image_paths
            self.image_to_captions = image_to_captions
            self.transform = transform
        def __len__(self):
            return len(self.image_paths)
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            # Get image name without .jpg extension
            img_name = os.path.basename(img_path).replace('.jpg', '')
            # Get individual captions for this image as a tuple
            captions_list = self.image_to_captions.get(img_name, [""])
            # Ensure we always have exactly 5 captions (pad with empty strings if needed)
            while len(captions_list) < 5:
                captions_list.append("")
            # Return tuple of captions (limit to first 5)
            captions_tuple = tuple(captions_list[:5])
            return image, captions_tuple

    image_dir = data_path('flickr30k', 'Images')
    test_split_file = data_path('flickr30k', 'flickr30k_test_karpathy.txt')
    
    # Get test image names from Karpathy split
    test_img_names = set()
    if os.path.exists(test_split_file):
        print(f"Loading test split from {test_split_file}")
        with open(test_split_file, 'r', encoding='utf-8') as f:
            # Skip header
            header = f.readline().strip()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                assert len(line.split(".jpg,")) == 2, f"Invalid line format in split file: {line}"
                key, caption = line.split(".jpg,")
                # Remove leading/trailing whitespace and quotes
                caption = caption.strip().strip('"').strip("'").strip()
                test_img_names.add(key + '.jpg')  # Add .jpg extension
        
        print(f"Found {len(test_img_names)} unique test images in Karpathy split")
    else:
        print(f"Warning: Test split file {test_split_file} not found, using random split")
        test_img_names = set()

    # Get all available images
    all_image_paths = sorted(glob.glob(f'{image_dir}/*.jpg'))
    if len(all_image_paths) == 0:
        raise RuntimeError(f"No images found in {image_dir}")
    print(f"Found {len(all_image_paths)} total images in {image_dir}")

    # Separate test and non-test images based on Karpathy split
    test_image_paths = []
    other_image_paths = []
    
    for img_path in all_image_paths:
        img_name = os.path.basename(img_path)
        if img_name in test_img_names:
            test_image_paths.append(img_path)
        else:
            other_image_paths.append(img_path)
    
    print(f"Using Karpathy split: {len(test_image_paths)} test images, {len(other_image_paths)} train/val images")
    
    # If no test split file was found, fall back to random split
    if not test_img_names:
        print("Using random 80/10/10 split instead")
        n = len(all_image_paths)
        indices = np.arange(n)
        # np.random.seed(GLOBAL_SEED)
        np.random.shuffle(indices)
        train_end = int(0.8 * n)
        valid_end = int(0.9 * n)
        train_idx = indices[:train_end]
        valid_idx = indices[train_end:valid_end]
        test_idx = indices[valid_end:]
        
        train_image_paths = [all_image_paths[i] for i in train_idx]
        valid_image_paths = [all_image_paths[i] for i in valid_idx]
        test_image_paths = [all_image_paths[i] for i in test_idx]
    else:
        # Split the remaining images into train/val (80/20 of non-test images)
        n_other = len(other_image_paths)
        if n_other > 0:
            # np.random.seed(GLOBAL_SEED)
            indices = np.arange(n_other)
            np.random.shuffle(indices)
            train_end = int(0.8 * n_other)
            train_idx = indices[:train_end]
            valid_idx = indices[train_end:]
            
            train_image_paths = [other_image_paths[i] for i in train_idx]
            valid_image_paths = [other_image_paths[i] for i in valid_idx]
        else:
            train_image_paths = []
            valid_image_paths = []

    # Load captions and create image-to-captions mapping
    captions_file = data_path('flickr30k', 'captions.txt')
    print(f"Loading captions from {captions_file}")
    
    try:
        caption_pairs = load_captions(captions_file)
        print(f"Loaded {len(caption_pairs)} caption pairs")
        
        # Group captions by image_id
        image_to_captions = {}
        for img_id, caption in caption_pairs:
            if img_id not in image_to_captions:
                image_to_captions[img_id] = []
            image_to_captions[img_id].append(caption)
        
        print(f"Created caption mapping for {len(image_to_captions)} unique images")
        
    except Exception as e:
        print(f"Warning: Could not load captions from {captions_file}: {e}")
        print("Using empty captions for all images")
        image_to_captions = {}

    if only_test:
        testset = Flickr30kImageDataset(test_image_paths, image_to_captions, transform=transformations)
        print(f"Created test dataset with {len(test_image_paths)} images")
        return None, None, testset
    else:
        if valid_split:
            trainset = Flickr30kImageDataset(train_image_paths, image_to_captions, transform=transformations)
            validset = Flickr30kImageDataset(valid_image_paths, image_to_captions, transform=transformations)
            testset = Flickr30kImageDataset(test_image_paths, image_to_captions, transform=transformations)
            print(f"Created datasets - Train: {len(train_image_paths)}, Valid: {len(valid_image_paths)}, Test: {len(test_image_paths)}")
            return trainset, validset, testset
        else:
            # Combine train and valid for training
            all_train_paths = train_image_paths + valid_image_paths
            trainset = Flickr30kImageDataset(all_train_paths, image_to_captions, transform=transformations)
            testset = Flickr30kImageDataset(test_image_paths, image_to_captions, transform=transformations)
            print(f"Created datasets - Train: {len(all_train_paths)}, Test: {len(test_image_paths)} (no separate valid split)")
            return trainset, None, testset

def flickr30k_collate_fn(batch):
    """
    Custom collate function for Flickr30k dataset to properly handle caption tuples.
    
    Args:
        batch: List of (image, caption_tuple) pairs
        
    Returns:
        images: Batched image tensor
        captions: List of caption tuples (one tuple per image in batch)
    """
    images = []
    captions = []
    
    for image, caption_tuple in batch:
        images.append(image)
        captions.append(caption_tuple)
    
    # Stack images into a batch tensor
    images = torch.stack(images, dim=0)
    
    # Keep captions as a list of tuples (don't let default collate transpose them)
    return images, captions

def get_flickr30k_dataloaders(batch_size, transformations, only_test=False, valid_split=False, shuffle=True):
    """
    Load Flickr30k dataset with train/valid/test splits.
    Each image is paired with a tuple of 5 captions per image.
    """
    trainset, validset, testset = get_flickr30k_datasets(transformations, only_test, valid_split)
    # Check the paths of the first 3 images in the testset?
    if testset is not None:
        print("Paths of the first 3 images in the testset:")
        for i in range(min(3, len(testset))):
            print(testset.image_paths[i])
    if only_test:
        dummy_train_dataloader, dummy_valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle, collate_fn=flickr30k_collate_fn)
        print("Flickr30k dataloaders created (train and valid are dummy, test contains actual data)")
        return dummy_train_dataloader, dummy_valid_dataloader, test_dataloader
    else:
        if valid_split:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, collate_fn=flickr30k_collate_fn)
            valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=shuffle, collate_fn=flickr30k_collate_fn)
            print(f"Flickr30k dataloaders created - Train: {len(trainset)}, Valid: {len(validset)}, Test: {len(testset)}")
        else:
            train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, collate_fn=flickr30k_collate_fn)
            _, valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
            print(f"Flickr30k dataloaders created - Train: {len(trainset)}, Test: {len(testset)} (no separate valid split)")
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle, collate_fn=flickr30k_collate_fn)
        return train_dataloader, valid_dataloader, test_dataloader

def create_dummy_dataloaders(batch_size, transformations=None):
    """
    Create dummy empty train and validation dataloaders.
    Used when only_test=True for datasets that typically only provide test data.
    
    Returns:
        tuple: (dummy_train_dataloader, dummy_valid_dataloader)
    """

    if transformations:        
        # Create dummy pil image
        pil_dummy = np.zeros((224, 224, 3), dtype=np.uint8)  # Single dummy RGB image as numpy array
        # Convert to PIL Image
        pil_image = Image.fromarray(pil_dummy)
        # Apply transformations if provided
        dummy_data = transformations(pil_image)
        # Add batch dimension
        dummy_data = dummy_data.unsqueeze(0)  # Shape (1, C, H, W)
        # Ensure it has 3 channels
        if dummy_data.shape[1] != 3:
            raise ValueError("Dummy data must have 3 channels (RGB).")
    else:
        dummy_data = torch.zeros((1, 3, 224, 224))
    dummy_labels = torch.zeros((1,), dtype=torch.long)  # Dummy label tensor

    dummy_trainset = TensorDataset(dummy_data, dummy_labels)
    dummy_validset = TensorDataset(dummy_data, dummy_labels)

    # Create dataloaders
    dummy_train_dataloader = DataLoader(dummy_trainset, batch_size=batch_size, shuffle=False)
    dummy_valid_dataloader = DataLoader(dummy_validset, batch_size=batch_size, shuffle=False)

    return dummy_train_dataloader, dummy_valid_dataloader


# Define NumPy version of train_test_split
def train_test_split_np(indices, test_size=0.2, stratify=None):
    indices = np.array(indices)
    
    if stratify is not None:
        # stratified split
        train_idx, test_idx = [], []
        labels_arr = np.array(stratify)
        for label in np.unique(labels_arr):
            label_mask = np.where(labels_arr == label)[0]
            np.random.shuffle(label_mask)
            n_test = int(len(label_mask) * test_size)
            test_idx.extend(label_mask[:n_test])
            train_idx.extend(label_mask[n_test:])
        return np.array(train_idx), np.array(test_idx)
    else:
        np.random.shuffle(indices)
        n_test = int(len(indices) * test_size)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        return train_idx, test_idx


def get_coco_datasets(transformations, only_test=False, valid_split=False):
    """
    Load COCO dataset using Karpathy splits (test set only).
    
    This dataset only supports the test set and requires only_test=True.
    Images are loaded from data/coco/coco_karpathy_test/
    Metadata is loaded from data/coco/dataset_coco_karpathy_test.json
    
    Args:
        transformations: Image transformations to apply
        only_test: Must be True for COCO (only test set is available)
        valid_split: Ignored for COCO
        
    Returns:
        tuple: (None, None, testset) - train and valid are not available
    """
    import json
    from PIL import Image
    from torch.utils.data import Dataset
    
    if not only_test:
        raise ValueError("COCO dataset only supports test set. Please set only_test=True")
    
    class CocoImageDataset(Dataset):
        def __init__(self, image_paths, image_to_captions, transform=None):
            self.image_paths = image_paths
            self.image_to_captions = image_to_captions
            self.transform = transform
            
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            # Get image name (filename)
            img_name = os.path.basename(img_path)
            # Get captions for this image as a tuple
            captions_list = self.image_to_captions.get(img_name, [""])
            # Ensure we always have exactly 5 captions (pad with empty strings if needed)
            while len(captions_list) < 5:
                captions_list.append("")
            # Return tuple of captions (limit to first 5)
            captions_tuple = tuple(captions_list[:5])
            return image, captions_tuple
    
    image_dir = data_path('coco', 'coco_karpathy_test')
    json_file = data_path('coco', 'dataset_coco_karpathy_test.json')
    
    # Load metadata from JSON
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"COCO metadata file not found: {json_file}")
    
    print(f"Loading COCO metadata from {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Build image-to-captions mapping
    image_to_captions = {}
    for img_data in data['images']:
        filename = img_data['filename']
        captions = []
        for sentence in img_data['sentences']:
            captions.append(sentence['raw'].strip())
        # Limit to first 5 captions only
        image_to_captions[filename] = captions[:5]
    
    print(f"Loaded {len(image_to_captions)} images with captions from JSON")
    
    # Get all available images from directory
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"COCO image directory not found: {image_dir}")
    
    all_image_paths = sorted(glob.glob(f'{image_dir}/*.jpg'))
    if len(all_image_paths) == 0:
        raise RuntimeError(f"No images found in {image_dir}")
    print(f"Found {len(all_image_paths)} total images in {image_dir}")
    
    # Filter to only include images that have metadata
    test_image_paths = []
    for img_path in all_image_paths:
        img_name = os.path.basename(img_path)
        if img_name in image_to_captions:
            test_image_paths.append(img_path)
    
    print(f"Using {len(test_image_paths)} images with available captions")
    
    # Create dataset
    testset = CocoImageDataset(test_image_paths, image_to_captions, transform=transformations)
    print(f"Created COCO test dataset with {len(test_image_paths)} images")
    
    return None, None, testset


def coco_collate_fn(batch):
    """
    Custom collate function for COCO dataset to properly handle caption tuples.
    
    Args:
        batch: List of (image, caption_tuple) pairs
        
    Returns:
        images: Batched image tensor
        captions: List of caption tuples (one tuple per image in batch)
    """
    images = []
    captions = []
    
    for image, caption_tuple in batch:
        images.append(image)
        captions.append(caption_tuple)
    
    # Stack images into a batch tensor
    images = torch.stack(images, dim=0)
    
    # Keep captions as a list of tuples
    return images, captions


def get_coco_dataloaders(batch_size, transformations, only_test=False, valid_split=False, shuffle=True):
    """
    Load COCO dataset with test split only.
    Each image is paired with a tuple of captions.
    
    Args:
        batch_size: Batch size for dataloader
        transformations: Image transformations
        only_test: Must be True for COCO
        valid_split: Ignored for COCO
        shuffle: Whether to shuffle the test set
        
    Returns:
        tuple: (dummy_train_dataloader, dummy_valid_dataloader, test_dataloader)
    """
    if not only_test:
        raise ValueError("COCO dataset only supports test set. Please set only_test=True")
    
    trainset, validset, testset = get_coco_datasets(transformations, only_test=True, valid_split=False)
    
    # Check the paths of the first 3 images
    if testset is not None:
        print("Paths of the first 3 images in the testset:")
        for i in range(min(3, len(testset))):
            print(testset.image_paths[i])
    
    dummy_train_dataloader, dummy_valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle, collate_fn=coco_collate_fn)
    print("COCO dataloaders created (train and valid are dummy, test contains actual data)")
    
    return dummy_train_dataloader, dummy_valid_dataloader, test_dataloader



def get_ham10000_datasets(transformations, only_test=False, valid_split=False):
    """
    Load HAM10000 dataset.
    """
    class HAM10000Dataset(torch.utils.data.Dataset):
        def __init__(self, split='train', transform=None):
            self.dataset = load_dataset("abaryan/ham10000_bbox", split=split)
            self.transform = transform
            self.label_to_idx = {
                "nv": 0,
                "mel": 1,
                "bkl": 2,
                "bcc": 3,
                "akiec": 4,
                "vasc": 5,
                "df": 6
            }

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            image = item['image']
            label_str = item['diagnosis']
            label = self.label_to_idx[label_str]

            if self.transform:
                image = self.transform(image)

            return image, label
    
    if only_test:
        testset = HAM10000Dataset(split='train', transform=transformations)
        return None, None, testset
    
    full_dataset = HAM10000Dataset(split='train', transform=transformations)
    
    # Split into train and test (80/20)
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    trainset, testset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=generator)
    
    if valid_split:
        train_size_new = int(0.8 * len(trainset))
        valid_size = len(trainset) - train_size_new
        trainset, validset = torch.utils.data.random_split(trainset, [train_size_new, valid_size], generator=generator)
        return trainset, validset, testset
    else:
        return trainset, None, testset

def get_ham10000_dataloaders(batch_size, transformations, only_test=False, valid_split=False, shuffle=True):
    """
    Load HAM10000 dataloaders.
    """
    trainset, validset, testset = get_ham10000_datasets(transformations, only_test, valid_split)
    
    if only_test:
        dummy_train_dataloader, dummy_valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
        print(f"HAM10000 dataloaders created (train and valid are dummy, test contains actual data)")
        return dummy_train_dataloader, dummy_valid_dataloader, test_dataloader
    
    if valid_split:
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
        valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=shuffle)
    else:
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
        _, valid_dataloader = create_dummy_dataloaders(batch_size, transformations)
        
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
    
    return train_dataloader, valid_dataloader, test_dataloader
