import torch
from torch.utils.data import DataLoader
from medmnist import PathMNIST, TissueMNIST, OrganAMNIST, BloodMNIST
import torchvision.transforms as transforms
from dataloaders.index_dataset import IndexDataset

# Custom Dataset Wrapper to Squeeze Labels
class MedMNISTWrapper(torch.utils.data.Dataset):
    """
    Dataset wrapper to squeeze MedMNIST labels from shape [1] → scalar.

    This is used because MedMNIST labels are provided as shape (1,)
    tensors, which can cause downstream compatibility issues.

    Args:
        dataset (torch.utils.data.Dataset): The base dataset to wrap.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        # Squeeze the label to remove the extra dimension
        label = label.squeeze()  # Converts [1] to scalar
        return img, label

# MedMNIST Dataset
def get_loaders(args, index_dataset: bool, device):
    """
    Prepare MedMNIST dataset loaders with optional normalization and index annotation.

    Supports the following subsets:
        - PathMNIST      (3-channel histopathology patches)
        - TissueMNIST    (1-channel kidney tissue microscopy)
        - OrganAMNIST    (1-channel CT slices of abdominal organs)
        - BloodMNIST     (3-channel blood-cell microscopy)

    Features:
        - Dataset-specific mean/std normalization (optional).
        - Auto-channel replication to RGB if dataset is grayscale.
        - Augmentation: horizontal flip + small rotations for non-index datasets.
        - Wrapping for label squeezing and optional index annotation.
        - Computation of per-channel pixel bounds for adversarial clamping.

    Args:
        args (argparse.Namespace): Contains dataset, root_path, batch_size, num_workers,
            normalize_dataset, and other configuration options.
        index_dataset (bool): If True, wraps dataset with IndexDataset to return
            (sample, label, index) tuples for tracking.
        device (str): Target device for bound/mu/std tensors (e.g., "cuda").

    Returns:
        tuple:
            trainloader (DataLoader): Augmented MedMNIST training loader.
            testloader (DataLoader): Normalized MedMNIST test loader.
            upper_limit (torch.Tensor): Per-channel max bound after normalization.
            lower_limit (torch.Tensor): Per-channel min bound after normalization.
            mu (torch.Tensor): Per-channel dataset mean (C×1×1).
            std (torch.Tensor): Per-channel dataset std (C×1×1).
            classes (list[str]): Class names for the dataset.
            num_classes (int): Number of classes in dataset.
            len_trainset (int): Number of training samples.
            len_testset (int): Number of test samples.

    References:
        Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., & Ni, B. (2023).
        MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification.
        *Scientific Data, 10*(1), 41.
        URL: https://medmnist.com/
    """
    # Map dataset name to class and metadata
    dataset_map = {
        'PathMNIST': {
            'class': PathMNIST,
            'channels': 3,
            'num_classes': 9,
            'mean': [0.740545  , 0.53298219, 0.70582885],
            'std': [0.12368222, 0.17676253, 0.12443067],
            'classes': [
                'adipose', 'background', 'debris', 'lymphocytes', 'mucus',
                'smooth muscle', 'normal colon', 'cancer-associated stroma', 'colorectal adenocarcinoma'
            ]
        },
        'TissueMNIST': {
            'class': TissueMNIST,
            'channels': 1,
            'num_classes': 8,
            'mean': [0.10204512936041181],
            'std': [0.10002610110235889],
            'classes': [
                'cortex', 'glomeruli', 'medulla', 'blood vessels',
                'pelvis', 'calyces', 'fat', 'background'
            ]
        },
        'OrganAMNIST': {
            'class': OrganAMNIST,
            'channels': 1,        # single‐channel CT slices
            'num_classes': 11,
            'mean': [0.46802823658325343],
            'std': [0.2974209246855977],
            'classes': [
                'spleen',
                'right kidney',
                'left kidney',
                'gallbladder',
                'esophagus',
                'liver',
                'stomach',
                'aorta',
                'pancreas',
                'right adrenal gland',
                'left adrenal gland'
            ]
        },
        'BloodMNIST': {                
            'class': BloodMNIST,
            'channels': 3,              # blood-cell microscope images are RGB
            'num_classes': 8,
            'mean': [0.79434784, 0.65965901, 0.69619251],
            'std': [0.79434784, 0.65965901, 0.69619251],
            'classes': [               # exact names from INFO in medmnist/info.py
                'erythrocyte',
                'eosinophil granulocyte',
                'large unstained cell',
                'lymphocyte',
                'monocyte',
                'neutrophil granulocyte',
                'basophil granulocyte',
                'platelet'
            ]
        }
    }

    # Validate input
    if args.dataset not in dataset_map:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    ds_info = dataset_map[args.dataset]
    to_rgb = (ds_info['channels'] == 1)

    # Define transforms
    def build_transform(train=True):
        ops = [transforms.ToTensor()]
        if to_rgb:
            # replicate single channel → 3 channels
            ops.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))

        if train:
            ops += [
                transforms.Pad(2),
                transforms.RandomRotation(10),
                transforms.Normalize(mean=ds_info['mean'], std=ds_info['std'])
            ]
        else:
            ops += [
                transforms.Pad(2),
                transforms.Normalize(mean=ds_info['mean'], std=ds_info['std'])
            ]
        return transforms.Compose(ops)


    if args.normalize_dataset:
        medmnist_mean = dataset_map[args.dataset]['mean'] # equals np.mean(train_set.train_data, axis=(0,1,2))/255
        medmnist_std = dataset_map[args.dataset]['std'] # equals np.std(train_set.train_data, axis=(0,1,2))/255
    else:
        if args.dataset in ["TissueMNIST", "OrganAMNIST"]:
            medmnist_mean = [0.]
            medmnist_std = [1.]
        else:
            medmnist_mean = [0., 0., 0.]
            medmnist_std = [1., 1., 1.]

    mu = torch.tensor(medmnist_mean).view(-1,1,1).to(device)
    std = torch.tensor(medmnist_std).view(-1,1,1).to(device)
    
    if index_dataset:
        train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(medmnist_mean, medmnist_std)
    ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),  # Flip for chest X-rays
            # transforms.Pad(2),
            transforms.RandomRotation(degrees=10),   # Small rotations
            transforms.Normalize(medmnist_mean, medmnist_std)
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Pad(2),
        transforms.Normalize(medmnist_mean, medmnist_std)
    ])

    # train_transform = build_transform(train=True)
    # test_transform  = build_transform(train=False)

    # Load the actual datasets
    trainset = ds_info['class'](
        root=f'{args.root_path}/Datasets/{args.dataset}',
        split='train', transform=train_transform,
        download=True, size=28
    )

    testset  = ds_info['class'](
        root=f'{args.root_path}/Datasets/{args.dataset}',
        split='test',  transform=test_transform,
        download=True, size=28
    )

    # Wrap datasets to fix label shape
    trainset = MedMNISTWrapper(trainset)
    trainset = IndexDataset(trainset) if index_dataset else trainset # Index Dataset

    testset = MedMNISTWrapper(testset)
    
    # Create DataLoaders
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.num_workers)

    # Legal limits of pixles after normalization
    upper_limit = ((1 - mu)/ std).to(device)
    lower_limit = ((0 - mu)/ std).to(device)
    
    return trainloader, testloader, upper_limit, lower_limit, mu, std, ds_info['classes'], ds_info['num_classes'], len(trainset), len(testset)
