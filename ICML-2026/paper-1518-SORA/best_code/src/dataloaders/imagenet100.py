import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from datasets import load_dataset
from dataloaders.index_dataset import IndexDataset
from PIL import Image
import os


# Wrap HF dataset in a PyTorch Dataset
class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"]
        label = item["label"]

        # FORCE RGB (fix grayscale images)
        if img.mode != "RGB":
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

        

def get_loaders(args, index_dataset: bool, device):
    """
    Prepare ImageNet-100 dataset loaders with optional index annotation and normalization.

    This function handles:
        - Dataset mean/std normalization (optional).
        - Choice between standard augmentation (random crop + horizontal flip)
          or deterministic padding for index-aware datasets.
        - Creation of train/test DataLoaders with specified batch size and workers.
        - Computation of normalized upper/lower pixel bounds for adversarial clamping.

    Args:
        args (argparse.Namespace): Contains dataset, root_path, batch_size, num_workers,
            normalize_dataset, and other configuration options.
        index_dataset (bool): If True, wraps the training set with IndexDataset to return
            (sample, label, index) tuples for dataset index tracking.
        device (str): Device on which normalization tensors and pixel bounds will be stored 
            (e.g., "cuda").

    Returns:
        tuple:
            trainloader (DataLoader): Augmented ImageNet-100 training loader.
            testloader (DataLoader): Normalized ImageNet-100 test loader.
            upper_limit (torch.Tensor): Per-channel perturbed pixel max bound (normalized).
            lower_limit (torch.Tensor): Per-channel perturbed pixel min bound (normalized).
            mu (torch.Tensor): Dataset channel means (C×1×1).
            std (torch.Tensor): Dataset channel standard deviations (C×1×1).
            classes (tuple[str]): Ordered tuple of ImageNet-100 class names.
            num_classes (int): Number of classes (100 for ImageNet-100).
            len_trainset (int): Number of training samples (117,000 for ImageNet-100).
            len_testset (int): Number of test samples (5,000 for ImageNet-100).

    References:
    
    """
    if args.normalize_dataset:
        imagenet_mean = [0.485, 0.456, 0.406] # equals np.mean(train_set.train_data, axis=(0,1,2))/255
        imagenet_std =  [0.229, 0.224, 0.225] # equals np.std(train_set.train_data, axis=(0,1,2))/255
    else:
        imagenet_mean = [0., 0., 0.]
        imagenet_std = [1., 1., 1.]
    
    mu = torch.tensor(imagenet_mean).view(3,1,1).to(device)
    std = torch.tensor(imagenet_std).view(3,1,1).to(device)
    
    if index_dataset:
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ])
    else:  
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # Load via Hugging Face datasets
    ds = load_dataset("ilee0022/ImageNet100", verification_mode="no_checks")
    # ds = load_dataset("ilee0022/ImageNet100", cache_dir="./imagenet100_cache")
    train_ds = ds["train"]
    test_ds = ds["test"]
    # val_ds = ds["validation"]

    # Download the dataset
    trainset = HFDatasetWrapper(train_ds, transform=train_transform)
    testset = HFDatasetWrapper(test_ds, transform=test_transform)
    # valset = HFDatasetWrapper(val_ds, transform=val_transform)

    trainset = IndexDataset(trainset) if index_dataset else trainset # Index Dataset

    # Create the loaders
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Legal limits of pixles after normalization
    upper_limit = ((1 - mu)/ std).to(device)
    lower_limit = ((0 - mu)/ std).to(device)

    # Name of Classes
    classes = (
        'bittern', 'conch', 'indigo bunting, indigo finch, indigo bird, Passerina cyanea', 'hornbill', 'bee eater', 'ptarmigan', 'stingray', 
        'bald eagle, American eagle, Haliaeetus leucocephalus', 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish', 
        'flatworm, platyhelminth', 'limpkin, Aramus pictus', 'sea slug, nudibranch', 'terrapin', 'boa constrictor, Constrictor constrictor', 
        'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus', 'sidewinder, horned rattlesnake, Crotalus cerastes', 
        'axolotl, mud puppy, Ambystoma mexicanum', 'tarantula', 'black grouse', 'hammerhead, hammerhead shark', 
        'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis', 'drake', 'peacock', 'hognose snake, puff adder, sand viper', 
        'Dungeness crab, Cancer magister', 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita', 'green lizard, Lacerta viridis', 'bustard', 
        'wombat', 'goldfish, Carassius auratus', 'hummingbird', 'flamingo', 'magpie', 'rock crab, Cancer irroratus', 'crane', 
        'tiger shark, Galeocerdo cuvieri', 'common iguana, iguana, Iguana iguana', 'spoonbill', 'thunder snake, worm snake, Carphophis amoenus', 
        'toucan', 'goose', 'bulbul', 'harvestman, daddy longlegs, Phalangium opilio', 'kite', 'wolf spider, hunting spider', 'albatross, mollymawk', 
        'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias', 'black and gold garden spider, Argiope aurantia', 
        'green snake, grass snake', 'crayfish, crawfish, crawdad, crawdaddy', 'garden spider, Aranea diademata', 'black swan, Cygnus atratus', 
        'common newt, Triturus vulgaris', 'hermit crab', 'electric ray, crampfish, numbfish, torpedo', 'great grey owl, great gray owl, Strix nebulosa', 
        'vine snake', 'American coot, marsh hen, mud hen, water hen, Fulica americana', 'lorikeet', 'banded gecko', 'hen', 'macaw', 'snail', 
        'water ouzel, dipper', 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui', 
        'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea', 'redshank, Tringa totanus', 'mud turtle', 
        'chiton, coat-of-mail shell, sea cradle, polyplacophore', 'American alligator, Alligator mississipiensis', 'goldfinch, Carduelis carduelis', 
        'red-backed sandpiper, dunlin, Erolia alpina', 'scorpion', 'tench, Tinca tinca', 'barn spider, Araneus cavaticus', 'nematode, nematode worm, roundworm', 
        'oystercatcher, oyster catcher', 'king snake, kingsnake', 'whiptail, whiptail lizard', 'agama', 'chambered nautilus, pearly nautilus, nautilus', 
        'sea snake', 'prairie chicken, prairie grouse, prairie fowl', 'black widow, Latrodectus mactans', 'pelican', 'night snake, Hypsiglena torquata', 
        'garter snake, grass snake', 'sea anemone, anemone', 'wallaby, brush kangaroo', 'tick', 'coucal', 'diamondback, diamondback rattlesnake, Crotalus adamanteus', 
        'sea lion', 'loggerhead, loggerhead turtle, Caretta caretta', 'rooster', 'green mamba', 'spotted salamander, Ambystoma maculatum', 'white stork, Ciconia ciconia', 
        'chickadee', 'jellyfish')

    return trainloader, testloader, upper_limit, lower_limit, mu, std, classes, len(classes), len(trainset), len(testset)
