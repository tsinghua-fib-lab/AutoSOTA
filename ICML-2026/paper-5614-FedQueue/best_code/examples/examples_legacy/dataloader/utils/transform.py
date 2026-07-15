import torchvision.transforms as transforms


def test_transform(dataset):
    """Return the test transformation for different datast (MNIST/CIFAR10)"""
    if dataset == "MNIST":
        return transforms.ToTensor()
    elif dataset == "CIFAR10":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        return transforms.Compose([transforms.ToTensor(), normalize])
    else:
        raise NotImplementedError


def train_transform(dataset):
    """Return the train transformation for different datast (MNIST/CIFAR10)"""
    if dataset == "MNIST":
        return transforms.ToTensor()
    elif dataset == "CIFAR10":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                normalize,
            ]
        )
    else:
        raise NotImplementedError
