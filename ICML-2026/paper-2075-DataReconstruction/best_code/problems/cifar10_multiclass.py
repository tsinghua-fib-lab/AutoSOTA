import torch
import torchvision.datasets
import torchvision.transforms


def balanced_dataset(num_classes, data_per_class, dataset):
    indices = torch.tensor([])
    for c in range(num_classes):
        equals = (torch.tensor(dataset.targets) == c)
        temp_indices = torch.nonzero(equals)[:, 0]
        indices = torch.cat((indices, temp_indices[0: data_per_class]), 0)
    dataset = torch.utils.data.Subset(dataset, indices[torch.randperm(indices.shape[0])].long())
    data_loader = torch.utils.data.DataLoader(dataset, data_per_class * num_classes, shuffle=False)
    return data_loader

def balanced_trainset_with_validation(num_classes, data_per_class_train, data_per_class_val, dataset):
    train_indices, val_indices = torch.tensor([]), torch.tensor([])
    for c in range(num_classes):
        equals = (torch.tensor(dataset.targets) == c)
        temp_indices = torch.nonzero(equals)[:, 0]
        train_indices = torch.cat((train_indices, temp_indices[0: data_per_class_train]), 0)
        val_indices = torch.cat((val_indices, temp_indices[data_per_class_train: data_per_class_train + data_per_class_val]), 0)
    train_set = torch.utils.data.Subset(dataset, train_indices[torch.randperm(train_indices.shape[0])].long())
    val_set = torch.utils.data.Subset(dataset, val_indices[torch.randperm(val_indices.shape[0])].long())
    train_loader = torch.utils.data.DataLoader(train_set, data_per_class_train * num_classes, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_set, data_per_class_val * num_classes, shuffle=False)
    return train_loader, val_loader


def fetch_cifar10(root, train=False, transform=None, target_transform=None):
    transform = transform if transform is not None else torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR10(root, train=train, transform=transform, target_transform=target_transform,
                                           download=True)
    return dataset


def load_cifar10(args, root, transform=None, target_transform=None, **kwargs):
    train_set = fetch_cifar10(root, train=True, transform=transform, target_transform=target_transform)
    test_set = fetch_cifar10(root, train=False, transform=transform, target_transform=target_transform)
    val_loader = None
    if args.data_per_class_val > 0:
        train_loader, val_loader = balanced_trainset_with_validation(args.num_classes, args.data_per_class_train, args.data_per_class_val, train_set)
    else:
        train_loader = balanced_dataset(args.num_classes, args.data_per_class_train, train_set)
    test_loader = balanced_dataset(args.num_classes, args.data_per_class_test, test_set)
    return train_loader, test_loader, val_loader


def move_to_type_device(args, data_loader):
    x0, y0 = next(iter(data_loader))

    x0, y0 = x0.to(args.device), y0.to(args.device)

    y0 = y0.long()
    x0 = x0.to(torch.get_default_dtype())

    return x0, y0


def load_cifar10_data(args):
    train_loader, test_loader, val_loader = load_cifar10(args=args, root=args.datasets_dir)

    x0_train, y0_train = move_to_type_device(args, train_loader)
    print('X_train:', x0_train.shape)
    print('y_train:', y0_train.shape)

    x0_test, y0_test = move_to_type_device(args, test_loader)
    print('X_test:', x0_test.shape)
    print('y_test:', y0_test.shape)

    if val_loader is not None:
        x0_val, y0_val = move_to_type_device(args, val_loader)
        print('X_train:', x0_val.shape)
        print('y_train:', y0_val.shape)
        val_loader = [(x0_val, y0_val)]

    return [(x0_train, y0_train)], [(x0_test, y0_test)], val_loader


def get_dataloader(args):
    args.input_dim = 32 * 32 * 3
    args.num_classes = 10
    args.output_dim = 10
    args.dataset = 'cifar10'
    args.input_channels = 3
    args.input_height = 32
    args.input_width = 32

    if args.run_mode == 'reconstruct':
        args.extraction_data_amount = args.extraction_data_amount_per_class * args.num_classes

    # for legacy:
    args.data_amount = args.data_per_class_train * args.num_classes
    args.data_use_test = True
    args.data_test_amount = 1000

    data_loader = load_cifar10_data(args)
    return data_loader