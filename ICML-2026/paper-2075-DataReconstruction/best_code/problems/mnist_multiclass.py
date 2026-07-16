import torch
import torchvision.datasets
import torchvision.transforms


def load_bound_dataset(dataset, batch_size, shuffle=False, start=None, end=None, **kwargs):
    def _bound_dataset(dataset, start, end):
        if start is None:
            start = 0
        if end is None:
            end = len(dataset)
        return torch.utils.data.Subset(dataset, range(start, end))

    dataset = _bound_dataset(dataset, start, end)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle, **kwargs)


def fetch_mnist(root, train=False, transform=None, target_transform=None):
    transform = transform if transform is not None else torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root, train=train, transform=transform, target_transform=target_transform, download=True)
    return dataset


def load_mnist(root, batch_size, train=False, transform=None, target_transform=None, **kwargs):
    dataset = fetch_mnist(root, train, transform, target_transform)
    return load_bound_dataset(dataset, batch_size, **kwargs)


def move_to_type_device(x, y, device):
    x = x.to(torch.get_default_dtype()).to(device)
    y = y.long().to(device)  
    return x, y


def get_balanced_data(args, data_loader, data_per_class):
    print('BALANCING DATASET...')

    num_classes = args.num_classes
    labels_counter = {c: 0 for c in range(num_classes)}

    x0, y0 = [], []
    got_enough = False

    for bx, by in data_loader:
        for i in range(len(bx)):
            label = int(by[i])

            if labels_counter[label] < data_per_class:
                labels_counter[label] += 1
                x0.append(bx[i])
                y0.append(by[i])

            if all(labels_counter[c] >= data_per_class for c in range(num_classes)):
                got_enough = True
                break

        if got_enough:
            break

    x0 = torch.stack(x0)
    y0 = torch.tensor(y0)

    return x0, y0


def load_mnist_data(args):
    # Train set
    train_loader = load_mnist(root=args.datasets_dir, batch_size=100, train=True, shuffle=False, start=0, end=50000)

    x0, y0 = get_balanced_data(args, train_loader, args.data_per_class_train)

    # Test set
    print('LOADING TEST SET')

    test_loader = load_mnist(root=args.datasets_dir, batch_size=100, train=False, shuffle=False, start=0, end=10000)

    data_per_class_test = args.data_test_amount // args.num_classes

    x0_test, y0_test = get_balanced_data(args,test_loader,data_per_class_test)

    x0, y0 = move_to_type_device(x0, y0, args.device)
    x0_test, y0_test = move_to_type_device(x0_test, y0_test, args.device)

    return [(x0, y0)], [(x0_test, y0_test)], None


def get_dataloader(args):
    args.input_dim = 28 * 28
    args.num_classes = 10
    args.output_dim = 10
    args.dataset = 'mnist'
    args.input_channels = 1
    args.input_height = 28
    args.input_width = 28

    if args.run_mode == 'reconstruct':
        args.extraction_data_amount = args.extraction_data_amount_per_class * args.num_classes

    # legacy 
    args.data_amount = args.data_per_class_train * args.num_classes
    args.data_use_test = True
    args.data_test_amount = 1000  

    data_loader = load_mnist_data(args)
    return data_loader
