try:
    from flamby.datasets.fed_tcga_brca import FedTcgaBrca
    from flamby.datasets.fed_heart_disease import FedHeartDisease
    from flamby.datasets.fed_ixi import FedIXITiny
    from flamby.datasets.fed_isic2019 import FedIsic2019
    from flamby.datasets.fed_kits19 import FedKits19
except:  # noqa E722
    pass


def get_flamby(dataset: str, num_clients: int):
    if dataset == "TcgaBrca":
        assert num_clients <= 6, "TcgaBrca dataset can support at most six clients"
        test_dataset = FedTcgaBrca(train=False, pooled=True)
        train_datasets = [
            FedTcgaBrca(train=True, center=i, pooled=False) for i in range(num_clients)
        ]
        return train_datasets, test_dataset
    elif dataset == "HeartDisease":
        assert num_clients <= 4, (
            "Heart disease dataset can support at most four clients"
        )
        test_dataset = FedHeartDisease(train=False, pooled=True)
        train_datasets = [
            FedHeartDisease(train=True, center=i, pooled=False)
            for i in range(num_clients)
        ]
        return train_datasets, test_dataset
    elif dataset == "IXI":
        assert num_clients <= 3, "IXI dataset can support at most three clients"
        test_dataset = FedIXITiny(train=False, pooled=True)
        train_datasets = [
            FedIXITiny(train=True, center=i, pooled=False) for i in range(num_clients)
        ]
        return train_datasets, test_dataset
    elif dataset == "ISIC2019":
        assert num_clients <= 6, "ISIC 2019 dataset can support at most six clients"
        test_dataset = FedIsic2019(train=False, pooled=True)
        train_datasets = [
            FedIsic2019(train=True, center=i, pooled=False) for i in range(num_clients)
        ]
        return train_datasets, test_dataset
    elif dataset == "Kits19":
        assert num_clients <= 6, "Kits19 dataset can support at most six clients"
        test_dataset = FedKits19(train=False, pooled=True)
        train_datasets = [
            FedKits19(train=True, center=i, pooled=False) for i in range(num_clients)
        ]
        return train_datasets, test_dataset
    else:
        raise NotImplementedError
