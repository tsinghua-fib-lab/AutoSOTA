def get_flamby(
    dataset: str,
    num_clients: int,
    client_id: int,
    **kwargs,
):
    if dataset == "TcgaBrca":
        from flamby.datasets.fed_tcga_brca import FedTcgaBrca

        assert num_clients <= 6, "TcgaBrca dataset can support at most six clients"
        test_dataset = FedTcgaBrca(train=False, pooled=True)
        train_dataset = FedTcgaBrca(train=True, center=client_id, pooled=False)
        return train_dataset, test_dataset
    elif dataset == "HeartDisease":
        from flamby.datasets.fed_heart_disease import FedHeartDisease

        assert num_clients <= 4, (
            "Heart disease dataset can support at most four clients"
        )
        test_dataset = FedHeartDisease(train=False, pooled=True)
        train_dataset = FedHeartDisease(train=True, center=client_id, pooled=False)
        return train_dataset, test_dataset
    elif dataset == "IXI":
        from flamby.datasets.fed_ixi import FedIXITiny

        assert num_clients <= 3, "IXI dataset can support at most three clients"
        test_dataset = FedIXITiny(train=False, pooled=True)
        train_dataset = FedIXITiny(train=True, center=client_id, pooled=False)
        return train_dataset, test_dataset
    elif dataset == "ISIC2019":
        from flamby.datasets.fed_isic2019 import FedIsic2019

        assert num_clients <= 6, "ISIC 2019 dataset can support at most six clients"
        test_dataset = FedIsic2019(train=False, pooled=True)
        train_dataset = FedIsic2019(train=True, center=client_id, pooled=False)
        return train_dataset, test_dataset
    elif dataset == "Kits19":
        from flamby.datasets.fed_kits19 import FedKits19

        assert num_clients <= 6, "Kits19 dataset can support at most six clients"
        test_dataset = FedKits19(train=False, pooled=True)
        train_dataset = FedKits19(train=True, center=client_id, pooled=False)
        return train_dataset, test_dataset
    else:
        raise NotImplementedError
