from appfl.misc.data_readiness.data_pollute import add_noise_to_subset


def get_flamby_ixi_polluted(
    dataset: str,
    num_clients: int,
    client_id: int,
    **kwargs,
):
    if dataset == "IXI":
        from flamby.datasets.fed_ixi import FedIXITiny

        assert num_clients <= 3, "IXI dataset can support at most three clients"
        test_dataset = FedIXITiny(train=False, pooled=True)
        train_dataset = FedIXITiny(train=True, center=client_id, pooled=False)
        train_dataset = add_noise_to_subset(train_dataset, scale=2, fraction=0.7)

        return train_dataset, test_dataset
    else:
        raise NotImplementedError
