from flamby.datasets.fed_heart_disease import FedHeartDisease


def get_flamby():
    test_dataset = FedHeartDisease(train=False, center=0, pooled=False)
    train_dataset = FedHeartDisease(train=True, center=0, pooled=False)
    return train_dataset, test_dataset
