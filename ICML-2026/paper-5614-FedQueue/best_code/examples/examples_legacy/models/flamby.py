try:
    import flamby.datasets.fed_ixi as IXI
    import flamby.datasets.fed_kits19 as Kits19
    import flamby.datasets.fed_tcga_brca as TcgaBrca
    import flamby.datasets.fed_isic2019 as ISIC2019
    import flamby.datasets.fed_heart_disease as HeartDisease
except:  # noqa E722
    print("FLamby is not installed.")


def flamby_train(dataset: str):
    if dataset == "TcgaBrca":
        return (
            TcgaBrca.Baseline(),
            TcgaBrca.BaselineLoss(),
            "Adam",
            TcgaBrca.LR,
            TcgaBrca.BATCH_SIZE,
            TcgaBrca.metric,
        )
    elif dataset == "HeartDisease":
        return (
            HeartDisease.Baseline(),
            HeartDisease.BaselineLoss(),
            "Adam",
            HeartDisease.LR,
            HeartDisease.BATCH_SIZE,
            HeartDisease.metric,
        )
    elif dataset == "IXI":
        return (
            IXI.Baseline(),
            IXI.BaselineLoss(),
            "AdamW",
            IXI.LR,
            IXI.BATCH_SIZE,
            IXI.metric,
        )
    elif dataset == "ISIC2019":
        return (
            ISIC2019.Baseline(),
            ISIC2019.BaselineLoss(),
            "Adam",
            ISIC2019.LR,
            ISIC2019.BATCH_SIZE,
            ISIC2019.metric,
        )
    elif dataset == "Kits19":
        return (
            Kits19.Baseline(),
            Kits19.BaselineLoss(),
            "Adam",
            Kits19.LR,
            Kits19.BATCH_SIZE,
            Kits19.metric,
        )
    else:
        raise NotImplementedError
