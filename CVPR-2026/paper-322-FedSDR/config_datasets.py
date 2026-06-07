def attribute(dataset):
    if dataset == ["PubMed"]:
        return 10, 3, 1000, 0.01
    elif dataset == ["CS"]:
        return 50, 15, 1000, 0.005
    elif dataset == ["Physics"]:
        return 100, 5, 1000, 0.01
    elif dataset == ["Actor"]:
        return 20, 5, 1000, 0.002
    elif dataset == ["Roman-empire"]:
        return 50, 18, 5000, 0.02
