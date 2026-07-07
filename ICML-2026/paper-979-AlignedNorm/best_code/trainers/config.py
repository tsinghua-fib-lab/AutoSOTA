def get_dataset_specified_config(dataset, trainer, task):
    """Get dataset specific."""
    assert task in ["B2N", "FS", "CD"], "The TASK must be either B2N, CD, or FS."
    assert trainer in ["MMRL", "MMRLpp"], "The TRAINER must be either MMRL or MMRLpp."
    if trainer == "MMRL":
        cfg = {
            "StanfordCars": {
                "TRAINER.MMRL.REG_WEIGHT": 7.0,
            },
            "FGVCAircraft": {
                "TRAINER.MMRL.REG_WEIGHT": 6.0,
            },
            "SUN397": {
                "TRAINER.MMRL.REG_WEIGHT": 6.0,
            },
            "DescribableTextures": {
                "TRAINER.MMRL.REG_WEIGHT": 6.0,
            },
            "Food101": {
                "TRAINER.MMRL.REG_WEIGHT": 5.0,
            },
            "OxfordFlowers": {
                "TRAINER.MMRL.REG_WEIGHT": 4.0,
            },
            "UCF101": {
                "TRAINER.MMRL.REG_WEIGHT": 3.0,
            },
            "ImageNet": {
                "TRAINER.MMRL.REG_WEIGHT": 0.5,
            },
            "Caltech101": {
                "TRAINER.MMRL.REG_WEIGHT": 0.5,
            },
            "OxfordPets": {
                "TRAINER.MMRL.REG_WEIGHT": 0.2,
            },
            "EuroSAT": {
                "TRAINER.MMRL.REP_DIM": 2048,
                "TRAINER.MMRL.REG_WEIGHT": 0.01,
            },
        }.get(dataset, {})
    else:
        if task in ["B2N", "FS"]:
            cfg = {
                "ImageNet": {
                    "TRAINER.MMRLpp.BETA": 0.9,
                    "TRAINER.MMRLpp.REG_WEIGHT": 0.2,
                },
                "FGVCAircraft": {
                    "TRAINER.MMRLpp.BETA": 0.9,
                    "TRAINER.MMRLpp.REG_WEIGHT": 2.0,
                },
                "UCF101": {
                    "TRAINER.MMRLpp.BETA": 0.9,
                    "TRAINER.MMRLpp.REG_WEIGHT": 3.0,
                },
                "DescribableTextures": {
                    "TRAINER.MMRLpp.BETA": 0.9,
                    "TRAINER.MMRLpp.REG_WEIGHT": 7.0,
                },
                "OxfordPets": {
                    "TRAINER.MMRLpp.BETA": 0.7,
                    "TRAINER.MMRLpp.REG_WEIGHT": 0.01,
                },
                "StanfordCars": {
                    "TRAINER.MMRLpp.BETA": 0.7,
                    "TRAINER.MMRLpp.REG_WEIGHT": 6.0,
                },
                "Caltech101": {
                    "TRAINER.MMRLpp.BETA": 0.6,
                    "TRAINER.MMRLpp.REG_WEIGHT": 3.0,
                },
                "SUN397": {
                    "TRAINER.MMRLpp.BETA": 0.5,
                    "TRAINER.MMRLpp.REG_WEIGHT": 3.0,
                },
                "OxfordFlowers": {
                    "TRAINER.MMRLpp.BETA": 0.4,
                    "TRAINER.MMRLpp.REG_WEIGHT": 7.0,
                },
                "EuroSAT": {
                    "TRAINER.MMRLpp.BETA": 0.2,
                    "TRAINER.MMRLpp.REG_WEIGHT": 0.01,
                },
                "Food101": {
                    "TRAINER.MMRLpp.BETA": 0.1,
                    "TRAINER.MMRLpp.REG_WEIGHT": 1.0,
                },
            }.get(dataset, {})
        else:
            cfg = {
                "ImageNet": {
                    "TRAINER.MMRLpp.BETA": 0.9,
                    "TRAINER.MMRLpp.REG_WEIGHT": 0.1,
                },
                "ImageNetV2":{
                    "TRAINER.MMRLpp.BETA": 0.9,
                },
                "ImageNetR":{
                    "TRAINER.MMRLpp.BETA": 0.9,
                },
                "ImageNetA":{
                    "TRAINER.MMRLpp.BETA": 0.8,
                },
                "ImageNetSketch":{
                    "TRAINER.MMRLpp.BETA": 0.7,
                },
                "FGVCAircraft": {
                    "TRAINER.MMRLpp.BETA": 0.9,
                },
                "UCF101": {
                    "TRAINER.MMRLpp.BETA": 0.9,
                },
                "SUN397": {
                    "TRAINER.MMRLpp.BETA": 0.7,
                },
                "OxfordPets": {
                    "TRAINER.MMRLpp.BETA": 0.6,
                },
                "Caltech101": {
                    "TRAINER.MMRLpp.BETA": 0.6,
                },
                "DescribableTextures": {
                    "TRAINER.MMRLpp.BETA": 0.5,
                },
                "OxfordFlowers": {
                    "TRAINER.MMRLpp.BETA": 0.4,
                },
                "StanfordCars": {
                    "TRAINER.MMRLpp.BETA": 0.3,
                },
                "EuroSAT": {
                    "TRAINER.MMRLpp.BETA": 0.3,
                },
                "Food101": {
                    "TRAINER.MMRLpp.BETA": 0.3,
                },
            }.get(dataset, {})

    return [item for pair in cfg.items() for item in pair]