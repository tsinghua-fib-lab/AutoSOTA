def get_dataset_specified_config(dataset, trainer, task):
    """Get dataset specific."""
    assert task in ["B2N", "FS", "CD"], "The TASK must be either B2N, CD, or FS."
    assert trainer == "ALIGNEDNORM", "The TRAINER must be ALIGNEDNORM."
    if trainer == "ALIGNEDNORM":
        if task == "B2N":
            cfg = {
                "ImageNet": { # best(0.9, 0.2, 0.005, 0.005)
                    "TRAINER.ALIGNEDNORM.BETA": 0.9, # 0.9
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 0.2, # 0.2
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.005,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.005,
                    "OPTIM.MAX_EPOCH": 5
                },
                "FGVCAircraft": { # best(0.9, 2.0, 0.1, 0.01)
                    "TRAINER.ALIGNEDNORM.BETA": 0.9,
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 2.0,
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.1,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.01,
                },
                "UCF101": { # best(0.9, 3.0, 0.15, 0.15)
                    "TRAINER.ALIGNEDNORM.BETA": 0.9,
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 3.0,
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.15,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.15,
                },
                "DescribableTextures": { # best(0.9, 7.0, 0.2, 0.15)
                    "TRAINER.ALIGNEDNORM.BETA": 0.9, # 0.9
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 7.0, # 7.0
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.2,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.15,
                },
                "OxfordPets": { # best(0.7, 0.01, 0.1, 0.01)
                    "TRAINER.ALIGNEDNORM.BETA": 0.7,
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 0.01,
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.1,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.01,
                },
                "StanfordCars": { # best(0.6, 6.0, 0.15, 0.15, 15)
                    "TRAINER.ALIGNEDNORM.BETA": 0.6,
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 6.0,
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.15,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.15,
                    "OPTIM.MAX_EPOCH": 13
                },
                "Caltech101": { # best(0.6, 3.0, 0.05, 0.05)
                    "TRAINER.ALIGNEDNORM.BETA": 0.6, # 0.6
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 3.0, # 3.0
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.05,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.05,
                },
                "SUN397": { # best(0.5, 3.0, 0.1, 0.01)
                    "TRAINER.ALIGNEDNORM.BETA": 0.5,
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 3.0,
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.1,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.01,
                },
                "OxfordFlowers": { # best(0.4, 7.0, 0.15, 0.1)
                    "TRAINER.ALIGNEDNORM.BETA": 0.4,
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 7.0,
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.15,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.1,
                },
                "EuroSAT": { # best(0.2, 0.01, 0.1, 0.05)
                    "TRAINER.ALIGNEDNORM.BETA": 0.2,
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 0.01,
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.1,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.05,
                },
                "Food101": { # best(0.1, 2.0, 0.15, 0.001)
                    "TRAINER.ALIGNEDNORM.BETA": 0.1,
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 2.0,
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.15,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.001,
                },
            }.get(dataset, {})
        elif task == "FS":
            cfg = {
                "ImageNet": {
                    "TRAINER.ALIGNEDNORM.BETA": 0.9, # 0.9
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 0.2, # 0.2
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.005,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.005,
                },
                "FGVCAircraft": {
                    "TRAINER.ALIGNEDNORM.BETA": 0.9,
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 2.0,
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.1,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.01,
                },
                "UCF101": {
                    "TRAINER.ALIGNEDNORM.BETA": 0.9,
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 3.0,
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.15,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.15,
                },
                "DescribableTextures": {
                    "TRAINER.ALIGNEDNORM.BETA": 0.9, # 0.9
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 7.0, # 7.0
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.2,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.15,
                },
                "OxfordPets": {
                    "TRAINER.ALIGNEDNORM.BETA": 0.7,
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 0.01,
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.1,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.01,
                },
                "StanfordCars": {
                    "TRAINER.ALIGNEDNORM.BETA": 0.6,
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 6.0,
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.15,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.15,
                },
                "Caltech101": {
                    "TRAINER.ALIGNEDNORM.BETA": 0.6, # 0.6
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 3.0, # 3.0
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.05,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.05,
                },
                "SUN397": {
                    "TRAINER.ALIGNEDNORM.BETA": 0.5,
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 3.0,
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.1,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.01,
                },
                "OxfordFlowers": {
                    "TRAINER.ALIGNEDNORM.BETA": 0.4,
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 7.0,
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.1,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.01,
                },
                "EuroSAT": {
                    "TRAINER.ALIGNEDNORM.BETA": 0.2,
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 0.01,
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.1,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.05,
                },
                "Food101": {
                    "TRAINER.ALIGNEDNORM.BETA": 0.1,
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 2.0,
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.15,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.001,
                },
            }.get(dataset, {})
        else:
            cfg = {
                "ImageNet": {
                    "TRAINER.ALIGNEDNORM.BETA": 0.9,
                    "TRAINER.ALIGNEDNORM.REG_WEIGHT": 0.1,
                    "TRAINER.ALIGNEDNORM.FINAL_NORM": 0.005,
                    "TRAINER.ALIGNEDNORM.SEQ_NORM": 0.005,
                },
                "ImageNetV2":{
                    "TRAINER.ALIGNEDNORM.BETA": 0.9,
                },
                "ImageNetR":{
                    "TRAINER.ALIGNEDNORM.BETA": 0.9,
                },
                "ImageNetA":{
                    "TRAINER.ALIGNEDNORM.BETA": 0.8,
                },
                "ImageNetSketch":{
                    "TRAINER.ALIGNEDNORM.BETA": 0.7,
                },
                "FGVCAircraft": {
                    "TRAINER.ALIGNEDNORM.BETA": 0.9,
                },
                "UCF101": {
                    "TRAINER.ALIGNEDNORM.BETA": 0.9,
                },
                "SUN397": {
                    "TRAINER.ALIGNEDNORM.BETA": 0.7,
                },
                "OxfordPets": {
                    "TRAINER.ALIGNEDNORM.BETA": 0.6,
                },
                "Caltech101": {
                    "TRAINER.ALIGNEDNORM.BETA": 0.6,
                },
                "DescribableTextures": {
                    "TRAINER.ALIGNEDNORM.BETA": 0.5,
                },
                "OxfordFlowers": {
                    "TRAINER.ALIGNEDNORM.BETA": 0.4,
                },
                "StanfordCars": {
                    "TRAINER.ALIGNEDNORM.BETA": 0.3,
                },
                "EuroSAT": {
                    "TRAINER.ALIGNEDNORM.BETA": 0.3,
                },
                "Food101": {
                    "TRAINER.ALIGNEDNORM.BETA": 0.3,
                },
            }.get(dataset, {})

    return [item for pair in cfg.items() for item in pair]
