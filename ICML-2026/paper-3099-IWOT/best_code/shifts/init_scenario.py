import shifts.mnist_to_usps
import shifts.usps_to_mnist
import shifts.mnist_to_mnist_m
import shifts.svhn_to_mnist
import shifts.cifar_corrupt
import shifts.image_clef
import shifts.office_31
import shifts.office_home
import shifts.portraits
import shifts.vis_da17


# TODO: The advantage of putting init here should be that I shouldn't have to type shifts every time
# but the fact that I do suggests there is a python path problem!
def init(config, fabric):
    dataloader_options = {
        "batch_size": config["batch_size"],
        "shuffle": config["shuffle"],
        "drop_last": False,
    }
    test_dataloader_options = {
        "batch_size": config["test_batch_size"],
        "shuffle": config["test_shuffle"],
        "drop_last": False,
    }
    if config["scenario"] == "MNIST_TO_USPS":
        scenario = shifts.mnist_to_usps.MNIST_to_USPS(
            dataloader_options, test_dataloader_options
        )
    elif config["scenario"] == "USPS_TO_MNIST":
        scenario = shifts.usps_to_mnist.USPS_to_MNIST(
            dataloader_options, test_dataloader_options
        )
    elif config["scenario"] == "MNIST_TO_MNIST_M":
        scenario = shifts.mnist_to_mnist_m.MNIST_to_MNIST_M(
            dataloader_options, test_dataloader_options, preprocess=config["preprocess"]
        )
    elif config["scenario"] == "SVHN_TO_MNIST":
        scenario = shifts.svhn_to_mnist.SVHN_to_MNIST(
            dataloader_options, test_dataloader_options
        )
    elif config["scenario"] == "CIFAR-10-C":
        scenario = shifts.cifar_corrupt.CIFAR_CORRUPT(
            dataloader_options,
            test_dataloader_options,
            corruptions=config["cifar-10-corruptions"],
        )
    elif config["scenario"] == "PORTRAITS":
        scenario = shifts.portraits.PORTRAITS(
            dataloader_options,
            test_dataloader_options,
            size=config["portraits-size"],
            grayscale=config["portraits-grayscale"],
            train_ratio=0.8,
        )
    elif config["scenario"] == "OFFICEHOME":
        scenario = shifts.office_home.OFFICEHOME(
            dataloader_options,
            test_dataloader_options,
            target_name=config["officehome-target"],
            size=config["officehome-size"],
        )
    elif config["scenario"] == "OFFICE_31":
        scenario = shifts.office_31.OFFICE31(
            dataloader_options,
            test_dataloader_options,
            target_name=config["office-31-target"],
            size=config["office-31-size"],
        )
    elif config["scenario"] == "IMAGECLEFDA":
        scenario = shifts.image_clef.IMAGECLEFDA(
            dataloader_options,
            test_dataloader_options,
            preprocess=config["preprocess"],
            target_name=config["imageclef-target"],
            size=config["imageclef-size"],
        )
    elif config["scenario"] == "VISDA17":
        scenario = shifts.vis_da17.VisDA17(
            dataloader_options, test_dataloader_options, size=config["visda17-size"]
        )
    else:
        raise Exception("Unknown scenario")
    scenario = _setup_fabric_dataloaders(fabric, scenario)
    return scenario


def _setup_fabric_dataloaders(fabric, scenario):
    scenario.source_dataloader = fabric.setup_dataloaders(scenario.source_dataloader)
    scenario.target_dataloader = fabric.setup_dataloaders(scenario.target_dataloader)
    scenario.source_test_dataloader = fabric.setup_dataloaders(
        scenario.source_test_dataloader
    )
    scenario.target_test_dataloader = fabric.setup_dataloaders(
        scenario.target_test_dataloader
    )
    return scenario
