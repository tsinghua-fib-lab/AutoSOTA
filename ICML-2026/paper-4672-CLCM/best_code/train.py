import argparse
import wandb
from omegaconf import OmegaConf

from networks.BP_network import BP_network
from networks.CSQN_network import CSQN_network
from networks.DFC_network import DFC_network
from networks.DER_network import DER_network, DERpp_network
from networks.EWC_network import EWC_network
from networks.EFC_network import EFC_network
from networks.oEWC_network import oEWC_network
from networks.SI_network import SI_network
from src.dataloaders_2 import (
    TaskILMNIST, ClassILMNIST5Task,
    TaskILCIFAR10, ClassILCIFAR5Task,
    TaskILTinyImageNet, ClassILTinyImageNet10Task
)

from src.trainers import WandBTrainerCL
from src.utils import str2bool

def parse_args():
    parser = argparse.ArgumentParser(description="Train continual learning model using CLI args.")

    # Config file (optional, CLI args override config file values)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (CLI args override config values)")

    # Network architecture & training hyperparameters:
    parser.add_argument("--run_name", type=str, default="default", help="Run name for wandb")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--mode", type=str, default="di", choices=["ndi", "di"],
                        help="whether to run with (di) or without (ndi) dynamic inversion")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")
    parser.add_argument("--loss_fn", type=str, default='ce', choices=["ce", "mse"],
                        help="whether to train with cross entropy ('ce') or mean squared error ('mse') loss")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"], help="Optimizer")
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR", help="Scheduler")
    parser.add_argument("--layer_size", type=int, default=100,
                        help="Size of the hidden layers")

    # Environment settings hyperparameters
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for saving training and evaluation logs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save", action="store_true", default=False, 
                        help="Whether to save the model")
    
    # EFC-specific hyperparameters:
    parser.add_argument("--beta_efc", type=float, default=1e-1, help="Beta parameter for EFC")
    parser.add_argument("--target_lr", type=float, default=2e-2, 
                        help="Target learning rate for EFC (needs to be < time_constant_ratio)")
    parser.add_argument("--alpha_di", type=float, default=0.0017, help="Alpha for dynamic inversion")
    parser.add_argument("--alpha_I", type=float, default=0.0017, help="Alpha for nondynamic inversion")
    parser.add_argument("--tau", type=float, default=0.032, help="tau parameter")
    parser.add_argument("--psi_lr", type=float, default=0.1, help="Learning rate for psi")
    parser.add_argument("--alpha_psi", type=float, default=0.0, help="Alpha parameter for psi")
    
    # EWC-specific hyperparameters:
    parser.add_argument("--importance_ewc", type=float, default=4.0, help="Importance parameter for EWC")

    # DER-specific hyperparameters:
    parser.add_argument("--der_alpha", type=float, default=0.5, help="Alpha parameter for DER (logit distillation weight)")
    parser.add_argument("--der_beta", type=float, default=0.5, help="Beta parameter for DER++ (replay CE weight)")

    # CSQN-specific hyperparameters:
    parser.add_argument("--csqn_M", type=int, default=20, help="M parameter for CSQN")
    parser.add_argument("--csqn_method", type=str, default="sr1", choices=["sr1", "bfgs"],
                        help="Method for CSQN ('sr1' or 'bfgs')")
    parser.add_argument("--csqn_reduce", type=str, default=None, choices=[None, "ct", "mrt"],
                        help="Reduce method for CSQN ('ct', 'mrt', or None)")
    parser.add_argument("--csqn_epsilon", type=float, default=1e-4, help="Epsilon parameter for CSQN")
    parser.add_argument("--csqn_kappa", type=float, default=1e-12, help="Kappa parameter for CSQN")
    
    # Additional parameters:
    parser.add_argument("--dt_di", type=float, default=0.02, help="dt for dynamic inversion")
    parser.add_argument("--time_constant_ratio", type=float, default=0.2, 
                        help="Time constant ratio (can be merged with dt_di)")
    parser.add_argument("--tmax_di", type=int, default=500, help="tmax for dynamic inversion")
    parser.add_argument("--k_p", type=float, default=2.0, help="Proportional gain for dynamic inversion")
    parser.add_argument("--eps", type=float, default=1e-4, 
                        help="Epsilon for convergence check (interplay with dt_di and target_lr)")
    
    # Continual learning settings:
    parser.add_argument("--method", type=str, default="efc", choices=["bp", "efc", "ewc", "oewc", "si", "der", "derpp", "csqn"],
                        help="Training method to use")
    parser.add_argument("--setting", type=str, default="ClassILCIFAR5Task",
                        choices=[
                            "TaskILMNIST", "ClassILMNIST5Task", "ClassILMNIST5Task_Encoded", "TaskILMNIST5Task_Encoded",
                            "TaskILCIFAR10", "ClassILCIFAR5Task",
                            "TaskILTinyImageNet", "ClassILTinyImageNet10Task"
                        ],
                        help="Continual learning setting to use")
    parser.add_argument("--peak", action="store_true", default=False,
                        help="Saves the peak model based on cumulative accuracy and restores it after each task")
    
    # CIFAR encoder parameters:
    parser.add_argument("--flatten_imgs", type=str, default="default",
                        help="Whether to flatten images ('true', 'false', or 'default')")
    parser.add_argument("--cnn_encoder", type=str, default="resnet18",
                        help="Encoder name (ensure first dimension is 512 for resnet18)")
    parser.add_argument("--cnn_pretrained", type=str2bool, default=True,
                        help="Whether to use pretrained weights for the CNN encoder")
    parser.add_argument("--encoder_freeze", type=str2bool, default=True,
                        help="Set to true to freeze encoder weights")


    # First pass: check if config file is provided
    args_initial, _ = parser.parse_known_args()

    # If config file provided, load it and set as defaults
    if args_initial.config is not None:
        config_from_file = OmegaConf.load(args_initial.config)
        config_dict = OmegaConf.to_container(config_from_file)
        
        # Flatten the 'parameters' section if it exists (wandb sweep format)
        if 'parameters' in config_dict:
            for key, val in config_dict['parameters'].items():
                if isinstance(val, dict) and 'value' in val:
                    config_dict[key] = val['value']
            # Optionally keep or remove the nested parameters
            # del config_dict['parameters']
        
        parser.set_defaults(**config_dict)

    # Second pass: parse all arguments (CLI args override config file)
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring unknown CLI arguments:", unknown)
    return args

def get_model(model_name: str, setting: str, config):
    """Get model based on name and setting."""
    models = {
        "bp": BP_network,
        "dfc": DFC_network,
        "der": DER_network,
        "derpp": DERpp_network,
        "ewc": EWC_network,
        "efc": EFC_network,
        "oewc": oEWC_network,
        "si": SI_network,
        "csqn": CSQN_network,
    }
    return models[model_name](config)

def get_dataset(setting: str, dataset: str, config):
    """Get dataset based on setting."""
    print(f"Getting dataset for setting: {setting}, dataset: {dataset}")

    # Map setting names to dataloader factory functions
    dataloader_map = {
        "TaskILMNIST": TaskILMNIST,
        "ClassILMNIST5Task": ClassILMNIST5Task,
        "ClassILMNIST5Task_Encoded": ClassILMNIST5Task,
        "TaskILMNIST5Task_Encoded": TaskILMNIST,
        "TaskILCIFAR10": TaskILCIFAR10,
        "ClassILCIFAR5Task": ClassILCIFAR5Task,
        "TaskILTinyImageNet": TaskILTinyImageNet,
        "ClassILTinyImageNet10Task": ClassILTinyImageNet10Task,
    }

    if setting not in dataloader_map:
        raise ValueError(f"Unknown setting: {setting}. Available: {list(dataloader_map.keys())}")

    return dataloader_map[setting](config).get_all_tasks_dataloaders()
    

def main():
    args = parse_args()
    
    # Update args with sweep values if running under wandb:
    if wandb.run is not None:
        sweep_config = dict(wandb.config)
        for key, value in sweep_config.items():
            setattr(args, key, value)
    
    # Both EFC and BP train on GPU
    args.device = "cuda"

    # Configure settings based on the chosen data modality
    setting_configs = {
        # MNIST: no encoder needed, 5 tasks x 2 classes, input=784
        "TaskILMNIST": {"use_cnn_encoder": False, "num_tasks": 5, "classes_per_task": 2, "input_dim": 784, "output_dim": 10},
        "ClassILMNIST5Task": {"use_cnn_encoder": False, "num_tasks": 5, "classes_per_task": 2, "input_dim": 784, "output_dim": 10},
        "ClassILMNIST5Task_Encoded": {"use_cnn_encoder": True, "flatten_imgs": False, "num_tasks": 5, "classes_per_task": 2, "input_dim": 512, "output_dim": 10},
        "TaskILMNIST5Task_Encoded": {"use_cnn_encoder": True, "flatten_imgs": False, "num_tasks": 5, "classes_per_task": 2, "input_dim": 512, "output_dim": 10},
        # CIFAR10: encoder needed, 5 tasks x 2 classes, input=512 (ResNet embedding)
        "TaskILCIFAR10": {"use_cnn_encoder": True, "num_tasks": 5, "classes_per_task": 2, "input_dim": 512, "output_dim": 10},
        "ClassILCIFAR5Task": {"use_cnn_encoder": True, "num_tasks": 5, "classes_per_task": 2, "input_dim": 512, "output_dim": 10},
        # TinyImageNet: encoder needed, 10 tasks x 20 classes, input=512 (ResNet embedding)
        "TaskILTinyImageNet": {"use_cnn_encoder": True, "num_tasks": 10, "classes_per_task": 20, "input_dim": 512, "output_dim": 200},
        "ClassILTinyImageNet10Task": {"use_cnn_encoder": True, "num_tasks": 10, "classes_per_task": 20, "input_dim": 512, "output_dim": 200},
    }

    if "MNIST" in args.setting:
        args.dataset = "MNIST"
    elif "CIFAR" in args.setting:
        args.dataset = "CIFAR10"
    elif "TinyImageNet" in args.setting:
        args.dataset = "TinyImageNet"

    cfg = setting_configs[args.setting]
    args.use_cnn_encoder = cfg["use_cnn_encoder"]
    args.num_tasks = cfg["num_tasks"]
    args.classes_per_task = cfg["classes_per_task"]
    args.layers = [cfg["input_dim"], args.layer_size, args.layer_size, cfg["output_dim"]]

    # Convert the Namespace to an OmegaConf config object.
    config = OmegaConf.create(vars(args))

    # Force SGD for BP method
    if config.method == "bp":
        config.optimizer = "SGD"

    print("Final configuration:")
    print("HERE!")
    print(OmegaConf.to_yaml(config))
    
    model = get_model(config.method, config.setting, config)
    tasks_dataloaders = get_dataset(config.setting, config.dataset, config)
    project_name = f"{config.setting}_{config.dataset}_incremental_learning_baselines"
    
    wandb.init(project=project_name, 
        entity="equilibrium-fisher-control",
        config=OmegaConf.to_container(config))
    
    trainer = WandBTrainerCL(model, tasks_dataloaders, config)
    trainer.train()

if __name__ == "__main__":
    main()