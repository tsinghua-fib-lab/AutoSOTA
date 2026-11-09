from src.util import *
from src.dataset import read_data, read_synthetic
from src.train_helper import training_loop
from src.config import read_config, prepare_args

if __name__ == "__main__":
    arg = prepare_args()
    Config = read_config(arg.model_config)
    if Config is not None:
        print("Training with ", arg.model_config, " Configuration")
        train_df, test_df = read_data(Config)
        device = get_device()
        Config.device = device
        SIGNAL_DICT = read_synthetic(Config)
        seed_torch(seed=Config.seed)
        training_loop(train_df, Config, SIGNAL_DICT)