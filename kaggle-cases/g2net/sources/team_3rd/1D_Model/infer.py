from src.config import *
from src.dataset import read_data
from src.infer_helper import *
from src.config import read_config, prepare_args


if __name__ == "__main__":
    arg = prepare_args()
    Config = read_config(arg.model_config)
    train_df, test_df = read_data(Config)
    device = get_device()
    seed_torch(seed=Config.seed)
    Config.device=device
    cv_score = 0
    if arg.gen_oof:
        print("Generating OOF....")
        cv_score, oof_all = get_oof_final(train_df, Config)
    if arg.gen_test:
        print("Generating test....")
        test_avg = get_test_avg(cv_score, test_df, Config)
