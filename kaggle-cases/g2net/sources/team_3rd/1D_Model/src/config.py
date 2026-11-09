import os
from argparse import ArgumentParser
import torch

DATA_LOC = "../../data/1D_Model/" # Path to a folder with models, change if necessary

class BaseConfig:
    # logistic
    seed = 48
    target_size = 1
    kaggleDataFolder = '../../data/g2net-gravitational-wave-detection/' # Path to a folder with GW data, change if necessary
    output_dir = DATA_LOC + "/Models/"
    PL_folder = DATA_LOC + "/PL_fold/"
    whiten_train_folder = DATA_LOC + "/whiten-train-w0/"
    whiten_test_folder = DATA_LOC + "/whiten-test-w0/"
    avr_w0_path = DATA_LOC + "/avr_w0.pth"
    sim_data_path = DATA_LOC + '/GW_sim_300k.pkl'

    use_raw_wave = True
    use_checkpoint = False
    checkpoint_folder=None
    prev_model_folder = None
    debug = False
    use_subset = False
    subset_frac = 0.4
    # augmentation
    do_advance_trans = False
    cons_funcs = None
    aggr_funcs = None
    cons_func_names = None
    aggr_func_names = None
    vflip = False
    shuffle01 = False
    time_shift = False
    time_stretch = False
    shuffle_channels = False  # need normalization first
    add_gaussian_noise = False  # need normalization first
    timemask = False
    shift_channel = False
    pitch_shift = False
    use_mixup = False
    mixup_alpha = 0.1
    cropping = False
    use_MC = False
    MC_folds = 64
    # logger
    print_num_steps = 350
    use_wandb = False
    wandb_post = ""
    wandb_project = "G2Net_Rep"
    wandb_key_path = DATA_LOC + "key.txt"
    # training related
    train_folds = [0, 1, 2, 3, 4]
    optim = 'Adam'
    warmup = 0.1
    crit = 'bce'
    epochs = 6
    batch_size = 256
    lr = 1e-2
    weight_decay = 1e-4
    gradient_accumulation_steps = 1
    scheduler = 'cosineWithWarmUp'
    # SWA
    use_swa = False
    swa_lr_ratio = 0  # in terms of max lr
    swa_lr = 0
    swa_start_step_epoch = 3
    swa_anneal_ratio = 999,  # 999 means anneal til the end of the training
    # speedup
    num_workers = 7
    use_cudnn = True
    use_dp = False  # dataparallel
    use_gradScaler = True
    use_autocast = False


    # model
    channels = 16
    proba_final_layer = 0.8
    sdrop = 0
    PL_hard = False
    synthetic = False


class V2_Config(BaseConfig):
    model_version = "V2_c16_sGW_vflip_sc01_PL"
    model_module = "V2SD"
    vflip = True
    shuffle01 = True
    use_MC = True

    channels = 16

    lr = 7e-3
    checkpoint_folder = True
    epochs = 6
    proba_final_layer = 1



class V2_Config_pretrain(V2_Config):
    checkpoint_folder = None
    PL_folder = None
    epochs = 2
    wandb_post = "_pretrain"
    synthetic = True


class V2SD_Config(BaseConfig):
    model_version = "V2SD_sGW_vflip_sc01_PL"
    model_module = "V2SD"
    vflip = True
    shuffle01 = True
    channels = 32

    checkpoint_folder = True
    epochs = 6
    proba_final_layer = 0.8


class V2SD_Config_pretrain(V2SD_Config):
    checkpoint_folder = None
    PL_folder = None
    epochs = 2
    wandb_post = "_pretrain"
    synthetic = True


class resnet34_Config(BaseConfig):
    model_version = "resnet34-sGW2ep-PL-sc01-5ep"
    model_module = "resnet34"

    encoder = "resnet34"
    shuffle01 = True
    checkpoint_folder = True
    epochs = 5
    use_dp = True


class resnet34_Config_pretrain(resnet34_Config):
    checkpoint_folder = None
    PL_folder = None
    epochs = 2
    wandb_post = "_pretrain"
    synthetic = True


class M3D_Config(BaseConfig):
    model_version = "3D-1ep1e2"
    model_module = "M3D"

    vflip = True
    shuffle01 = True
    MC_folds = 64

    lr = 5e-3
    checkpoint_folder = None
    epochs = 1
    use_dp = True

    model_1D = "V2"
    model_1D_emb = 128
    model_1D_pretrain_dir = BaseConfig.output_dir + "/V2_c16_sGW_vflip_sc01_PL/"

    model_2D = 'resnet34'
    encoder = "resnet34"
    model_2D_emb = 128
    model_2D_pretrain_dir = BaseConfig.output_dir + "/resnet34-sGW2ep-PL-sc01-5ep/"

    first = 512
    ps = 0.5


class M_1D_Config(BaseConfig):
    model_version = "M-1D"
    model_module = "V2SD"
    vflip = True
    shuffle01 = True
    channels = 32
    use_MC = True

    checkpoint_folder = True
    epochs = 6
    optim = 'RangerLars'
    warmup = 0
    lr = 7e-3
    sdrop = 0.05

    use_autocast = True
    proba_final_layer = 1


class M_1D_Config_pretrain(M_1D_Config):
    checkpoint_folder = None
    PL_folder = None
    epochs = 4
    wandb_post = "_pretrain"
    optim = 'Adam'
    warmup = 0.1
    synthetic = True


class M_1D_Config_adjust(M_1D_Config):
    checkpoint_folder = True
    epochs = 3
    wandb_post = "_adjust"
    optim = 'RangerLars'
    warmup = 0
    lr = 2e-4
    crit = 'rank'
    PL_hard = True
    sdrop = 0


class M_1DC16_Config(BaseConfig):
    model_version = "M-1DC16"
    model_module = "V2SD"
    vflip = True
    shuffle01 = True
    channels = 16
    use_MC = True

    checkpoint_folder = True
    epochs = 6
    optim = 'RangerLars'
    warmup = 0
    lr = 7e-3
    sdrop = 0.05

    use_autocast = True
    proba_final_layer = 1

class M_1DC16_Config_pretrain(M_1DC16_Config):
    checkpoint_folder = None
    PL_folder = None
    epochs = 4
    wandb_post = "_pretrain"
    optim = 'Adam'
    warmup = 0.1
    synthetic = True


class M_1DC16_Config_adjust(M_1DC16_Config):
    checkpoint_folder = True
    epochs = 2
    wandb_post = "_adjust"
    optim = 'RangerLars'
    warmup = 0
    lr = 2e-4
    crit = 'rank'
    PL_hard = True
    sdrop = 0


class M_1DS32_Config(BaseConfig):
    model_version = "M-1DS32"
    model_module = "V2S"
    vflip = True
    shuffle01 = True
    channels = 32
    use_MC = True

    checkpoint_folder = True
    epochs = 6
    optim = 'RangerLars'
    warmup = 0
    lr = 7e-3
    sdrop = 0.05

    use_autocast = True


class M_1DS32_Config_pretrain(M_1DS32_Config):
    checkpoint_folder = None
    PL_folder = None
    epochs = 4
    wandb_post = "_pretrain"
    optim = 'Adam'
    warmup = 0.1
    synthetic = True


class M_1DS32_Config_adjust(M_1DS32_Config):
    checkpoint_folder = True
    epochs = 2
    wandb_post = "_adjust"
    optim = 'RangerLars'
    warmup = 0
    lr = 2e-4
    crit = 'rank'
    PL_hard = True
    sdrop = 0


class M_SD16_Config(BaseConfig):
    model_version = "M-SD16"
    model_module = "V2SD"
    vflip = True
    shuffle01 = True
    channels = 16
    use_MC = True

    checkpoint_folder = True
    epochs = 6
    optim = 'RangerLars'
    warmup = 0
    lr = 7e-3
    sdrop = 0.05
    proba_final_layer = 0.5

    use_autocast = True


class M_SD16_Config_pretrain(M_SD16_Config):
    checkpoint_folder = None
    PL_folder = None
    epochs = 4
    wandb_post = "_pretrain"
    optim = 'Adam'
    warmup = 0.1
    synthetic = True
    proba_final_layer = 1 #do pretraining without stochastic depth


class M_SD16_Config_adjust(M_SD16_Config):
    checkpoint_folder = True
    epochs = 2
    wandb_post = "_adjust"
    optim = 'RangerLars'
    warmup = 0
    lr = 2e-4
    crit = 'rank'
    PL_hard = True
    sdrop = 0


class M_SD32_Config(BaseConfig):
    model_version = "M-SD32"
    model_module = "V2SD"
    vflip = True
    shuffle01 = True
    use_MC = True
    channels = 32

    checkpoint_folder = True
    epochs = 6
    optim = 'RangerLars'
    warmup = 0
    lr = 7e-3
    sdrop = 0.05
    proba_final_layer = 0.5

    use_autocast = True


class M_SD32_Config_pretrain(M_SD32_Config):
    checkpoint_folder = None
    PL_folder = None
    epochs = 4
    wandb_post = "_pretrain"
    optim = 'Adam'
    warmup = 0.1
    synthetic = True
    proba_final_layer = 1 #do pretraining without stochastic depth


class M_SD32_Config_adjust(M_SD32_Config):
    checkpoint_folder = True
    epochs = 2
    wandb_post = "_adjust"
    optim = 'RangerLars'
    warmup = 0
    lr = 2e-4
    crit = 'rank'
    PL_hard = True
    sdrop = 0


class R_aug(BaseConfig):
    conservative_aug = []
    aggressive_aug_proba = []
    aggressive_aug = ['vflip']

    vflip = True
    vflip_proba = 0.5
    vflip_weight = 1.0
    add_gaussian_noise = False
    add_gaussian_noise_proba = 0.5
    add_gaussian_noise_weight = 1.0
    timemask = False
    timemask_proba = 0.35
    timemask_weight = 0.8
    shuffle01 = False
    shuffle01_proba = 0.35
    shuffle01_weight = 0.8
    time_shift = False
    time_shift_left = 96
    time_shift_right = 96
    time_shift_proba = 0.35
    time_shift_weight = 0.4

    shift_channel = False
    shift_channel_left = 16
    shift_channel_right = 16
    shift_channel_proba = 0.5
    shift_channel_weight = 1.0
    shift_two_channels = False  # tba
    shift_two_channels_proba = 0.5
    shift_two_channels_weight = 1.0
    reduce_SNR = False
    reduce_SNR_ratio = 0.9998
    reduce_SNR_proba = 0.5
    reduce_SNR_weight = 1.0

    time_stretch = False
    divide_std = False
    shuffle_channels = False
    pitch_shift = False
    cropping = False


class R_base(R_aug):
    PL_folder = None
    use_raw_wave = False
    do_advance_trans = True
    num_workers = 0
    weight_decay = 0
    epochs = 6
    batch_size = 256
    lr = 2e-3

    #CNN structure
    channels = 32
    reduction = 1.0
    proba_final_layer = 0.8
    CBAM_SG_kernel_size = 15

class Config_R35(R_base):
    model_version = "main_35th_GeM_vflip_shuffle01_5fold"
    model_module = 'Model1DCNNGEM'
    PL_folder = None
    # augmentation
    conservative_aug = ['vflip', 'shuffle01', ]
    aggressive_aug = []
    vflip = True
    vflip_proba = 0.5
    shuffle01 = True
    shuffle01_proba = 0.5

    # training
    epochs = 12
    lr = 5e-3


class Config_R112(R_base):
    # frequently changed
    model_version = "main_112th_V2SD_PL_6ep_5Fold"
    model_module = 'V2SD'
    PL_folder = DATA_LOC + "main_35th_GeM_vflip_shuffle01_5fold/"

    # augmentation
    conservative_aug = ['vflip', 'add_gaussian_noise']
    aggressive_aug = []
    vflip = True
    vflip_proba = 0.5
    add_gaussian_noise = True
    add_gaussian_noise_proba = 0.5

    # CNN structure
    proba_final_layer = 0.8


class Config_R120(R_base):
    # frequently changed
    model_version = "120th_V2_PL_6ep_1em3lr_32ch_vf_s01"
    model_module = 'V2SD'
    proba_final_layer = 1
    PL_folder = DATA_LOC + "main_112th_V2SD_PL_6ep_5Fold/"
    # conservative
    conservative_aug = []
    # aggressive, OneOf
    aggressive_aug_proba = 0.75
    aggressive_aug = ['vflip', 'add_gaussian_noise', 'shuffle01', 'timemask', 'time_shift', ]  # 'reduce_SNR'
    vflip = True
    vflip_weight = 1.0
    add_gaussian_noise = False
    add_gaussian_noise_weight = 1.0
    timemask = False
    timemask_weight = 1.0
    shuffle01 = True
    shuffle01_weight = 1.0
    time_shift = False
    time_shift_left = 96
    time_shift_right = 96
    time_shift_weight = 0.5
    # training
    batch_size = 64
    lr = 1e-3


class Config_R121(R_base):
    # frequently changed
    model_version = "121st_V2SD_PL_6ep_2em3lr_32ch_vf+gn+sc01+tm+ts"
    model_module = 'V2SD'
    PL_folder = DATA_LOC + "/main_112th_V2SD_PL_6ep_5Fold/"

    # conservative
    conservative_aug = []
    # aggressive, OneOf
    aggressive_aug_proba = 0.80
    aggressive_aug = ['vflip', 'add_gaussian_noise', 'shuffle01', 'timemask', 'time_shift', ]
    vflip = True
    vflip_weight = 1.0
    add_gaussian_noise = True
    add_gaussian_noise_weight = 1.0
    timemask = True
    timemask_weight = 0.8
    shuffle01 = True
    shuffle01_weight = 0.8
    time_shift = True
    time_shift_left = 96
    time_shift_right = 96
    time_shift_weight = 0.4


class Config_R122(R_base):
    # frequently changed
    model_version = "122nd_V2_PL_6ep_2em3lr_32ch_vf+gn+sc01+tm+ts"
    model_module = 'V2SD'
    proba_final_layer = 1
    PL_folder = DATA_LOC + "main_112th_V2SD_PL_6ep_5Fold/"

    # conservative
    conservative_aug = []
    # aggressive, OneOf
    aggressive_aug_proba = 0.80
    aggressive_aug = ['vflip', 'add_gaussian_noise', 'shuffle01', 'timemask', 'time_shift', ]  # 'reduce_SNR'
    vflip = True
    vflip_weight = 1.0
    add_gaussian_noise = True
    add_gaussian_noise_weight = 1.0
    timemask = True
    timemask_weight = 0.8
    shuffle01 = True
    shuffle01_weight = 0.8
    time_shift = True
    time_shift_left = 96
    time_shift_right = 96
    time_shift_weight = 0.4


class Config_R124(R_base):
    # frequently changed
    model_version = "124th_V2SDCBAM_PL_6ep_2em3lr_32ch_vf+gn+sc01+tm+ts"
    model_module = 'V2SDCBAM'
    PL_folder = DATA_LOC + "/main_112th_V2SD_PL_6ep_5Fold/"

    # conservative
    conservative_aug = []
    # aggressive, OneOf
    aggressive_aug_proba = 0.80
    aggressive_aug = ['vflip', 'add_gaussian_noise', 'shuffle01', 'timemask', 'time_shift', ]  # 'reduce_SNR'
    vflip = True
    vflip_weight = 1.0
    add_gaussian_noise = True
    add_gaussian_noise_weight = 1.0
    timemask = True
    timemask_weight = 0.8
    shuffle01 = True
    shuffle01_weight = 0.8
    time_shift = True
    time_shift_left = 96
    time_shift_right = 96
    time_shift_weight = 0.4


class Config_R133(R_base):
    # frequently changed
    model_version = "133rd_V2SD_PL_4ep_2em3lr_32ch_vf_sc01_drop05"
    model_module = 'V2SD'
    PL_folder = DATA_LOC + "/120th_V2_PL_6ep_1em3lr_32ch_vf_s01_5Fold/"

    # conservative aug
    conservative_aug = []
    # aggressive, OneOf
    aggressive_aug_proba = 2.0 / 3.0
    aggressive_aug = ['vflip', 'shuffle01']
    vflip = True
    vflip_weight = 1.0
    shuffle01 = True
    shuffle01_weight = 0.8

    epochs = 4

    # CNN structure
    proba_final_layer = 0.50


# ======================================================================================
# M-1D, M-1DS32, M-1DC16, M-SD16, M-SD32
config_dict = {
    'V2': V2_Config, 'V2_pretrain': V2_Config_pretrain,
    'resnet34': resnet34_Config, 'resnet34_pretrain': resnet34_Config_pretrain,
    'V-3D': M3D_Config,
    'V-V2SD': V2SD_Config, 'V-V2SD_pretrain': V2SD_Config_pretrain,
    'M-1D': M_1D_Config, 'M-1D_pretrain': M_1D_Config_pretrain, 'M-1D_adjust': M_1D_Config_adjust,
    'M-1DC16': M_1DC16_Config, 'M-1DC16_pretrain': M_1DC16_Config_pretrain, 'M-1DC16_adjust': M_1DC16_Config_adjust,
    'M-1DS32': M_1DS32_Config, 'M-1DS32_pretrain': M_1DS32_Config_pretrain, 'M-1DS32_adjust': M_1DS32_Config_adjust,
    'M-SD16': M_SD16_Config, 'M-SD16_pretrain': M_SD16_Config_pretrain, 'M-SD16_adjust': M_SD16_Config_adjust,
    'M-SD32': M_SD32_Config, 'M-SD32_pretrain': M_SD32_Config_pretrain, 'M-SD32_adjust': M_SD32_Config_adjust,
    "R-35": Config_R35, "R-112": Config_R112, "R-120": Config_R120, "R-121": Config_R121,
    "R-122": Config_R122, "R-124": Config_R124, "R-133": Config_R133
}


def read_config(name):
    print("Read Configuration")
    if name not in config_dict:
        print(f"Configuration {name} is not found")
        return None

    Config = config_dict[name]
    Config.model_output_folder = Config.output_dir + Config.model_version + "/"
    if Config.checkpoint_folder:
        Config.checkpoint_folder = Config.output_dir + Config.prev_model_folder + "/" \
            if Config.prev_model_folder is not None else Config.model_output_folder
    if Config.model_output_folder and not os.path.exists(Config.model_output_folder):
        os.makedirs(Config.model_output_folder)
    if Config.debug:
        Config.epochs = 1
    torch.backends.cudnn.benchmark = Config.use_cudnn
    print("Model Output Folder:", Config.model_output_folder)
    return Config


def prepare_args():
    parser = ArgumentParser()
    parser.add_argument('--model_config', type=str, help='configuration name for this run')
    parser.add_argument('--gen_oof', nargs='?', const=1, type=bool,
                        help='generate oof prediction during inference or not')
    parser.add_argument('--gen_test', nargs='?', const=1, type=bool,
                        help='generate test prediction during inference or not')
    args = parser.parse_args()
    return args
