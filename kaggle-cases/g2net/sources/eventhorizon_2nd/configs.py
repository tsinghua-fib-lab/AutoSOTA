from pathlib import Path

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedKFold
from nnAudio.Spectrogram import CQT
from cwt_pytorch import ComplexMorletCWT

from kuma_utils.torch.callbacks import (
    EarlyStopping, SaveSnapshot, SaveEveryEpoch, SaveAllSnapshots)
from kuma_utils.torch.hooks import TrainHook

from datasets import G2NetDataset
from architectures import SpectroCNN, MultiInstanceSCNN
from models1d_pytorch import *
from loss_functions import BCEWithLogitsLoss
from metrics import AUC
from transforms import *


INPUT_DIR = Path('input/').expanduser()
HW_CFG = {
    'RTX3090': (16, 128, 1, 24), # CPU cores, RAM amount, GPU count, GPU RAM total
    'A100': (9, 60, 1, 40), 
}


class Baseline:
    name = 'baseline'
    seed = 2021
    train_path = INPUT_DIR/'train.csv'
    test_path = INPUT_DIR/'test.csv'
    train_cache = None # You can add the path to your dataset cache
    test_cache = None  #
    cv = 5
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    dataset = G2NetDataset
    dataset_params = dict()

    model = SpectroCNN
    model_params = dict(
        model_name='tf_efficientnet_b7',
        pretrained=True,
        num_classes=1,
        spectrogram=CQT,
        spec_params=dict(sr=2048, 
                         fmin=20, 
                         fmax=1024, 
                         hop_length=64),
    )
    weight_path = None
    num_epochs = 5
    batch_size = 64
    optimizer = optim.Adam
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)
    scheduler = CosineAnnealingWarmRestarts
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)
    scheduler_target = None
    batch_scheduler = False
    criterion = nn.BCEWithLogitsLoss()
    eval_metric = AUC().torch
    monitor_metrics = []
    amp = True
    parallel = None
    deterministic = False
    clip_grad = 'value'
    max_grad_norm = 10000
    hook = TrainHook()
    callbacks = [
        EarlyStopping(patience=5, maximize=True),
        SaveSnapshot()
    ]

    transforms = dict(
        train=None, 
        test=None,
        tta=None
    )

    pseudo_labels = None
    debug = False


class Resized08aug4(Baseline):
    name = 'resized_08_aug_4'
    model_params = dict(
        model_name='tf_efficientnet_b2',
        pretrained=True,
        num_classes=1,
        spectrogram=CQT,
        spec_params=dict(sr=2048, 
                         fmin=16, 
                         fmax=1024, 
                         hop_length=8),
        resize_img=(256, 512),
        upsample='bicubic'
    )
    transforms = dict(
        train=Compose([
            GaussianNoiseSNR(min_snr=15, max_snr=30, p=0.25)
        ]),
        test=None,
        tta=None
    )
    dataset_params = dict(
        norm_factor=[4.61e-20, 4.23e-20, 1.11e-20])
    num_epochs = 8
    scheduler_params = dict(T_0=8, T_mult=1, eta_min=1e-6)
    optimizer_params = dict(lr=1e-3, weight_decay=1e-6)


class Nspec12(Resized08aug4):
    name = 'nspec_12'
    model_params = dict(
        model_name='tf_efficientnet_b2',
        pretrained=True,
        num_classes=1,
        spectrogram=ComplexMorletCWT,
        spec_params=dict(fs=2048, 
                         lower_freq=16, 
                         upper_freq=1024, 
                         wavelet_width=3,
                         stride=8,
                         n_scales=256),
        resize_img=(256, 512),
        upsample='bicubic'
    )
    transforms = dict(
        train=Compose([
            BandPass(lower=12, upper=512),
            GaussianNoiseSNR(min_snr=15, max_snr=30, p=0.5),
        ]),
        test=BandPass(lower=12, upper=512),
        tta=BandPass(lower=12, upper=512)
    )


class Nspec12arch0(Nspec12):
    name = 'nspec_12_arch_0'
    model_params = dict(
        model_name='densenet201',
        pretrained=True,
        num_classes=1,
        spectrogram=ComplexMorletCWT,
        spec_params=dict(fs=2048, 
                         lower_freq=16, 
                         upper_freq=1024, 
                         wavelet_width=8,
                         stride=8,
                         n_scales=256),
        resize_img=(256, 512),
        upsample='bicubic'
    )
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)


class Nspec16(Resized08aug4): 
    name = 'nspec_16'
    model_params = dict(
        model_name='tf_efficientnet_b2',
        pretrained=True,
        num_classes=1,
        spectrogram=ComplexMorletCWT,
        spec_params=dict(fs=2048, 
                         lower_freq=16, 
                         upper_freq=1024, 
                         wavelet_width=8,
                         trainable_width=True, 
                         stride=4,
                         n_scales=128),
        resize_img=(128, 1024),
        custom_classifier='gem', 
        upsample='bicubic'
    )
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=12, upper=512),
            GaussianNoiseSNR(min_snr=15, max_snr=30, p=0.5),
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=12, upper=512),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=12, upper=512),
        ])
    )
    dataset_params = dict()


class Nspec16spec13(Nspec16):
    name = 'nspec_16_spec_13'
    model_params = dict(
        model_name='tf_efficientnet_b2',
        pretrained=True,
        num_classes=1,
        spectrogram=ComplexMorletCWT,
        spec_params=dict(fs=2048, 
                         lower_freq=16, 
                         upper_freq=512, 
                         wavelet_width=8,
                         trainable_width=True, 
                         stride=4,
                         n_scales=128),
        custom_classifier='gem', 
        upsample='bicubic',
    )
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=12, upper=360, order=4),
            GaussianNoiseSNR(min_snr=15, max_snr=30, p=0.5),
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=12, upper=360, order=4),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=12, upper=360, order=4),
        ])
    )


class Nspec16arch17(Nspec16):
    name = 'nspec_16_arch_17'
    model_params = dict(
        model_name='tf_efficientnet_b7_ns',
        pretrained=True,
        num_classes=1,
        spectrogram=ComplexMorletCWT,
        spec_params=dict(fs=2048, 
                         lower_freq=16, 
                         upper_freq=1024, 
                         wavelet_width=8,
                         trainable_width=True, 
                         stride=4,
                         n_scales=128),
        resize_img=None,
        custom_classifier='gem', 
        upsample='bicubic'
    )
    batch_size = 32
    optimizer_params = dict(lr=5e-4, weight_decay=1e-6)


class Nspec21(Resized08aug4):
    name = 'nspec_21'
    model_params = dict(
        model_name='tf_efficientnet_b4_ns',
        pretrained=True,
        num_classes=1,
        spectrogram=ComplexMorletCWT,
        spec_params=dict(fs=2048, 
                         lower_freq=16, 
                         upper_freq=1024, 
                         wavelet_width=8,
                         trainable_width=True, 
                         stride=4,
                         n_scales=256),
        resize_img=(256, 1024),
        custom_classifier='gem', 
        upsample='bicubic'
    )
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=12, upper=512),
            GaussianNoiseSNR(min_snr=15, max_snr=30, p=0.5),
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=12, upper=512),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=12, upper=512),
        ])
    )
    dataset_params = dict()


class Nspec22(Resized08aug4):
    name = 'nspec_22'
    model_params = dict(
        model_name='tf_efficientnet_b2',
        pretrained=True,
        num_classes=1,
        spectrogram=WaveNetSpectrogram,
        spec_params=dict(
            base_filters=128,
            wave_layers=(10, 6, 2),
            kernel_size=3,
        ), 
        resize_img=None,
        custom_classifier='gem', 
        upsample='bicubic'
    )
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=16, upper=512),
            GaussianNoiseSNR(min_snr=15, max_snr=30, p=0.5),
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=16, upper=512),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=16, upper=512),
        ]),
    )
    dataset_params = dict()


class Nspec22aug1(Nspec22):
    name = 'nspec_22_aug_1'
    model_params = Nspec22.model_params.copy()
    model_params['spec_params'] =dict(
        wave_block='none',
        base_filters=128,
        wave_layers=(10, 6, 2),
        kernel_size=3,
    )
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=16, upper=512),
            GaussianNoiseSNR(min_snr=15, max_snr=30, p=0.5),
            FlipWave(p=0.5)
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=16, upper=512),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=16, upper=512),
        ]),
    )


class Nspec22arch2(Nspec22): 
    name = 'nspec_22_arch_2'
    model_params = Nspec22aug1.model_params.copy()
    model_params['model_name'] = 'tf_efficientnet_b6_ns'
    transforms = Nspec22aug1.transforms.copy()


class Nspec22arch6(Nspec22):
    name = 'nspec_22_arch_6'
    model_params = Nspec22aug1.model_params.copy()
    model_params['model_name'] = 'densenet201'
    transforms = Nspec22aug1.transforms.copy()
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)


class Nspec22arch7(Nspec22):
    name = 'nspec_22_arch_7'
    model_params = Nspec22aug1.model_params.copy()
    model_params['model_name'] = 'tf_efficientnetv2_m'
    transforms = Nspec22aug1.transforms.copy()
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)


class Nspec22arch10(Nspec22):
    name = 'nspec_22_arch_10'
    model_params = Nspec22aug1.model_params.copy()
    model_params['model_name'] = 'resnet200d'
    transforms = Nspec22aug1.transforms.copy()
    optimizer_params = dict(lr=5e-4, weight_decay=1e-6)


class Nspec22arch12(Nspec22):
    name = 'nspec_22_arch_12'
    model_params = Nspec22aug1.model_params.copy()
    model_params['model_name'] = 'tf_efficientnetv2_l'
    transforms = Nspec22aug1.transforms.copy()
    batch_size = 32
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)


class Nspec23(Nspec22):
    name = 'nspec_23'
    model_params = dict(
        model_name='tf_efficientnet_b2',
        pretrained=True,
        num_classes=1,
        spectrogram=CNNSpectrogram,
        spec_params=dict(
            base_filters=128, 
            kernel_sizes=(32, 16, 4), 
        ),
        resize_img=None,
        custom_classifier='gem', 
        upsample='bicubic'
    )


class Nspec23arch3(Nspec23):
    name = 'nspec_23_arch_3'
    model_params = Nspec23.model_params.copy()
    model_params['spec_params'] = dict(
        base_filters=128, 
        kernel_sizes=(64, 16, 4), 
    )
    model_params['model_name'] = 'tf_efficientnet_b6_ns'
    transforms = Nspec22aug1.transforms.copy()


class Nspec23arch5(Nspec23):
    name = 'nspec_23_arch_5'
    model_params = Nspec23arch3.model_params.copy()
    model_params['model_name'] = 'tf_efficientnetv2_m'
    transforms = Nspec22aug1.transforms.copy()
    optimizer_params = dict(lr=5e-4, weight_decay=1e-6)


class Nspec25(Nspec22):
    name = 'nspec_25'
    model_params = dict(
        model_name='tf_efficientnet_b2',
        pretrained=True,
        num_classes=1,
        spectrogram=WaveNetSpectrogram,
        spec_params=dict(
            wave_block='none',
            base_filters=256,
            wave_layers=(10, 6, 2),
            downsample=4,
        ), 
        resize_img=None,
        custom_classifier='gem', 
        upsample='bicubic'
    )
    transforms = Nspec22aug1.transforms.copy()


class Nspec25arch1(Nspec25):
    name = 'nspec_25_arch_1'
    model_params = Nspec25.model_params
    model_params['model_name'] = 'tf_efficientnet_b3_ns'


class Nspec30(Nspec22):
    name = 'nspec_30'
    model_params = dict(
        model_name='tf_efficientnet_b2',
        pretrained=True,
        num_classes=1,
        spectrogram=WaveNetSpectrogram,
        spec_params=dict(
            separate_channel=True,
            in_channels=1, 
            wave_block='none',
            base_filters=128,
            wave_layers=(10, 10, 10),
            kernel_size=3,
        ),
        resize_img=None,
        custom_classifier='gem', 
        upsample='bicubic'
    )
    transforms = Nspec22aug1.transforms.copy()


class Nspec30arch2(Nspec30):
    name = 'nspec_30_arch_2'
    model_params = Nspec30.model_params.copy()
    model_params['spec_params'] = dict(
        separate_channel=True,
        in_channels=1, 
        wave_block='none',
        base_filters=128,
        wave_layers=(8, 8, 8),
        kernel_size=3,
    )
    model_params['model_name'] = 'tf_efficientnet_b6_ns'


class MultiInstance04(Nspec16):
    name = 'multi_instance_04'
    model = MultiInstanceSCNN
    model_params = dict(
        model_name='xcit_tiny_12_p16_384_dist', 
        pretrained=True,
        num_classes=1,
        spectrogram=ComplexMorletCWT,
        spec_params=dict(fs=2048, 
                         lower_freq=16, 
                         upper_freq=1024, 
                         wavelet_width=8,
                         stride=4,
                         n_scales=384),
        resize_img=(384, 768),
        n_patch=2, 
        custom_classifier='gem', 
        upsample='bicubic',
    )
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)


'''
Sequential model
'''
class Seq00(Resized08aug4):
    name = 'seq_00'
    model = ResNet1d
    model_params = dict(
        in_channels=3, 
        base_filters=64,
        kernel_size=16, 
        stride=2, 
        groups=64, 
        n_block=16, 
        n_classes=1,
        use_bn=True,
        dropout=0.2
    )
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=24, upper=350, order=4),
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=24, upper=350, order=4),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=24, upper=350, order=4),
        ])
    )
    dataset_params = dict()
    optimizer_params = dict(lr=1e-3, weight_decay=1e-6)
    num_epochs = 5
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)


class Seq02(Seq00):
    name = 'seq_02'
    model_params = Seq00.model_params.copy()
    model_params.update(dict(
        base_filters=32,
        groups=32, 
        dropout=0.0
    ))
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=24, upper=300, order=4),
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=24, upper=300, order=4),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=24, upper=300, order=4),
        ])
    )


class Seq03(Seq02):
    name = 'seq_03'
    model_params = Seq02.model_params.copy()
    model_params.update(dict(
        base_filters=128,
        groups=32,
        dropout=0.0
    ))
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=30, upper=300, order=4),
            GaussianNoiseSNR(min_snr=15, max_snr=30, p=0.5),
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=30, upper=300, order=4),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=30, upper=300, order=4),
        ])
    )
    num_epochs = 8
    scheduler_params = dict(T_0=8, T_mult=1, eta_min=1e-6)
    optimizer_params = dict(lr=1e-3, weight_decay=1e-6)
    max_grad_norm = 1000
    clip_grad = 'value'


class Seq03aug3(Seq03):
    name = 'seq_03_aug_3'
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=30, upper=300, order=4),
            FlipWave(p=0.5)
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=30, upper=300, order=4),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=30, upper=300, order=4),
        ])
    )
    num_epochs = 5
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)


class Seq09(Seq02):
    name = 'seq_09'
    model = densenet121_1d
    model_params = dict()
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=30, upper=300, order=4),
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=30, upper=300, order=4),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=30, upper=300, order=4),
        ])
    )


class Seq12(Seq02):
    name = 'seq_12'
    model = WaveNet1d
    model_params = dict(
        in_channels=3, 
        hidden_dims=(128, 256, 512, 1024),
        wave_block='simplified',
        num_classes=1,
        reinit=True,
        downsample=True,
    )
    num_epochs = 5
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)
    max_grad_norm = 1000
    clip_grad = 'value'


class Seq12arch4(Seq12):
    name = 'seq_12_arch_4'
    model_params = Seq12.model_params.copy()
    model_params.update(dict(
        kernel_size=16,
        hidden_dims=(128, 128, 256, 256, 512, 512, 1024, 1024),
        wave_layers=(12, 10, 8, 8, 6, 6, 4, 2)
    ))


'''
PSEUDO LABEL
'''
class Pseudo06(Nspec12):
    name = 'pseudo_06'
    weight_path = None
    pseudo_labels = dict(
        path=Path('results/nspec_12/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )


class Pseudo07(Nspec16):
    name = 'pseudo_07'
    weight_path = None
    pseudo_labels = dict(
        path=Path('results/nspec_16/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )


class Pseudo10(Nspec16spec13):
    name = 'pseudo_10'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_16_spec_13/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo12(Nspec12arch0):
    name = 'pseudo_12'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_12_arch_0/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo13(MultiInstance04):
    name = 'pseudo_13'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/multi_instance_04/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo14(Nspec16arch17):
    name = 'pseudo_14'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_16_arch_17/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo15(Nspec22aug1):
    name = 'pseudo_15'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_22_aug_1/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo16(Nspec22arch2):
    name = 'pseudo_16'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_22_arch_2/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo17(Nspec23arch3):
    name = 'pseudo_17'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_23_arch_3/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo18(Nspec21):
    name = 'pseudo_18'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_21/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo19(Nspec22arch6):
    name = 'pseudo_19'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_22_arch_6/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo21(Nspec22arch7):
    name = 'pseudo_21'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_22_arch_7/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo22(Nspec23arch5):
    name = 'pseudo_22'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_23_arch_5/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo23(Nspec22arch12):
    name = 'pseudo_23'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_22_arch_12/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo24(Nspec30arch2):
    name = 'pseudo_24'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_30_arch_2/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo25(Nspec25arch1):
    name = 'pseudo_25'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_25_arch_1/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class Pseudo26(Nspec22arch10):
    name = 'pseudo_26'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/nspec_22_arch_10/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 10
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)


class PseudoSeq03(Seq09):
    name = 'pseudo_seq_03'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/seq_09/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )


class PseudoSeq04(Seq03aug3):
    name = 'pseudo_seq_04'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/seq_03_aug_3/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )


class PseudoSeq07(Seq12arch4):
    name = 'pseudo_seq_07'
    criterion = BCEWithLogitsLoss(smooth_eps=0.025)
    pseudo_labels = dict(
        path=Path('results/seq_12_arch_4/predictions.npy'),
        confident_samples=None,
        hard_label=False
    )
    num_epochs = 6
    scheduler_params = dict(T_0=6, T_mult=1, eta_min=1e-6)


class Debug(Seq12):
    name = 'debug'
    debug = True
    num_epochs = 2
