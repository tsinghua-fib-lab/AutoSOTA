from .models_1d import *
from .models_2d import *
from .models_3d import *


def M1D(config):
    if config.model_module == 'V2SD':
        model = V2StochasticDepth(n=config.channels,
                                  proba_final_layer=config.proba_final_layer,
                                  sdrop=config.sdrop,
                                  use_raw_wave=config.use_raw_wave,
                                  avr_w0_path=config.avr_w0_path)
    elif config.model_module == "V2S":
        model = ModelIafossV2S(n=config.channels,
                               sdrop=config.sdrop,
                               use_raw_wave=config.use_raw_wave,
                               avr_w0_path=config.avr_w0_path)
    elif config.model_module == "Model1DCNNGEM":
        model = Model1DCNNGEM(initial_channnels=config.channels)
    elif config.model_module == 'V2SDCBAM':
        model = V2SDCBAM(n=config.channels,
                         proba_final_layer=config.proba_final_layer,
                         reduction=config.reduction,
                         CBAM_SG_kernel_size=config.CBAM_SG_kernel_size
                         )

    return model


def M2D(config):
    if config.model_module == 'resnet34':
        model = Model_2D(encoder=config.encoder,
                         use_raw_wave=config.use_raw_wave,
                         avrSpecDir=config.inputDataFolder)
    return model


def M3D(config):
    if config.model_module == "M3D":
        fold = config.fold

        config.model_module = config.model_1D
        model_1d = M1D(config)
        if config.model_1D_pretrain_dir is not None and fold is not None:
            path = f"{config.model_1D_pretrain_dir}/Fold_{fold}_best_model.pth"
            print("Loading model from path: ", path)
            checkpoint = torch.load(path, map_location='cuda:0')
            model_1d.load_state_dict(checkpoint['model_state_dict'])
        model_1d.use_raw_wave = False

        config.model_module = config.model_2D
        model_2d = M2D(config)
        if config.model_2D_pretrain_dir is not None and fold is not None:
            path = f"{config.model_2D_pretrain_dir}/Fold_{fold}_best_model.pth"
            print("Loading model from path: ", path)
            checkpoint = torch.load(path, map_location='cuda:0')
            model_2d.load_state_dict(checkpoint['model_state_dict'])
        model_2d.use_raw_wave = False

        model = Combined1D2D(model_1d, model_2d,
                             emb_1d=config.model_1D_emb,
                             emb_2d=config.model_2D_emb,
                             first=config.first,
                             ps=config.ps,
                             avrSpecDir=config.inputDataFolder)
        model.freeze_conv(req_grad=False)
        config.model_module = "M3D"
    return model


def getModel(config):
    if config.model_module in ['resnet34']:
        return M2D(config)
    elif config.model_module in ['M3D']:
        return M3D(config)
    else:
        return M1D(config)
