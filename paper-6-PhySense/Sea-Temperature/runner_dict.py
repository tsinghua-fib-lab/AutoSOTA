from runners import exp_senseiver_rect_flow_sea_walk
from runners import exp_senseiver_rect_flow_sea

def get_runner(args, config):
    model_dict = {
        'exp_senseiver_rect_flow_sea': exp_senseiver_rect_flow_sea,
        'exp_senseiver_rect_flow_sea_walk': exp_senseiver_rect_flow_sea_walk,
    }
    return model_dict[args.runner].Diffusion(args, config)