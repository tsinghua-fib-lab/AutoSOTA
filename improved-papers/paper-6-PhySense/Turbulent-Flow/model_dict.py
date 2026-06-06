from models import physense_for_pipe_crossattn, physense_for_pipe_crossattn_walk

import operator
from functools import reduce

def get_model(config):
    model_dict = {
        'physense_for_pipe_crossattn': physense_for_pipe_crossattn,
        'physense_for_pipe_crossattn_walk': physense_for_pipe_crossattn_walk,
    }

    if config.model.name == 'LSM_Irregular_Geo' or config.model.name == 'FNO_Irregular_Geo' or config.model.name == 'FNO_Irregular_Geo_adaptive_weights':
        return model_dict[config.model.name].Model(args=config).cuda(), model_dict[config.model.name].IPHI().cuda()
    else:
        return model_dict[config.model.name].Model(args=config).cuda()

# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c