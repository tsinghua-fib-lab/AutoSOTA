from models import physense_transolver_car, physense_transolver_car_walk

import operator
from functools import reduce

def get_model(key):
    model_dict = {
        'physense_transolver_car': physense_transolver_car,
        'physense_transolver_car_walk': physense_transolver_car_walk,   
    }
    
    
    return model_dict[key].Model(n_hidden=374,
                           n_layers=12,
                           space_dim=3,
                           fun_dim=0,
                           n_head=8,
                           mlp_ratio=2,
                           out_dim=1,
                           slice_num=32,
                           unified_pos=1).cuda()
    
    
# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c