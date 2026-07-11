from PlanetAlign.datasets import *
from PlanetAlign.algorithms import *


dataset_classes_dict = {
    'foursquare-twitter': FoursquareTwitter,
    'Douban': Douban,
    'FlickrLastFM': FlickrLastFM,
    'FlickrMySpace': FlickrMySpace,
    'ACM-DBLP-A': ACM_DBLP,
    'ACM-DBLP-P': ACM_DBLP,
    'Cora': Cora,
    'ArXiv': ArXiv,
    'SacchCere': SacchCere,
    'PPI': PPI,
    'GGI': GGI,
    'DBP15K_ZH_EN': DBP15K_ZH_EN,
    'DBP15K_JA_EN': DBP15K_JA_EN,
    'DBP15K_FR_EN': DBP15K_FR_EN,
    'Italy': Italy,
    'Airport': Airport,
    'PeMS08': PeMS08,
    'phone-email': PhoneEmail,
    'Arenas': Arenas,
}

alg_classes_dict = {
    'PARROT': PARROT,
    'JOENA': JOENA
}

init_params_dict = {
    'PARROT': ['alpha', 'rwr_restart_prob', 'gamma', 'lambda_p', 'lambda_e', 'lambda_n', 'lambda_a'],
    'JOENA': ['mode', 'alpha', 'gamma_p', 'init_lambda', 'hid_dim', 'out_dim', 'lr']
}

train_params_dict = {
    'PARROT': ['max_iters_sep_rwr', 'max_iters_prod_rwr', 'inner_iters', 'outer_iters'],
    'JOENA': ['total_epochs']
}

eps_dict = {
    'PARROT': ['lambda_p', 'lambda_n', 'lambda_a'],
    'JOENA': ['gamma_p']
}
