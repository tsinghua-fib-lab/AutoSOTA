"""
Original code from: https://github.com/gengchenmai/csp

Original source:
Mai, Gengchen; Lao, Ni; He, Yutong; Song, Jiaming; Ermon, Stefano.
"CSP: Self-Supervised Contrastive Spatial Pre-Training for Geospatial-Visual Representations."
Proceedings of the 40th International Conference on Machine Learning (ICML), 2023.
"""


def get_paths(variable_name):
    paths = {
        "mask_dir": "../data/",
        "inat_2018_data_dir": "../geo_prior_data_csp/data/inat_2018/",
        "fmow_data_dir": "../geo_prior_data_csp/data/fmow/",
    }
    return paths[variable_name]
