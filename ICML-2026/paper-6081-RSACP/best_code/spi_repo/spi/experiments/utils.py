import yaml
list_args = ['methods', 'alpha']


def load_config(config_path):
    config = load_yml_config(config_path)
    params = config['run_params']
    real_data_params = config['real_data_params']
    params = get_all_params(params)
    return real_data_params, params


def get_all_params(config):
    list_params = get_list_params(config)
    if len(list_params) == 0:
        return [config.copy()]
    all_params = []
    param = list_params[0]
    param_values = config[param]
    curr_config = config.copy()
    for v in param_values:
        curr_config[param] = v
        curr_params = get_all_params(curr_config)
        all_params.extend(curr_params)
    return all_params


def load_yml_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def get_list_params(params):
    list_params = []
    for k, v in params.items():
        if isinstance(v, list) and k not in list_args:
            list_params.append(k)
    return list_params

