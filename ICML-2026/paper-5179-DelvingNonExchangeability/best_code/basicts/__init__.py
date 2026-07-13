import yaml
import os
import shutil

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
config_file = os.path.join(base_dir, 'config.yaml')

default_config_file = os.path.join(base_dir, 'default_config.yaml')
if not os.path.exists(config_file):
    shutil.copy(default_config_file, config_file)

with open(config_file, 'r') as fp:
    config = yaml.load(fp, Loader=yaml.FullLoader)
for k, v in config.items():
    if k in ['config_dir', 'data_dir', 'logs_dir']:
        config[k] = os.path.join(base_dir, v)
