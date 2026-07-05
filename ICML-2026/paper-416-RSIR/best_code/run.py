import os
import yaml
import quickstart
import pprint

from utils import *

if __name__ == '__main__':
    parser = get_default_parser()
    config = vars(parser.parse_args())

    config = load_config(config)

    setup_environment(config['train'])

    if config.get('mode') == 'recommendation':
        quickstart.run_recommender(config)
    elif config.get('mode') == 'generation':
        quickstart.run_generation(config)
    else:
        raise ValueError(f"Invalid mode: '{config.get('mode')}'. Mode must be 'recommendation' or 'generation'.")

