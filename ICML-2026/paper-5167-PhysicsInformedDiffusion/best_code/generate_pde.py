import yaml
from argparse import ArgumentParser
from scripts import generate_poisson_hat, generate_helmholtz_hat, generate_ns_nonbounded_hat
if __name__ == "__main__":
    parser = ArgumentParser(description='Generate PDE file')
    parser.add_argument('--config', type=str, help='Path to config file')
    options = parser.parse_args()
    config_path = options.config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    name = config['data']['name']
    if name == 'Poisson_fourier':
        print('Solving Poisson equation...')
        generate_poisson_hat(config)
    elif name == 'Helmholtz_fourier':
        print('Solving Helmholtz equation...')
        generate_helmholtz_hat(config)
    elif name == 'NS-NonBounded_fourier':
        print('Solving non-bounded NS equation...')
        generate_ns_nonbounded_hat(config)
    else:
        print('PDE not found')
        exit(1)
