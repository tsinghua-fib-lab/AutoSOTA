import argparse
import os
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='Directory containing configuration files.')
    args = parser.parse_args()

    yaml_files = [f for f in os.listdir(args.dir) if f.endswith('.yaml')]

    for f in yaml_files:
        file_path = os.path.join(args.dir, f)
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        modified = False

        if 'optim' in data and data['optim'].get('max_epoch') == 400:
            data['optim']['max_epoch'] = 300
            modified = True

        if modified:
            with open(file_path, 'w') as file:
                yaml.safe_dump(data, file)

if __name__ == '__main__':
    main()

