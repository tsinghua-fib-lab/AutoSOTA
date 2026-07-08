import json
import argparse
from trainer import train

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args_dict = vars(args) # Converting argparse Namespace to a dict.
    
    # Save command line device if specified
    cmd_device = None
    if args_dict.get('device') is not None:
        # Convert device strings to integers
        cmd_device = [int(d) for d in args_dict['device']]
    
    # Merge JSON config (JSON values take precedence for most params)
    args_dict.update(param)
    
    # Override with command line device if specified
    if cmd_device is not None:
        args_dict['device'] = cmd_device
    
    # Ensure device is in correct format (list of ints)
    if isinstance(args_dict.get('device'), list) and len(args_dict.get('device', [])) > 0:
        if isinstance(args_dict['device'][0], str):
            args_dict['device'] = [int(d) for d in args_dict['device']]

    train(args_dict)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/simplecil/simplecil.json',
                        help='Json file of settings.')
    parser.add_argument('--device', type=str, nargs='+', default=None,
                        help='Device(s) to use. Can specify multiple devices, e.g., --device 0 or --device 0 1. Use -1 for CPU.')
    return parser

if __name__ == '__main__':
    main()
