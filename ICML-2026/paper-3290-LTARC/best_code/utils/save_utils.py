import argparse
import os
import datetime
import json
import subprocess
import logging


def generate_folder_path(base_folder, exp_folder_name):
    """Generate a folder path with the current timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d__%H_%M_%S")
    return os.path.join(os.getcwd(), base_folder, exp_folder_name, timestamp)


def set_out_dir(args):
    folder = generate_folder_path("result", args.exp_folder_name)
    os.makedirs(folder, exist_ok=True)
    return folder


def get_full_path(folder, name, prefix=''):
    return os.path.join(folder, (prefix + '_' + name) if prefix else name)


def save_hyperparams(folder, args, prefix=''):
    with open(get_full_path(folder, 'config.json', prefix), 'w') as handle:
        json.dump(vars(args), handle, indent='\t')


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_git_diff() -> str:
    diff = subprocess.check_output(['git', 'diff', 'HEAD']).decode('ascii').strip()
    if diff:
        logging.warning('There are uncommitted changes in the code.')
    return diff


def save_git_hash(folder, prefix=''):
    try:
        with open(get_full_path(folder, 'git.txt', prefix), 'w', encoding='utf-8') as handle:
            handle.write(get_git_revision_hash() + '\n')
            handle.write(get_git_diff())
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning('Not a git repository. Skipping git hash save.')


def save_logging(folder, args, prefix=''):
    handlers = [
        logging.FileHandler(get_full_path(folder, 'debug.log', prefix)),
    ]
    if args.print_logs:
        handlers += [logging.StreamHandler()]
    if args.debug:
        logging.basicConfig(handlers=handlers, encoding='utf-8', level=logging.DEBUG)
    else:
        logging.basicConfig(handlers=handlers, encoding='utf-8', level=logging.INFO)

    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)


def save_setup(args):
    out_dir = set_out_dir(args)
    save_logging(out_dir, args)
    save_hyperparams(out_dir, args)
    save_git_hash(out_dir)
    return out_dir


def add_specific_args(parent_parser):
    parser = parent_parser.add_argument_group("Save utils")

    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, default=False,
                        help='logging in debug mode')
    parser.add_argument('--print_logs', action=argparse.BooleanOptionalAction, default=False,
                        help='print logging')

    parser.add_argument('--use_default_folder', action=argparse.BooleanOptionalAction, default=True,
                        help='use default folder for logging')
    parser.add_argument('--path_to_save', default='',
                        help='specify folder path (if --no-default_folder)')
    parser.add_argument('--exp_folder_name', default='exp',
                        help='name of experiment folder (default: %(default)s)')
    return parent_parser
