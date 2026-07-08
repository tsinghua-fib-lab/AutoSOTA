import argparse
import time
from parsers.parser import Parser
from parsers.config import get_config, print_cfg
from trainer import Trainer, Trainer_resume
from sampler import Sampler, Sampler_mol


def main(work_type_args):
    args = Parser().parse()
    config = get_config(args.config, args.seed)
    if args.print_cfg:
        print_cfg(config)

    # -------- Train --------
    if work_type_args.type == 'train':
        trainer = Trainer(config) 
        config.ckpt = trainer.train(time.strftime('%b%d-%H:%M:%S', time.gmtime()))

    # -------- Resume train --------
    elif work_type_args.type == 'resume':
        trainer = Trainer_resume(config) 
        config.ckpt = trainer.train()

    # -------- Generation --------
    elif work_type_args.type == 'sample':
        sampler = Sampler_mol(config) if config.data.data in ['QM9', 'ZINC250k'] else Sampler(config) 
        sampler.sample()

    else:
        raise ValueError(f'Wrong type : {work_type_args.type}')

if __name__ == '__main__':
    work_type_parser = argparse.ArgumentParser()
    work_type_parser.add_argument('--type', type=str, required=True)
    main(work_type_parser.parse_known_args()[0])
