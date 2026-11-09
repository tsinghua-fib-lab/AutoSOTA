import pickle
import argparse
import traceback
from pathlib import Path
import sys
import os


class CustomUnpickler(pickle.Unpickler):
    '''
    With this unpickler, 
    you can unpickle objects defined in your origin __main__
    '''
    def __init__(self, f, main):
        super().__init__(f)
        if main[-3:] == '.py':
            main = main[:-3]
        self.main = main
    
    def find_class(self, module, name):
        if module == "__main__":
            module = self.main
        return super().find_class(module, name)


def ddp_worker(path, rank, origin):
    origin = Path(origin)
    main_file = origin.stem
    main_dir = origin.parent
    sys.path = sys.path[1:] # prevent internal import in kuma_utils
    sys.path.append(str(main_dir)) # add origin directory 

    with open(path, 'rb') as f:
        unpickler = CustomUnpickler(f, main=main_file)
        ddp_tmp = unpickler.load()
    trainer = ddp_tmp['trainer']
    dist_url = ddp_tmp['dist_url']
    loader =  ddp_tmp['loader']
    loader_valid = ddp_tmp['loader_valid']
    num_epochs = ddp_tmp['num_epochs']
    
    assert rank < trainer.world_size
    trainer._train_ddp(rank, dist_url, loader, loader_valid, num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--origin', type=str, required=True)
    opt = parser.parse_args()
    try:
        ddp_worker(opt.path, opt.local_rank, opt.origin)
    except Exception as e:
        print(traceback.format_exc())
        if Path(opt.path).exists():
            Path(opt.path).unlink()
