import torch
import datetime
import os
import shutil
import itertools
import json


now = lambda: datetime.datetime.now()  # .strftime('%H:%d:%S')


flatten = lambda x: list(itertools.chain.from_iterable(x))


def dump_obj_with_dict(obj, save_to_path):
    if isinstance(obj, dict):
        d = obj
        d['dump_time'] = now()
    else:
        obj.dump_time = now()
        d = obj.__dict__
    with open(save_to_path, 'w') as f:
        json.dump(d, f, indent=4, sort_keys=True, default=str)


def load_dict_to_obj(load_from_path):
    class A:
        pass
    args = A()
    with open(load_from_path, 'r') as f:
        args.__dict__ = json.load(f)
    return args


@torch.no_grad()
def calc_model_parameters(model):
    l = [torch.tensor(p.shape).prod() for p in model.parameters()]
    print('Parameters per Layer:', l)
    print('Total Parameters:', torch.tensor(l).sum().item())


def save_weights(dirpath, model, epoch=None, batch=None, ext_text=None):
    weights_fname = 'weights'
    if epoch is not None:
        weights_fname += '-%d' % epoch
    if batch is not None:
        # weights_fname = 'weights-%d-%d-%s.pth' % (epoch, batch, ext_text)
        weights_fname += '-%d' % batch
    if ext_text is not None:
        weights_fname += '-%s' % ext_text
    weights_fname += '.pth'

    weights_fpath = os.path.join(dirpath, weights_fname)
    torch.save({
            'batch': batch,
            'epoch': epoch,
            'state_dict': model.state_dict()
        }, weights_fpath)
    print('saved weights to:', weights_fpath)
    shutil.copyfile(weights_fpath, os.path.join(dirpath, 'latest.th'))


def load_weights(model, fpath, device='cuda'):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath, map_location=device)

    # weights['state_dict'] = {k.replace('convnet','layers'): v for k, v in weights['state_dict'].items()}

    model.load_state_dict(weights['state_dict'])
    return model


class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
