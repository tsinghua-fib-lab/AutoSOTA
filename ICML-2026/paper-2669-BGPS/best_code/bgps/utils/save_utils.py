import json
import torch
import errno
import os
import os.path as op
from .dist_utils import is_master, get_world_size, barrier, get_rank

def mkdir(path):
    # if it is the current folder, skip.
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def save_predict(prediction, filename):
    with open(filename, 'w') as fp:
        json.dump(prediction, fp, indent=4)

def concat_cache_files_list(cache_files, predict_file):
    results = []
    for f in cache_files:
        temp = json.load(open(f))
        results += temp
        os.remove(f)

    save_predict(results,predict_file)
    return
def concat_cache_files_dict(cache_files, predict_file):
    results = {}
    for f in cache_files:
        temp = json.load(open(f))
        results.update(temp)
        os.remove(f)

    save_predict(results,predict_file)
    return


def save_predict_ddp(results, predict_file, concat=True):
    world_size = get_world_size()

    if world_size == 1:
        save_predict(results, predict_file)
        print("Inference file saved")
        return

    else:
        cache_file = op.splitext(predict_file)[0] \
                     + f'_{get_rank()}_{world_size}' \
                     + op.splitext(predict_file)[1]

        save_predict(results, cache_file)
        barrier()
        if concat:
            if is_master():

                cache_files = [op.splitext(predict_file)[0] + '_{}_{}'.format(i, world_size) + \
                               op.splitext(predict_file)[1] for i in range(world_size)]
                if isinstance(results, list):
                    concat_cache_files_list(cache_files, predict_file)
                elif isinstance(results, dict):
                    concat_cache_files_dict(cache_files, predict_file)
                else:
                    ValueError("results are not list or dict")

                print("Inference file saved")

            # to ensure the master running time
            barrier()
    return