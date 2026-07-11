"""
Generic helper functions and classes
"""

import copy as python_copy
import time

import PIL.Image
import IPython.display


class Cycle:
    "Cyclic list"
    def __init__(self, l):
        self.l = l
        self.n = len(l)
    def __getitem__(self, i):
        return self.l[i % self.n]

colors = Cycle(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'])

class DictToObj:
    "Convert a dictionary into a python object"
    def __init__(self, copy=True, **kwargs):
        "Input dictionary by values DictToObj(**dict)"
        assert type(copy)==bool
        for key in kwargs.keys():
            if copy:
                item = python_copy.deepcopy(kwargs[key])
                key = python_copy.deepcopy(key)
            else:
                item = kwargs[key]
            self[key] = item

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, item):
        setattr(self, key, item)

    def __str__(self):
        s = repr(self) + '\n'
        for k in vars(self): s+=f"{k}: {self[k]}\n"
        return s

class Timer:
    "Simple timer context manager"

    def __init__(self, name=None, verbose=True):
        self.name = name
        self.verbose=verbose

    def __enter__(self):
        self.start = time.time()
        return self# so we can access this using "with Timer as timer"

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        tag = f" ({self.name})" if self.name is not None else ""
        if self.verbose: print(f"Time elapsed{tag}: {self.interval:.4f} s")

def save_gif(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [PIL.Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='gif', append_images=imgs[1:],
                 save_all=True, duration=int(1000/fps), loop=loop)
    im = IPython.display.Image(filename=outfile)
    im.reload()
    return im



if __name__ == "__main__":

    d = {"a":[1,2,3], "b":2}

    a = DictToObj(**d)
    b = DictToObj(copy=False, **d)
    b.fun = "fun"
    b["yo"] = "yo"

    print(a,b)
    d["a"][0]=10
    print(a,b)

    with Timer(verbose=True) as timer:
        time.sleep(1)
    print(timer.interval)

    with Timer("test") as timer:
        time.sleep(1)
    print(timer.interval)