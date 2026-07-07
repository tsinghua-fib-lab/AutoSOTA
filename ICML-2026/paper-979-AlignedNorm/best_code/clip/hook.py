from typing import Dict, Text, Callable, List
from collections import defaultdict


class HookManager(object):
    def __init__(self, hook_dict: Dict[Text, List[Callable]] = None):
        self.hook_dict = hook_dict or defaultdict(list)
        self.called = defaultdict(int)
        self.forks = dict()

    def register(self, name: Text, func: Callable):
        assert name
        found_successor = False
        for header, d in self.forks.items():
            if name.startswith(header.split('.')[0]+'.'):
                next_ = name[len(header.split('.')[0]+'.'):].split('.')[0]
                prev_ = header.split('.')[0]
                if next_.isnumeric() and  prev_ + '.' + next_ == header:
                    d.register(name[len(header)+1:], func)
                elif next_ == '*':
                    d.register(name[len(prev_ + '.*')+1:], func)
                else:
                    d.register(name[len(header)+1:], func) 
                found_successor = True
        if not found_successor:
            self.hook_dict[name].append(func)
    
    def unregister(self, name: Text, func: Callable):
        assert name
        found_successor = False
        for header, d in self.forks.items():
            if name.startswith(header.split('.')[0]+'.'):
                next_ = name[len(header.split('.')[0]+'.'):].split('.')[0]
                prev_ = header.split('.')[0]
                if next_.isnumeric() and  prev_ + '.' + next_ == header:
                    d.register(name[len(header)+1:], func)
                elif next_ == '*':
                    d.register(name[len(prev_ + '.*')+1:], func)
                else:
                    d.register(name[len(header)+1:], func) 
                found_successor = True
        if not found_successor and func in self.hook_dict[name]:
            self.hook_dict[name].remove(func)
    
    def __call__(self, name: Text, **kwargs):
        if name in self.hook_dict:
            self.called[name] += 1
            for function in self.hook_dict[name]:
                ret = function(**kwargs)
            if len(self.hook_dict[name]) > 1: 
                last = self.hook_dict[name][-1]
                # print(f'The last returned value comes from func {last}')
            return ret
        else:
            return kwargs['ret']

    def fork(self, name):
        if name in self.forks:
            raise ValueError(f'Forking with the same name is not allowed. Already forked with {name}.')
        filtered_hooks = [(k[len(name)+1:], v) for k, v in self.hook_dict.items() if k.startswith(name+'.')]
        filtered_hooks_d = defaultdict(list)
        for i, j in filtered_hooks:
            if isinstance(j, list):
                filtered_hooks_d[i].extend(j)
            else:
                filtered_hooks_d[i].append(j)
        new_hook = HookManager(filtered_hooks_d)
        self.forks[name] = new_hook
        return new_hook

    def fork_iterative(self, name, iteration):
        filtered_hooks = [(k[len(name+'.'+str(iteration))+1:], v) for k, v in self.hook_dict.items() if k.startswith(name+'.'+str(iteration)+'.')]
        filtered_hooks += [(k[len(name+'.*')+1:], v) for k, v in self.hook_dict.items() if k.startswith(name+'.*.')]
        filtered_hooks_d = defaultdict(list)
        for i, j in filtered_hooks:
            if isinstance(j, list):
                filtered_hooks_d[i].extend(j)
            else:
                filtered_hooks_d[i].append(j)
        new_hook = HookManager(filtered_hooks_d)
        self.forks[name+'.'+str(iteration)] = new_hook
        return new_hook
    
    def finalize(self):
        for name in self.hook_dict.keys():
            if self.called[name] == 0:
                raise ValueError(f'Hook {name} was registered but never used!')