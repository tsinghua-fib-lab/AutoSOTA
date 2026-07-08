from pathlib import Path

import numpy as np
import scipy.stats as stats

# prefix components:
space: str = '    '
branch: str = '│   '
# pointers:
tee: str = '├── '
last: str = '└── '


def tree(dir_path: str, prefix: str = '', show_hidden: bool = False):
    """A recursive generator, given a directory Path object
    will yield a visual tree structure line by line
    with each line prefixed by the same characters

    copy-pasted and adapted from
    https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
    """
    dir_path = Path(dir_path)
    contents = list(dir_path.iterdir())
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents):
        if not path.name.startswith('.') or show_hidden:
            yield prefix + pointer + path.name

        if path.is_dir():  # extend the prefix and recurse:
            extension = branch if pointer == tee else space
            # i.e. space because last, └── , above so no more |
            yield from tree(path, prefix=prefix+extension)


def flatten_dict(
        dictionary: dict,
        parent_key: str = '',
        separator: str = '_'
) -> dict[str, ...]:

    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
            
    return dict(items)


def ci2(
        array: np.ndarray, alpha: float, axis: int = 0
) -> dict[str, np.ndarray]:
    # Return two-sided frequentist CI with sample-size.
    mean = np.mean(array, axis=axis)
    std = np.std(array, axis=axis, ddof=1)

    t = stats.t.ppf(1 - alpha / 2, df=array.shape[axis])
    mean_var = std * t / np.sqrt(array.shape[axis])

    return {
        'mean': mean,
        'variance': mean_var,
        'sample_size': np.broadcast_to(array.shape[axis], mean.shape)
    }


class ListAsSet:

    @staticmethod
    def union(list1, list2):
        return list(dict.fromkeys(list1 + list2))

    @staticmethod
    def intersect(list1, list2):
        set2 = set(list2)
        return [item for item in list1 if item in set2]

    @staticmethod
    def diff(list1, list2):
        set2 = set(list2)
        return [item for item in list1 if item not in set2]

    @staticmethod
    def complement_intersect(list1, list2):
        set1, set2 = set(list1), set(list2)
        return [item for item in list1 + list2
                if (item in set1) ^ (item in set2)]


def ci2_boot(
        array: np.ndarray, alpha: float, axis: int = 0
) -> dict[str, np.ndarray]:
    # Return two-sided frequentist CI with sample-size.

    mean = np.nanmean(array, axis=axis)
    ci = stats.bootstrap(array[None, :], np.nanmean, axis=axis, confidence_level=1-alpha)

    return {
            'mean': mean,
            'low': ci.confidence_interval.low,
            'high': ci.confidence_interval.high,
            'sample_size': np.broadcast_to(array.shape[axis], mean.shape)
        }

def list_ci2_boot(
        array: np.ndarray, alpha: float, axis: int = 0
) -> dict[str, np.ndarray]:
    # Return two-sided frequentist CI with sample-size.

    mean = np.mean(array, axis=axis)
    ci = stats.bootstrap(array[None, :], np.mean, axis=axis, confidence_level=1-alpha)

    return {
            'mean': mean,
            'low': ci.confidence_interval.low,
            'high': ci.confidence_interval.high,
            'sample_size': np.broadcast_to(array.shape[axis], mean.shape)
        }