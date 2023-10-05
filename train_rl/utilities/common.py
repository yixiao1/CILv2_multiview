import numbers
import random 
import numpy as np
import torch
import sys

def format_number(x):
    if isinstance(x, int):
        return x
    elif isinstance(x, float):
        # return float("%.3g" % float(x))
        return float(f"{float(x):.3f}")

def recursive_format(dictionary, function):
    if isinstance(dictionary, dict):
        return type(dictionary)((key, recursive_format(value, function)) for key, value in dictionary.items())
    if isinstance(dictionary, list):
        return type(dictionary)(recursive_format(value, function) for value in dictionary)
    if isinstance(dictionary, numbers.Number):
        return function(dictionary)
    return dictionary

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

