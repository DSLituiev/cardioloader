#!/usr/bin/env python3
# coding: utf-8

import os
import numpy as np
import pydicom
import pandas as pd
import re
from warnings import warn
import parsing
from task1 import get_slice_set, read_slice_with_annotations
from torch.utils.data import Dataset, DataLoader


class HeartDataset(Dataset):

    """An class representing a heart MRI dataset.
    """
    def __init__(self, filenames):
        self.filenames = filenames
        
    def __getitem__(self, index):
        slicedict = self.filenames.iloc[index]
        sample = read_slice_with_annotations(slicedict)
        return sample

    def __len__(self):
        return len(self.filenames)

    def __add__(self, other):
        return HeartDataset(pd.concat[self.filenames, other.filenames])


def np_collate(batch):
    """Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain arrays, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch, axis=0)
    
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_'             and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))
            return np.stack(batch, 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return (list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return np.stack(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], str):
        return batch
    elif isinstance(batch[0], dict):
        return {key: np_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], (list, tuple)):
        transposed = zip(*batch)
        return [np_collate(samples) for samples in transposed]
    raise TypeError((error_msg.format(type(batch[0]))))


def np_collate_dict_to_tuple(batch):
    """collates a batch of dictionaries with keys: ('image', 'imask') 
    to a tuple of arrays (image, imask)"""
    results = [np_collate([d[key] for d in batch]) for key in ['image', 'imask']]
    return tuple(results)# results['image'], results['imask']


# ### Note: this file uses pytorch-style Dataset and pytorch-original DataLoader classes
# Dataset embeds the table of filenames of paired slices and a method to read them for each slice (table row).
# DataLoader takes care of loading, batching, and asynchronous file reading.

if __name__ == '__main__':
    dir_data = 'final_data'
    fn_link = f'{dir_data}/link.csv'

    metadata = pd.read_csv(fn_link)
    print( f'{metadata.shape[0]} series parsed')

    filenames = get_slice_set(metadata, dir_data)
    print( f'{filenames.shape[0]} files parsed')

    hdset = HeartDataset(filenames)
    hdloader = DataLoader(hdset, batch_size=8, shuffle=True, collate_fn=np_collate_dict_to_tuple)

    # ## Run batch loader for 4 iterations
    for ii, x in enumerate(hdloader):
        print(f'loading batch {ii}')
        if ii>=4:
            break

