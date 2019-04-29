
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

from task1 import get_slice_set
from task2 import HeartDataset, np_collate_dict_to_tuple


def test_task2_output_2_elements(batch):
    print('testing: 2 elements in batch')
    assert len(batch)==2
    print('passed')
    

def test_task2_output_1st_images(batch):
    print('testing: batch is an numpy array')
    assert isinstance(batch[0], np.ndarray)
    print('passed')

    print('testing: batch is a 2D (color or grayscale) numpy array')
    assert (len(batch[0].shape) in (2,3))
    print('passed')

def test_shuffling(hdset, shuffle=True):
    print(f"testing for shuffle={shuffle}")
    num_reps = 3
    first_batches = []
    for rep in range(num_reps):
        hdloader = DataLoader(hdset, batch_size=8, shuffle=shuffle,
                              collate_fn=np_collate_dict_to_tuple)
        for x in hdloader:
            break
        first_batches.append(x[0])

    imask_sum_first_batch = [tuple(x.sum(-1).sum(-1)) for x in first_batches]
    
    assert (len(set(imask_sum_first_batch)) == num_reps) == shuffle
    print('passed')


if __name__ == '__main__':

    dir_data = 'final_data'
    fn_link = f'{dir_data}/link.csv'

    metadata = pd.read_csv(fn_link)
    print( f'{metadata.shape[0]} series parsed')

    filenames = get_slice_set(metadata, dir_data)
    print( f'{filenames.shape[0]} files parsed')

    hdset = HeartDataset(filenames)

    # basic tests
    hdloader = DataLoader(hdset, batch_size=8, shuffle=True,
                          collate_fn=np_collate_dict_to_tuple)
    for batch in hdloader:
        break

    test_task2_output_2_elements(batch)

    test_task2_output_1st_images(batch)

    test_shuffling(hdset, shuffle = True)
    test_shuffling(hdset, shuffle = False)

