
# coding: utf-8
import os
import numpy as np
import pydicom
import pandas as pd
import re
from warnings import warn
import parsing
from typing import List, Dict

class Dir():
    '''base class for directories
    methods:
        parse_name -- to be defined in subclasses
    
    properties:
        content    -- a pandas.Series of filenames indexed by slice number
    '''
    def __init__(self, path):
        """":param path: directory location
        """
        self.path = path
    
    def __repr__(self):
        return 'Directory: {}'.format(self.path)
        
    @classmethod
    def parse_name(cls, filename):
        raise NotImplementedError

    def _list_to_series(self):
        'returns a pandas.Series of filenames indexed by slice number'
        if not hasattr(self, 'series_name'):
            series_name = os.path.basename(os.path.dirname(self.path))
        ser = pd.Series(os.listdir(self.path))
        ser.index = ser.map(self.parse_name).tolist()
        ser.name = series_name
        return ser

    @property
    def content(self):
        self._content = self._list_to_series()
        self._content = self._content.map(lambda x: os.path.join(self.path, x))
        return self._content

    def __len__(self):
        return len(self.content)

    
class DicomDir(Dir):
    @classmethod
    def parse_name(cls, fn_dcm):
        """Parses the DICOM file name
        :param fn_dcm: filepath to DICOM file
        :return: integer for slice number, or None if failed to parse"""

        matches = re.match('([\d]+).dcm', fn_dcm)
        if matches is not None:
            slice_num = matches.groups()[0]
            slice_num = int(slice_num)
            return slice_num
        else:
            warn(f'unrecognized file:\t{fn_dcm}')
            return None

        
class ContourDir(Dir):
    @classmethod
    def parse_name(cls, fn_contour: str):
        """Parses the contour file name
        :param fn_contour: filepath to contour file
        :return: integer for slice number, or None if failed to parse"""

        matches = re.match('IM-[\d]+-([\d]+)-[a-z]contour-manual.txt', fn_contour)
        if matches is not None:
            slice_num = matches.groups()[0]
            slice_num = int(slice_num)
            return slice_num
        else:
            warn(f'unrecognized file:\t{fn_contour}')
            return None


def match_case_filenames(dirname_dicom: str, 
                         dirname_i_contour:str,
                         join:str='inner') -> pd.DataFrame:
    '''matches slices in a pair of DICOM and i-contour directories
    for one series/case
    :param dirname_dicom:      path to dicom series directory
    :param dirname_i_contour:  path to i-contour series directory
    :param join:               join type (default: inner)
    :return filenames_matched: a pandas.DataFrame with matched locations
    '''
    filenames_dicom = DicomDir(dirname_dicom).content
    filenames_icontour = ContourDir(dirname_i_contour).content
    filenames_matched = pd.concat([filenames_dicom, filenames_icontour],
                                 axis=1, join=join)
    if len(filenames_matched) < len(filenames_icontour):
        warn(f'some dicom slices are missing: {filenames_matched.shape[0]} ' +
              f'matches out of {filenames_icontour.shape[0]} contours')
    
    filenames_matched.index.name = 'slice_id'
    filenames_matched.reset_index(inplace=True)
    return filenames_matched


def read_slice_with_annotations(slicedict: Dict, with_contour=False) -> Dict:
    """produces a dictionary with keys: [image, i-mask, i-contour]
    given a dictionary with disk locations of an image and a mask
    """
    dcm = parsing.parse_dicom_file(slicedict['dicoms'])
    height, width = dcm['pixel_data'].shape
    
    contour = parsing.parse_contour_file(slicedict['i-contours'])
    mask = parsing.poly_to_mask(contour, width, height)

    result = {'image': dcm['pixel_data'], 
              'imask': mask}

    if with_contour:
        result['icontour'] = contour

    return result


def get_slice_set(metadata: pd.DataFrame, dir_data = 'final_data') -> pd.DataFrame:
    '''construct a table with paired image & annotation  paths
    for a set of series/cases listed in the metadata DataFrame
    :param metadata:   metadata from link.csv
    :param dir_data:   top-level data directory
    :return filenames: DataFrame of filenames for images and annotations 
                       as well as case and slide ids
    '''
    filenames = []
    for ii, vv in metadata.iterrows():
        dirname_dicom =  '{}/dicoms/{}'.format(dir_data, vv['patient_id'])
        dirname_contours =  '{}/contourfiles/{}'.format(dir_data, vv['original_id'])
        dirname_i_contour =  '{}/i-contours/'.format(dirname_contours)
        # dirname_o_contour =  '{}/o-contours/'.format(dirname_contours)
        dir_filenames = match_case_filenames(dirname_dicom, 
                             dirname_i_contour)
        # add a case-wide identifier
        dir_filenames.loc[:,'original_id'] = vv['original_id']
        filenames.append(dir_filenames)
        
    filenames = pd.concat(filenames).reset_index(drop=True)
    return filenames


if __name__ == '__main__':

    dir_data = 'final_data'
    fn_link = f'{dir_data}/link.csv'

    metadata = pd.read_csv(fn_link)

    filenames = get_slice_set(metadata, dir_data)

    print( f'{filenames.shape[0]} files parsed')

    print(filenames.head())

    print('reading first 5 slices')
    for kk, slicedict in filenames.iterrows():
        if kk>=5:
            break
        sample = read_slice_with_annotations(slicedict)
        print(f'slice {kk}')

    print("done")
