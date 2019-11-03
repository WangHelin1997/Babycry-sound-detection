import torch
import numpy as np
from feature import *
from matplotlib import pyplot as plt
import math

def read_metadata(metadata_path):
    '''Read metadata from a csv file. 
    '''
    
    df = pd.read_csv(metadata_path,header=None, names=["filename","start","end","label"], sep='\t')
    print(df)
    meta_dict = {}
    meta_dict['filename'] = np.array(
        [name for name in df['filename'].tolist()])
    
    if 'start' in df.keys():
        ax = np.array(df['start'])
        for a in range(len(ax)):
            if math.isnan(ax[a]):
                ax[a] = -1.
        meta_dict['start'] = ax
    if 'end' in df.keys():
        ax = np.array(df['end'])
        for a in range(len(ax)):
            if math.isnan(ax[a]):
                ax[a] = -1.
        meta_dict['end'] = ax
    if 'label' in df.keys():
        ax = np.array(df['label'])
        for a in range(len(ax)):
            if ax[a]!='babycry':
                ax[a] = 'None'
        meta_dict['label'] = ax

    return meta_dict

if __name__ == '__main__':
    meta_path = '/home/cdd/code/MTF-CRNN/applications/data/TUT-rare-sound-events-2017-development/generated_data/mixtures_devtest_0367e094f3f5c81ef017d128ebff4a3c/meta/event_list_devtest_babycry.csv'
    meta_path = '/home/cdd/code/MTF-CRNN/applications/data/TUT-rare-sound-events-2017-development/generated_data/mixtures_devtest_0367e094f3f5c81ef017d128ebff4a3c/meta/meta.csv'
    a = read_metadata(meta_path)
    print(a)


