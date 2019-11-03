import torch
import os
import sys
import numpy as np
import soundfile
import librosa
import h5py
import math
import pandas as pd
from sklearn import metrics
import logging
import matplotlib.pyplot as plt
import config

def mixup_data(x, y, alpha=0.2):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        while True:
            lam = np.random.beta(alpha, alpha)
            if lam > 0.65 or lam < 0.35 :
                break
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(class_criterion, pred, y_a, y_b, lam):
    return lam * class_criterion(pred, y_a) + (1 - lam) * class_criterion(pred, y_b)
def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split('/')[-1]
    name = os.path.splitext(name_ext)[0]
    return name


def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging
   

def read_audio(audio_path, target_fs=None):
    (audio, fs) = soundfile.read(audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
        
    return audio, fs

    
def pad_truncate_sequence(x, max_len):
# Data length Regularization
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]
    
   
    
def scale(x, mean, std):
    return (x - mean) / std
    
    
def inverse_scale(x, mean, std):
    return x * std + mean
    
        
        
def read_metadata(metadata_path):
    '''Read metadata from a csv file. 
    '''
    
    df = pd.read_csv(metadata_path,header=None, names=["filename","start","end","label"], sep='\t')
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
    
    
def sparse_to_categorical(x, n_out):

    x = np.array(x,dtype=int)
    shape = x.shape
    x = x.flatten()
    N = len(x)
    x_categ = np.zeros((N,n_out))
    x_categ[np.arange(N), x] = 1
    return x_categ.reshape((shape)+(n_out,))
    
    
