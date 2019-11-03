import os
import sys
import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import (create_folder, get_filename, create_logging, read_audio, sparse_to_categorical, pad_truncate_sequence)
from data_generator import DataGenerator, EvaluationDataGenerator
from net import Cnns, Crnn
from losses import nll_loss
from evaluate import Evaluator, StatisticsContainer
from pytorch_utils import move_data_to_gpu, forward
import config
from feature import LogMelExtractor

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def test_one(checkpoint_path, model_type, cuda, test_fold, test_wav, test_segment):
#     test_bgn_time = time.time()
    Model = eval(model_type)
    model = Model(config.classes_num, activation='logsoftmax')         
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    if cuda:
        model.cuda()
#     test_fin_time = time.time()
#     test_time = test_fin_time - test_bgn_time
#     print(test_time)
    audio_path = os.path.join(test_fold, test_wav)
    sample_rate = config.sample_rate
    window_size = config.window_size
    hop_size = config.hop_size
    mel_bins = config.mel_bins
    fmin = config.fmin
    fmax = config.fmax
    frames_per_second = config.frames_per_second
    frames_num = config.frames_num
    frames_num_clip = config.frames_num_clip
    total_samples = config.total_samples
    lb_to_idx = config.lb_to_idx
    audio_duration_clip = config.audio_duration_clip
    audio_stride_clip = config.audio_stride_clip
    audio_duration = config.audio_duration
    audio_num = config.audio_num
    total_frames = config.total_frames
    
    (audio, _) = read_audio(
            audio_path=audio_path, 
            target_fs=sample_rate)
    feature_extractor = LogMelExtractor(
        sample_rate=sample_rate, 
        window_size=window_size, 
        hop_size=hop_size, 
        mel_bins=mel_bins, 
        fmin=fmin, 
        fmax=fmax)
    audio = pad_truncate_sequence(audio, total_samples)
    fea_list = np.zeros((1, audio_num, frames_num_clip, mel_bins))
    feature = feature_extractor.transform(audio)
    feature = feature[0 : total_frames]
    for i in range(audio_num):
        feature_clip = feature[i*frames_per_second*audio_stride_clip: (i+audio_duration_clip)*frames_per_second*audio_stride_clip]
        fea_list[0, i ,: ,:]= feature_clip
    fea_list = move_data_to_gpu(fea_list, cuda)
    
    pred = np.zeros((audio_num), dtype=int)
    for i in range(audio_num): 
        output = model(fea_list[:, i, :, :])
        output = np.argmax(output.data.cpu().numpy(), axis=-1)
        pred[i] = output
    start = -1
    end = -1
#     print(pred)
    for i in range(len(pred)):
        # first second
        if pred[i] == 0 and start == -1:
            start = i
            end = i+3
        if pred[i] == 0 and start != -1:
            end = i+3
    if start != -1:
        return True, start, end
    else :
        return False, -1, -1
            
            
if __name__ == '__main__':
    logs_dir = '.logs/'
    create_logging(logs_dir, 'w')
    test_bgn_time = time.time()
    checkpoint_path = '/home/cdd/code/cry_sound_detection/FPNet/workspace/checkpoints/main/logmel_50frames_64melbins.h5/holdout_fold=1/Cnns/2000_iterations.pth'
    model_type = 'Cnns'
    test_fold = '/home/cdd/code/cry_sound_detection/FPNet/util/test_babycry'
#     test_wav = 'mixture_devtest_babycry_000_c02f92b79f2bbefa98d008f3c2d9b704.wav'
#     test_wav = 'mixture_devtest_babycry_004_19c4bcb10576524449c83b01fae0b731.wav'  
    for root, dirs, files in os.walk(test_fold):    
        file = files
    logging.info('------------------------------------')
    for f in file:
        test_wav = f
        logging.info('Test: {}'.format(test_wav))
        res, start, end = test_one(checkpoint_path=checkpoint_path, model_type=model_type, cuda=True, test_fold=test_fold, test_wav=test_wav, test_segment= False)
        if res == True:
            logging.info('Test result: Babycry! ({}~{}s)'.format(start, end))
        else :
            logging.info('Test result: None. ') 
    test_fin_time = time.time()
    test_time = test_fin_time - test_bgn_time
    logging.info('Test time: {:.3f} s'.format(test_time))