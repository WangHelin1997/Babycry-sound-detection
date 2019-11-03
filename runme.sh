#!/bin/bash
# You need to modify this path to your downloaded dataset directory
DATASET_DIR='/home/cdd/code/MTF-CRNN/applications/data/TUT-rare-sound-events-2017-development/generated_data'

# You need to modify this path to your workspace to store features and models
WORKSPACE='/home/cdd/code/cry_sound_detection/FPNet/workspace'

# Hyper-parameters
GPU_ID=2
MODEL_TYPE='Cnns'
# MODEL_TYPE='Crnn'
BATCH_SIZE=64

############ Train and validate on dataset ############
# Calculate feature
# python util/feature.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Tarin
CUDA_VISIBLE_DEVICES=$GPU_ID python util/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --batch_size=$BATCH_SIZE --cuda
