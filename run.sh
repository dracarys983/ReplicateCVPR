#!/bin/bash

python train.py --train_dir=/home/procastinator/nturgbd_cvprcnn --dataset_dir=/home/procastinator/ActionRecognition/Preprocessed/nturgb+d_images_cvpr \
    --splits_dir=/home/procastinator/ActionRecognition/Datasets/NTU_data --split_num 1 --checkpoint_file=/home/procastinator/ActionRecognition/pretrainedCheckpoints/vgg_19.ckpt
