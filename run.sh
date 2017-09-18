#!/bin/bash

python train.py --train_dir=/home/procastinator/nturgbd_cvprcnn --dataset_dir=/home/procastinator/nturgb+d_images_cvpr \
    --splits_dir=/home/procastinator/NTU_data --split_num 1 --checkpoint_file=/home/procastinator/pretrainedCheckpoints/vgg_19.ckpt
