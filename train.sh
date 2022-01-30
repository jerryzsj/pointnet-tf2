#!/bin/bash

PYTRAIN=pointnet.py

EXPNAME=0130-1-mech12-B100-10-Epo30-rotate_full-jitter_no-2021-colab
TRAINNAME=3cam_origin_1000_norm
TESTNAME=3cam_origin_1000_norm
DATATYPE=mech12
python3 $PYTRAIN --reiteration=10 --exp_start_idx=0 --rotate_pcl=True --rotate_type=full --jitter_pcl=False --max_epoch=30 --num_class=12 --num_point=1000 --batch_size=100 --exp_name=$EXPNAME --train_name=$TRAINNAME --train_type=$DATATYPE --test_name=$TESTNAME --test_type=$DATATYPE

