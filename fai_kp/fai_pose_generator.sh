#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Generate train & val data.

SOURCE_DIR='/mnt/data/warm_up_train_20180222/'
DEST_DIR='/mnt/data/fai_kp_coco/'
IMAGE_SET='train'

python3.6 fai_pose_generator.py --source_dir $SOURCE_DIR \
                                --image_set $IMAGE_SET \
                                --dest_dir $DEST_DIR \
