#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Generate train & val data.

SOURCE_DIR='/mnt/data/fai_kp/ROUND2/train_0427'
DEST_DIR='/mnt/data/fai_kp/ROUND2/train_0427'
# IMAGE_SET='train_0427'
IMAGE_SET='val_0427'
# IMAGE_SET='test'

# if [ x$1 == x ] ; then
#     echo "no image set input, use 'train' as default image set"
#     IMAGE_SET='train'
# else
#     IMAGE_SET=$1
# fi

echo "source_dir: $SOURCE_DIR, dest_dir: $DEST_DIR, image_set: $IMAGE_SET"

python3.6 fai_pose_generator.py --source_dir $SOURCE_DIR \
                                --image_set $IMAGE_SET \
                                --dest_dir $DEST_DIR
