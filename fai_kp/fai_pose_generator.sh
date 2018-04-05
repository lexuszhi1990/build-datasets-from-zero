#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Generate train & val data.

SOURCE_DIR='/mnt/data'
DEST_DIR='/mnt/data'
# IMAGE_SET='train'
IMAGE_SET='val'

python3.6 fai_pose_generator.py --source_dir $SOURCE_DIR \
                                --image_set $IMAGE_SET \
                                --dest_dir $DEST_DIR \
