#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Generate train & val data.

# SOURCE_DIR='/mnt/data'
# DEST_DIR='/mnt/data'
# IMAGE_SET='train'
# IMAGE_SET='val'
# IMAGE_SET='test'

# SOURCE_DIR='/mnt/data/fai_attr/raw_data/test_v1'
# DEST_DIR='/mnt/data/fai_attr/raw_data/test_v1'
# IMAGE_SET='question'

# SOURCE_DIR='/mnt/data/fai_attr/raw_data/val_v1'
# DEST_DIR='/mnt/data/fai_attr/raw_data/val_v1'
# IMAGE_SET='question'

# SOURCE_DIR='/mnt/data/fai_attr/datasets_david/round2_train'
# DEST_DIR='/mnt/data/fai_attr/datasets_david/round2_train'
# IMAGE_SET='label'

# SOURCE_DIR='/mnt/data/fai_attr/datasets_david/round2_rank'
# DEST_DIR='/mnt/data/fai_attr/datasets_david/round2_rank'
# IMAGE_SET='question'

# SOURCE_DIR='/mnt/data/fai_attr/datasets_david/round1_round2_train'
# DEST_DIR='/mnt/data/fai_attr/datasets_david/round1_round2_train'
# IMAGE_SET='label'

SOURCE_DIR='/mnt/data/fai_attr/datasets_david/round2_train_dir/TRAIN_V1'
DEST_DIR='/mnt/data/fai_attr/datasets_david/round2_train_dir/TRAIN_V1'
IMAGE_SET='train'

# if [ x$1 == x ] ; then
#     echo "no image set input, use 'train' as default image set"
#     IMAGE_SET='test_attr'
# else
#     IMAGE_SET=$1
# fi

echo "source_dir: $SOURCE_DIR, dest_dir: $DEST_DIR, image_set: $IMAGE_SET"

python3.6 fai_attr_generator.py --source_dir $SOURCE_DIR \
                                --image_set $IMAGE_SET \
                                --dest_dir $DEST_DIR
