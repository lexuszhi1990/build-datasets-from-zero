# -*- coding: utf-8 -*-

import csv
from pathlib import Path
from random import random

# warmup_source_dir = '/mnt/data/warm_up_train_20180222'
source_dir = '/mnt/data/train'
output_dir = '/mnt/data/fai_kp_coco'

data_anno_path = Path(source_dir, 'Annotations', 'train.csv')
if not data_anno_path.exists():
    print('source annotation file %s not exist...' % data_anno_path.as_posix())
    exit(1)

output_path = Path(output_dir)
train_output_path = output_path.joinpath('annotations', 'train.csv')
val_output_path = output_path.joinpath('annotations', 'val.csv')

source_reader = csv.DictReader(data_anno_path.open('r'))
fieldnames = source_reader.fieldnames

train_writer = csv.DictWriter(train_output_path.open('w+'), fieldnames=fieldnames)
val_writer = csv.DictWriter(val_output_path.open('w+'), fieldnames=fieldnames)
train_writer.writeheader()
val_writer.writeheader()

for anno_dict in source_reader:
    if random() > 0.75:
        val_writer.writerow(anno_dict)
        print("val: %s" % anno_dict['image_id'])
    else:
        train_writer.writerow(anno_dict)
        print("train: %s" % anno_dict['image_id'])
    print("line : %d" % source_reader.line_num)

print('done...')
