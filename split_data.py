# -*- coding: utf-8 -*-

from pathlib import Path
from random import random

source_dir = '/data/fashion/data/keypoint/warm_up_train_20180222'
output_dir = '/data/fashion/data/keypoint/fai_kp_coco'
image_set = 'train'
data_anno_path = Path(source_dir, 'Annotations', image_set + '.csv')
output_anno_path = Path(output_dir, 'annotations', image_set + '.json')
assert data_anno_path.exists(), "data_anno_path not exists"
if not output_anno_path.exists():
    output_anno_path.mkdir()
    print('create dir %s' % output_anno_path.as_posix())


classes = ['__background__', 'blouse', 'skirt', 'outwear', 'dress', 'trousers']

keypoints_name = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right', 'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right', 'crotch', 'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']
anno_kp = np.array([anno_dict.get(key).split('_') for key in keypoints_name]).astype(np.int16)
anno_kp = anno_kp[np.where(anno_kp[:, 2]>=0)].astype(np.uint16)
xmax, ymax, _ = anno_kp.max(axis=0)
xmin, ymin, _ = anno_kp.min(axis=0)


train_anno_op = open(data_anno_path.as_posix(), 'r')
csv_reader = csv.DictReader(train_anno_op)
data_raw = [row for row in csv_reader]


data_coco={}
gt_roidb = [load_fashion_kp_annotations(row) for row in csv_reader]




img_path = Path(img_dir)
output_path = Path(output_dir)
train_output_path = output_path / 'train.txt'
val_output_path = output_path / 'val.txt'

train_file=open(train_output_path, 'w+')
val_file=open(val_output_path, 'w+')
for img in img_path.glob('*.jpg'):
    if random() > 0.75:
        train_file.write("{}\n".format(img.stem))
    else:
        val_file.write("{}\n".format(img.stem))

val_file.close()
train_file.close()



mask = {
  "blouse":[0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14],
  "dress":[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18],
  "outwear":[0, 1, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14],
  "skirt":[15, 16, 17, 18, 19, 20, 21, 22, 23],
  "trousers":[17, 18, 19, 20, 21, 22, 23]
},

details = {
  "limb_seq": [[1,2], [1,3], [2,3], [1,4], [4,11], [6,10], [2,5], [5,13],
               [7,12], [4,6], [5,7], [6,14], [7,15], [6,8], [7,9], [16,17], [8,18],
               [9,19], [16,18], [17, 19], [16, 22], [17, 24], [16, 20], [17, 20],
               [20, 21], [20, 23]],
  "color_list": [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
                 [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
                 [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
                 [255, 0, 255], [255, 0, 170], [255, 0, 85]],
  "pose_id_dict": {
    "neckline_left":1, "neckline_right":2, "center_front":3, "shoulder_left":4,
    "shoulder_right":5, "armpit_left":6, "armpit_right":7, "waistline_left":8,
    "waistline_right":9, "cuff_left_in":10, "cuff_left_out":11, "cuff_right_in":12,
    "cuff_right_out":13, "top_hem_left":14, "top_hem_right":15, "waistband_left":16,
    "waistband_right":17, "hemline_left":18, "hemline_right":19, "crotch":20,
    "bottom_left_in":21, "bottom_left_out":22, "bottom_right_in":23, "bottom_right_out":24
  },
  "pose_name_seq": ["neckline_left", "neckline_right", "center_front", "shoulder_left",
                    "shoulder_right", "armpit_left", "armpit_right", "waistline_left",
                    "waistline_right", "cuff_left_in", "cuff_left_out", "cuff_right_in",
                    "cuff_right_out", "top_hem_left", "top_hem_right", "waistband_left",
                    "waistband_right", "hemline_left", "hemline_right", "crotch",
                    "bottom_left_in", "bottom_left_out", "bottom_right_in", "bottom_right_out"]
}


anno_keys = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right', 'armpit_left', 'armpit_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right']

[[1, 2], [1, 3], [2, 3], [1, 4], [2, 5], [4, 6], [6, 7], [7, 10], [10, 11], [10, 12], [12, 13], [13, 11], [11, 9], [9, 8] [8, 5], [5, 2]]

[(1, 'neckline_left'), (2, 'neckline_right'), (3, 'center_front'), (4, 'shoulder_left'), (5, 'shoulder_right'), (6, 'armpit_left'), (7, 'armpit_right'), (8, 'cuff_left_in'), (9, 'cuff_left_out'), (10, 'cuff_right_in'), (11, 'cuff_right_out'), (12, 'top_hem_left'), (13, 'top_hem_right')]

sorted_blouse_keys = ['neckline_left', 'shoulder_left', 'cuff_left_out', 'cuff_left_in', 'armpit_left', 'top_hem_left', 'top_hem_right', 'armpit_right', 'cuff_right_in', 'cuff_right_out', 'shoulder_right', 'neckline_right', 'center_front']
anno_kp = np.array([anno_dict.get(key).split('_') for key in sorted_blouse_keys]).astype(np.int16)
anno_kp = anno_kp[np.where(anno_kp[:, 2]>=0)].astype(np.uint16)


# for segmentation
sorted_blouse_keys = ['neckline_left', 'shoulder_left', 'cuff_left_out', 'cuff_left_in', 'armpit_left', 'top_hem_left', 'top_hem_right', 'armpit_right', 'cuff_right_in', 'cuff_right_out', 'shoulder_right', 'neckline_right', 'center_front']
anno_kp = np.array([anno_dict.get(key).split('_') for key in sorted_blouse_keys]).astype(np.int16)
visable_mask_anno_kp = anno_kp[np.where(anno_kp[:, 2]>=0)].astype(np.float)

print(len(np.where(anno_kp[:, 2] >= 0)[0]))
print(len(np.where(anno_kp[:, 2] > 0)[0]))
print(anno_kp[:, :2])


[ values[0] for values in visable_mask_anno_kp]
mask_op = visable_mask_anno_kp[:, :2].flatten()
print(data_coco['annotations'][0])
data_coco['annotations'][0]['segmentation'] = [mask_op.tolist()]
print(data_coco)



## Dress
| 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- |
| neckline_left | neckline_right | shoulder_left | shoulder_right | center_front |
| 6 | 7 | 8 | 9 | 10  |
| --- | --- | --- | --- | --- | --- |
| armpit_left | armpit_right |waistline_left | waistline_right | cuff_left_in|+
| 11| 12 | 13 | 14 | 15 |
| --- | --- | --- | --- |--- |
| cuff_left_out | cuff_right_in | cuff_right_out |hemline_left | hemline_right |

## Skirt
| 1 | 2 | 3 | 4 |
| --- | --- | --- | --- |
| waistband_left | waistband_right | hemline_left | hemline_right |

## Trousers
| 1 | 2 | 3 |
| --- | --- | --- |
| waistband_left | waistband_right | crotch |+
|4| 5 | 6 | 7 |
|--- | --- | --- |--- |
| bottom_left_in | bottom_left_out | bottom_right_in | bottom_right_out |

## Outwear
| 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- |
| neckline_left | neckline_right | shoulder_left | shoulder_right | armpit_left |
| 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- |
| armpit_right | waistline_left | waistline_right | cuff_left_in | cuff_left_out|
| 11 | 12 | 13 | 14 |+
| --- | --- | --- | --- |+
| cuff_right_in | cuff_right_out | top_hem_left |top_hem_right  |

## Blouse
|1|2|3|4|5|
|---|---|---|---|---|
|neckline_left|neckline_right|shoulder_left|shoulder_right|center_front|
|6|7|8|9|
|---|---|---|---|
|armpit_left|armpit_right|top_hem_left|top_hem_right|
|10|11|12|13|
|---|---|---|---|
|cuff_left_in|cuff_left_out|cuff_right_in|cuff_right_out|


kp=[3, 1, 4, 11, 10, 6, 8, 18, 19, 9, 7, 12, 13, 5, 2]
[[kp[i], kp[i+1]] for i in range(len(kp)-1)]




#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-

import os
import math
import json
import numpy as np
import csv
import argparse
import cv2
from pathlib import Path
import string
from random import choice
import secrets

IMAGE_DIR = 'images'
ANNO_DIR = 'annotations'
RESULTS_DIR = 'results'


# KEYPONT_NAMES = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right', 'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right', 'crotch', 'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']

FAI_CAT_LIST = ['blouse', 'dress', 'outwear', 'skirt', 'trousers']
KEYPONT_NAME_DICT= {
  1:"neckline_left", 2:"neckline_right", 3:"center_front", 4:"shoulder_left",
  5:"shoulder_right", 6:"armpit_left", 7:"armpit_right", 8:"waistline_left",
  9:"waistline_right", 10:"cuff_left_in", 11:"cuff_left_out", 12:"cuff_right_in",
  13:"cuff_right_out", 14:"top_hem_left", 15:"top_hem_right", 16:"waistband_left",
  17:"waistband_right", 18:"hemline_left", 19:"hemline_right", 20:"crotch",
  21:"bottom_left_in", 22:"bottom_left_out", 23:"bottom_right_in", 24:"bottom_right_out"
}

blouse_kp = [ KEYPONT_NAME_DICT[kp_index] for kp_index in [3, 1, 4, 11, 10, 6, 14, 15, 7, 12, 13, 5, 2] ]
blouse_skeleton_ep = [ [KEYPONT_NAME_DICT[kp_ep[0]], KEYPONT_NAME_DICT[kp_ep[1]]] for kp_ep in [[3, 1], [1, 4], [4, 11], [11, 10], [10, 6], [6, 14], [14, 15], [15, 7], [7, 12], [12, 13], [13, 5], [5, 2], [2, 3], [6, 7]] ]
blouse_skeleton = [ [blouse_kp.index(ep[0]), blouse_kp.index(ep[1])] for ep in blouse_skeleton_ep ]

dress_kp = [ KEYPONT_NAME_DICT[kp_index] for kp_index in [3, 1, 4, 11, 10, 6, 8, 18, 19, 9, 7, 12, 13, 5, 2] ]
dress_skeleton_ep = [ [KEYPONT_NAME_DICT[kp_ep[0]], KEYPONT_NAME_DICT[kp_ep[1]]] for kp_ep in [[3, 1], [1, 4], [4, 11], [11, 10], [10, 6], [6, 8], [8, 18], [18, 19], [19, 9], [9, 7], [7, 12], [12, 13], [13, 5], [5, 2], [2, 3], [8, 9]] ]
dress_skeleton = [ [dress_kp.index(ep[0]), dress_kp.index(ep[1])] for ep in dress_skeleton_ep ]

skirt_kp = [ KEYPONT_NAME_DICT[kp_index] for kp_index in [16, 17, 18, 19]]
skirt_skeleton_ep = [ [KEYPONT_NAME_DICT[kp_ep[0]], KEYPONT_NAME_DICT[kp_ep[1]]] for kp_ep in [[16, 17], [17, 18], [18, 19], [19, 16]] ]
skirt_skeleton = [ [skirt_kp.index(ep[0]), skirt_kp.index(ep[1])] for ep in skirt_skeleton_ep ]

trousers_kp = [ KEYPONT_NAME_DICT[kp_index] for kp_index in [16, 17, 23, 24, 20, 21, 22]]
trousers_skeleton_ep = [ [KEYPONT_NAME_DICT[kp_ep[0]], KEYPONT_NAME_DICT[kp_ep[1]]] for kp_ep in [[16, 17], [17, 23], [23, 24], [24, 20], [20, 21], [21, 22], [22, 16], [20, 16], [20, 17]] ]
trousers_skeleton = [ [trousers_kp.index(ep[0]), trousers_kp.index(ep[1])] for ep in trousers_skeleton_ep ]

outwear_kp = [ KEYPONT_NAME_DICT[kp_index] for kp_index in [16, 17, 23, 24, 20, 21, 22] ]
outwear_skeleton_ep = [ [KEYPONT_NAME_DICT[kp_ep[0]], KEYPONT_NAME_DICT[kp_ep[1]]] for kp_ep in [[1, 4], [4, 11], [11, 10], [10, 6], [6, 8], [8, 18], [18, 19], [19, 9], [9, 7], [7, 12], [12, 13], [13, 5], [5, 2], [2, 1], [8, 9]] ]
outwear_skeleton = [ [outwear_kp.index(ep[0]), outwear_kp.index(ep[1])] for ep in outwear_skeleton_ep ]

class FaiPoseGenerator(object):

    def __init__(self, source_dir, image_set, dest_dir, generate_bbox=True, generate_segm=True, generate_kp=True):

        self.source_dir = source_dir
        self.image_set = image_set
        self.dest_dir = dest_dir
        self.generate_bbox = generate_bbox
        self.generate_segm = generate_segm
        self.generate_kp = generate_kp

        self.images = []
        self.categories = []
        self.annotations = []
        self.data_coco = {}

    def create_categories(self):
        for index, cat in enumerate(FAI_CAT_LIST):
            category = {}
            category['supercategory'] = cat
            category['id'] = index+1
            category['name']= cat

            category['keypoints'] = vars()[cat+'_kp']
            category['skeleton'] = vars()[cat+'_skeleton']

            self.categories.append(category)

    def get_category(self, name):
        for cat in self.categories:
            if cat['name'] == name:
                return cat

        return None

    def generate_label(self):
        data_anno_path = Path(self.source_dir, 'annotations', self.image_set + '.csv')
        if not data_anno_path.exists():
            print("path %s not exists" % data_anno_path)
            exit(-1)

        for anno_dict in csv.DictReader(data_anno_path.open('r')):
            cat_name = anno_dict['image_category']
            cat_dict = self.get_category(cat_name)
            image_path = anno_dict['image_id']
            ab_image_path = Path(self.source_dir, image_path.lower())
            if not ab_image_path.exists():
                print("Path does not exist: {}".format(ab_image_path))
                continue
            else:
                print('processing %s %s' % (self.image_set, image_path))

            image={}
            image_raw = cv2.imread(ab_image_path.as_posix())
            image['height'], image['width'], _ = image_raw.shape
            image_name = image_path.split('/')[-1]
            image['file_name'] = image_name
            image['id'] = secrets.randbits(64)

            image['category'] = cat_name
            self.images.append(image)

            category_id = FAI_CAT_LIST.index(cat_name) + 1
            annotation={'segmentation': [], 'bbox': [], 'keypoints': [],
                'iscrowd': 0, 'image_id': image['id'], 'category_id': category_id}
            annotation['id'] = secrets.randbits(64)

            anno_kp = np.array([anno_dict.get(key).split('_') for key in cat_dict['keypoints']]).astype(np.int16)

            if self.generate_bbox:
                bbox_anno_kp = anno_kp[np.where(anno_kp[:, 2]>=0)].astype(np.uint16)
                xmax, ymax, _ = bbox_anno_kp.max(axis=0)
                xmin, ymin, _ = bbox_anno_kp.min(axis=0)
                bbox = np.array([xmin, ymin, xmax-xmin, ymax-ymin]).tolist()
                annotation['bbox'] = bbox
                annotation['area'] = int(xmax-xmin) * int(ymax-ymin)

            if self.generate_segm:
                mask_op = anno_kp[:, :2].flatten()
                annotation['segmentation'] = [mask_op.tolist()]

            if self.generate_kp:
                anno_kps = [[aa[0], aa[1], aa[2]+1] for aa in anno_kp.tolist()]
                annotation['"num_keypoints"'] = len(anno_kps)
                annotation['keypoints'] = [ p for kp in anno_kps for p in kp]

            self.annotations.append(annotation)

    def data2coco(self):
        self.data_coco['images'] = self.images
        self.data_coco['categories'] = self.categories
        self.data_coco['annotations'] = self.annotations

        return self.data_coco

    def save(self):
        output_anno_path = Path(self.dest_dir, 'annotations', self.image_set + '.json')
        json.dump(self.data2coco(), output_anno_path.open(mode='w+'), indent=4)
        print("save results to %s" % output_anno_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default=None, type=str,
                        dest='source_dir', help='The directory to save the ground truth.')
    parser.add_argument('--image_set', default=None, type=str,
                        dest='image_set', help='train|val|test')
    parser.add_argument('--dest_dir', default=None, type=str,
                        dest='dest_dir', help='The data dir corresponding to coco anno file.')

    args = parser.parse_args()

    fai_pose_generator = FaiPoseGenerator(args.source_dir, args.image_set, args.dest_dir)
    fai_pose_generator.create_categories()
    fai_pose_generator.generate_label()
    fai_pose_generator.save()



blouse_kp = [ KEYPONT_NAME_DICT[kp_index] for kp_index in [3, 1, 4, 11, 10, 6, 14, 15, 7, 12, 13, 5, 2] ]
blouse_skeleton_ep = [ [KEYPONT_NAME_DICT[kp_ep[0]], KEYPONT_NAME_DICT[kp_ep[1]]] for kp_ep in [[3, 1], [1, 4], [4, 11], [11, 10], [10, 6], [6, 14], [14, 15], [15, 7], [7, 12], [12, 13], [13, 5], [5, 2], [2, 3], [6, 7]] ]
blouse_skeleton = [ [blouse_kp.index(ep[0])+1, blouse_kp.index(ep[1])+1] for ep in blouse_skeleton_ep ]

dress_kp = [ KEYPONT_NAME_DICT[kp_index] for kp_index in [3, 1, 4, 11, 10, 6, 8, 18, 19, 9, 7, 12, 13, 5, 2] ]
dress_skeleton_ep = [ [KEYPONT_NAME_DICT[kp_ep[0]], KEYPONT_NAME_DICT[kp_ep[1]]] for kp_ep in [[3, 1], [1, 4], [4, 11], [11, 10], [10, 6], [6, 8], [8, 18], [18, 19], [19, 9], [9, 7], [7, 12], [12, 13], [13, 5], [5, 2], [2, 3], [8, 9]] ]
dress_skeleton = [ [dress_kp.index(ep[0])+1, dress_kp.index(ep[1])+1] for ep in dress_skeleton_ep ]

skirt_kp = [ KEYPONT_NAME_DICT[kp_index] for kp_index in [16, 17, 18, 19]]
skirt_skeleton_ep = [ [KEYPONT_NAME_DICT[kp_ep[0]], KEYPONT_NAME_DICT[kp_ep[1]]] for kp_ep in [[16, 17], [17, 19], [19, 18], [18, 16]] ]
skirt_skeleton = [ [skirt_kp.index(ep[0])+1, skirt_kp.index(ep[1])+1] for ep in skirt_skeleton_ep ]

trousers_kp = [ KEYPONT_NAME_DICT[kp_index] for kp_index in [16, 17, 23, 24, 20, 21, 22]]
trousers_skeleton_ep = [ [KEYPONT_NAME_DICT[kp_ep[0]], KEYPONT_NAME_DICT[kp_ep[1]]] for kp_ep in [[16, 17], [17, 23], [23, 24], [24, 20], [20, 21], [21, 22], [22, 16], [20, 16], [20, 17]] ]
trousers_skeleton = [ [trousers_kp.index(ep[0])+1, trousers_kp.index(ep[1])+1] for ep in trousers_skeleton_ep ]

outwear_kp = [ KEYPONT_NAME_DICT[kp_index] for kp_index in [1, 4, 11, 10, 6, 8, 18, 19, 9, 7, 12, 13, 5, 2] ]
outwear_skeleton_ep = [ [KEYPONT_NAME_DICT[kp_ep[0]], KEYPONT_NAME_DICT[kp_ep[1]]] for kp_ep in [[1, 4], [4, 11], [11, 10], [10, 6], [6, 8], [8, 18], [18, 19], [19, 9], [9, 7], [7, 12], [12, 13], [13, 5], [5, 2], [2, 1], [8, 9]] ]
outwear_skeleton = [ [outwear_kp.index(ep[0])+1, outwear_kp.index(ep[1])+1] for ep in outwear_skeleton_ep ]



### difficult dress

/mnt/data/images/dress/ec3cf1bc8f83449074054f2dc2ef1f6e.jpg
/mnt/data/images/outwear/72d1be6f2534f3c279f24d0d561b82f6.jpg
