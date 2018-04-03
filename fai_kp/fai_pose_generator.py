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

IMAGE_DIR = 'images'
ANNO_DIR = 'annotations'
RESULTS_DIR = 'results'

Cat_List = ['blouse', 'dress', 'outwear', 'skirt', 'trousers']
KEYPONT_NAMES = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right', 'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right', 'crotch', 'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']


def generate_secret_key(length=8):
    """Generate secret key"""
    return "".join([choice(string.hexdigits).lower() for i in range(length)])


class FaiPoseGenerator(object):

    def __init__(self, source_dir, image_set, dest_dir, bbox=True, mask=False, keypoint=False):

        self.source_dir = source_dir
        self.image_set = image_set
        self.dest_dir = dest_dir
        self.bbox = bbox
        self.mask = mask
        self.keypoint = keypoint

        self.images = []
        self.categories = []
        self.annotations = []
        self.data_coco = {}



    def create_categories(self):
        for index, cat in enumerate(Cat_List):
            category = {}
            category['supercategory'] = cat
            category['id'] = index+1
            category['name']= cat

            category['keypoints'] = ["neckline_left", "neckline_right", "center_front", "shoulder_left",
                "shoulder_right", "armpit_left", "armpit_right", "waistline_left",
                "waistline_right", "cuff_left_in", "cuff_left_out", "cuff_right_in",
                "cuff_right_out", "top_hem_left", "top_hem_right", "waistband_left",
                "waistband_right", "hemline_left", "hemline_right", "crotch",
                "bottom_left_in", "bottom_left_out", "bottom_right_in", "bottom_right_out"]
            category['skeleton'] = [[1,2], [1,3], [2,3], [1,4], [4,11], [6,7], [6,10], [2,5], [5,13],
                [7,12], [4,6], [5,7], [6,14], [14,15], [7,15], [6,8], [7,9], [16,17], [8,18],
                [9,19], [16,18], [17, 19], [16, 22], [17, 24], [16, 20], [17, 20],
                [20, 21], [20, 23]]

            self.categories.append(category)


    def generate_label(self):
        data_anno_path = Path(self.source_dir, 'Annotations', self.image_set + '.csv')
        assert data_anno_path.exists(), "data_anno_path not exists"

        for anno_dict in csv.DictReader(data_anno_path.open('r')):
            image={}
            image_path = anno_dict['image_id']
            ab_image_path = Path(self.source_dir, image_path)
            if not ab_image_path.exists():
                print("Path does not exist: {}".format(ab_image_path))
                continue

            image_raw = cv2.imread(ab_image_path.as_posix())
            image['height'], image['width'], _ = image_raw.shape
            image_name = image_path.split('/')[-1]
            image['file_name'] = image_name
            image['id'] = image_name.split('.')[0]
            self.images.append(image)

            category = anno_dict['image_category']
            category_id = Cat_List.index(category) + 1
            annotation={'segmentation': [], 'bbox': [], 'keypoints': [],
                'iscrowd': 0, 'image_id': image['id'], 'category_id': category_id}
            annotation['id'] = generate_secret_key(16)

            anno_kp = np.array([anno_dict.get(key).split('_') for key in KEYPONT_NAMES]).astype(np.int16)
            bb_anno_kp = anno_kp[np.where(anno_kp[:, 2]>=0)].astype(np.uint16)
            xmax, ymax, _ = bb_anno_kp.max(axis=0)
            xmin, ymin, _ = bb_anno_kp.min(axis=0)
            bbox = np.array([xmin, ymin, xmax-xmin, ymax-ymin]).tolist()
            annotation['bbox'] = bbox

            anno_kps = [[aa[0], aa[1], aa[2]+1] for aa in anno_kp.tolist()]
            annotation['keypoints'] = [ p for kp in anno_kps for p in kp]
            self.annotations.append(annotation)

            print('processing %s end' % image_path)

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
