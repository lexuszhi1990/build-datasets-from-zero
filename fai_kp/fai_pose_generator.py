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



FAI_CAT_LIST = ['blouse', 'dress', 'outwear', 'skirt', 'trousers']
KEYPONT_NAMES = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right', 'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right', 'crotch', 'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']

blouse_kp = ['center_front', 'neckline_left', 'shoulder_left', 'cuff_left_out', 'cuff_left_in', 'armpit_left', 'top_hem_left', 'top_hem_right', 'armpit_right', 'cuff_right_in', 'cuff_right_out', 'shoulder_right', 'neckline_right']
blouse_mask = [['center_front', 'neckline_left', 'shoulder_left', 'armpit_left', 'top_hem_left', 'top_hem_right', 'armpit_right', 'shoulder_right', 'neckline_right'], ['neckline_left', 'shoulder_left', 'cuff_left_out', 'cuff_left_in', 'armpit_left'], ['armpit_right', 'cuff_right_in', 'cuff_right_out', 'shoulder_right', 'neckline_right']]
blouse_skeleton_ep = [['center_front', 'neckline_left'], ['neckline_left', 'shoulder_left'], ['shoulder_left', 'cuff_left_out'], ['cuff_left_out', 'cuff_left_in'], ['cuff_left_in', 'armpit_left'], ['armpit_left', 'top_hem_left'], ['top_hem_left', 'top_hem_right'], ['top_hem_right', 'armpit_right'], ['armpit_right', 'cuff_right_in'], ['cuff_right_in', 'cuff_right_out'], ['cuff_right_out', 'shoulder_right'], ['shoulder_right', 'neckline_right'], ['neckline_right', 'center_front'], ['armpit_left', 'armpit_right'], ['armpit_left', 'shoulder_left'], ['armpit_left', 'neckline_left'], ['armpit_right', 'neckline_right'], ['armpit_right', 'shoulder_right']]
blouse_skeleton = [ [blouse_kp.index(ep[0])+1, blouse_kp.index(ep[1])+1] for ep in blouse_skeleton_ep ]

dress_kp = ['center_front', 'neckline_left', 'shoulder_left', 'cuff_left_out', 'cuff_left_in', 'armpit_left', 'waistline_left', 'hemline_left', 'hemline_right', 'waistline_right', 'armpit_right', 'cuff_right_in', 'cuff_right_out', 'shoulder_right', 'neckline_right']
dress_mask = [['center_front', 'neckline_left', 'shoulder_left', 'armpit_left', 'waistline_left', 'hemline_left', 'hemline_right', 'waistline_right', 'armpit_right', 'shoulder_right', 'neckline_right'], ['neckline_left', 'shoulder_left', 'cuff_left_out', 'cuff_left_in', 'armpit_left'], ['armpit_right', 'cuff_right_in', 'cuff_right_out', 'shoulder_right', 'neckline_right']]
dress_skeleton_ep = [['center_front', 'neckline_left'], ['center_front', 'neckline_right'], ['neckline_left', 'neckline_right'], ['neckline_left', 'shoulder_left'], ['neckline_left', 'armpit_left'], ['shoulder_left', 'armpit_left'], ['shoulder_left', 'cuff_left_out'], ['cuff_left_out', 'cuff_left_in'], ['cuff_left_in', 'armpit_left'], ['armpit_left', 'waistline_left'], ['armpit_left', 'hemline_left'], ['armpit_left', 'armpit_right'], ['waistline_left', 'hemline_left'], ['waistline_left', 'waistline_right'], ['hemline_left', 'hemline_right'], ['hemline_right', 'waistline_right'], ['hemline_right', 'armpit_right'], ['waistline_right', 'armpit_right'], ['armpit_right', 'shoulder_right'], ['armpit_right', 'neckline_right'], ['armpit_right', 'cuff_right_in'], ['cuff_right_in', 'cuff_right_out'], ['cuff_right_out', 'shoulder_right'], ['shoulder_right', 'neckline_right']]
dress_skeleton = [ [dress_kp.index(ep[0])+1, dress_kp.index(ep[1])+1] for ep in dress_skeleton_ep ]

skirt_kp = ['waistband_left', 'waistband_right', 'hemline_right', 'hemline_left']
skirt_mask = [['waistband_left', 'waistband_right', 'hemline_right', 'hemline_left']]
skirt_skeleton_ep = [['waistband_left', 'waistband_right'], ['waistband_right', 'hemline_right'], ['hemline_right', 'hemline_left'], ['hemline_left', 'waistband_left']]
skirt_skeleton = [ [skirt_kp.index(ep[0])+1, skirt_kp.index(ep[1])+1] for ep in skirt_skeleton_ep ]

trousers_kp = ['waistband_left', 'waistband_right', 'bottom_right_out', 'bottom_right_in', 'crotch', 'bottom_left_in', 'bottom_left_out']
trousers_mask = [['waistband_left', 'waistband_right', 'crotch'], ['waistband_right', 'bottom_right_out', 'bottom_right_in', 'crotch'], ['waistband_left', 'bottom_left_out', 'bottom_left_in', 'crotch']]
trousers_skeleton_ep = [['waistband_left', 'waistband_right'], ['waistband_right', 'bottom_right_out'], ['bottom_right_out', 'bottom_right_in'], ['bottom_right_in', 'crotch'], ['crotch', 'bottom_left_in'], ['bottom_left_in', 'bottom_left_out'], ['bottom_left_out', 'waistband_left'], ['crotch', 'waistband_left'], ['crotch', 'waistband_right']]
trousers_skeleton = [ [trousers_kp.index(ep[0])+1, trousers_kp.index(ep[1])+1] for ep in trousers_skeleton_ep ]

 # neckline_left  neckline_right  shoulder_left  shoulder_right  armpit_left
 # armpit_right  waistline_left  waistline_right  cuff_left_in  cuff_left_out
 # cuff_right_in  cuff_right_out  top_hem_left top_hem_right
outwear_kp = ['neckline_left', 'shoulder_left', 'cuff_left_out', 'cuff_left_in', 'armpit_left', 'waistline_left', 'top_hem_left', 'top_hem_right', 'waistline_right', 'armpit_right', 'cuff_right_in', 'cuff_right_out', 'shoulder_right', 'neckline_right']
outwear_mask = [['neckline_left', 'shoulder_left', 'armpit_left', 'waistline_left', 'top_hem_left', 'top_hem_right', 'waistline_right', 'armpit_right', 'shoulder_right', 'neckline_right'], ['neckline_left', 'shoulder_left', 'cuff_left_out', 'cuff_left_in', 'armpit_left'], ['armpit_right', 'cuff_right_in', 'cuff_right_out', 'shoulder_right', 'neckline_right']]
outwear_skeleton_ep = [['neckline_left', 'shoulder_left'], ['neckline_left', 'armpit_left'], ['neckline_left', 'neckline_right'], ['shoulder_left', 'cuff_left_out'], ['shoulder_left', 'armpit_left'], ['cuff_left_out', 'cuff_left_in'], ['cuff_left_in', 'armpit_left'], ['armpit_left', 'armpit_right'], ['armpit_left', 'waistline_left'], ['armpit_left', 'top_hem_left'], ['waistline_left', 'waistline_right'], ['waistline_left', 'top_hem_left'], ['top_hem_left', 'top_hem_right'], ['top_hem_right', 'waistline_right'], ['top_hem_right', 'armpit_right'], ['waistline_right', 'armpit_right'], ['armpit_right', 'cuff_right_in'], ['armpit_right', 'shoulder_right'], ['armpit_right', 'neckline_right'], ['cuff_right_in', 'cuff_right_out'], ['cuff_right_out', 'shoulder_right'], ['shoulder_right', 'neckline_right']]
outwear_skeleton = [ [outwear_kp.index(ep[0])+1, outwear_kp.index(ep[1])+1] for ep in outwear_skeleton_ep ]

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

            category['keypoints'] = globals()[cat+'_kp']
            category['mask_list'] = globals()[cat+'_mask']
            category['skeleton'] = globals()[cat+'_skeleton']

            self.categories.append(category)

    def get_category(self, name):
        for cat in self.categories:
            if cat['name'] == name:
                return cat
        return None

    def _build_annotation(self, anno_dict):
        image_name = anno_dict['image_id'].split('/')[-1]
        cat_name = anno_dict['image_category']
        cat_dict = self.get_category(cat_name)
        category_id = FAI_CAT_LIST.index(cat_name) + 1
        annotation={'segmentation': [], 'bbox': [], 'keypoints': [],
                    'iscrowd': 0, 'image_id': anno_dict['new_img_id'],
                    'category_id': category_id}
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
            for mask_kp in cat_dict['mask_list']:
                segm_mask = np.array([anno_dict.get(key).split('_') for key in mask_kp]).astype(np.int16)
                segm_anno_kp = segm_mask[np.where(segm_mask[:, 2]>=0)].astype(np.uint16)
                segm_anno_op = segm_anno_kp[:, :2].flatten().tolist()
                if len(segm_anno_op) >= 3:
                    annotation['segmentation'].append(segm_anno_op)

        if self.generate_kp:
            anno_kps = [[aa[0], aa[1], aa[2]+1] for aa in anno_kp.tolist()]
            annotation['"num_keypoints"'] = len(anno_kps)
            annotation['keypoints'] = [ p for kp in anno_kps for p in kp]

        return annotation

    def generate_label(self):
        data_anno_path = Path(self.source_dir, 'annotations', self.image_set + '.csv')
        if not data_anno_path.exists():
            print("path %s not exists" % data_anno_path)
            exit(-1)

        for anno_dict in csv.DictReader(data_anno_path.open('r')):
            image_name = anno_dict['image_id']
            ab_image_path = Path(self.source_dir, 'images', self.image_set, image_name)
            if not ab_image_path.exists():
                print("Path does not exist: {}".format(ab_image_path))
                continue
            else:
                print('processing %s %s' % (self.image_set, ab_image_path))

            image={}
            image_raw = cv2.imread(ab_image_path.as_posix())
            image['height'], image['width'], _ = image_raw.shape
            image['file_name'] = image_name
            image['id'] = secrets.randbits(64)
            image['category'] = anno_dict['image_category']
            self.images.append(image)

            if self.image_set.find('test') < 0:
                anno_dict['new_img_id'] = image['id']
                annotation = self._build_annotation(anno_dict)
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
