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
