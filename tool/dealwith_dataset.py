# -*- coding:utf8 -*-
import argparse
import os
import shutil
import cv2
import csv
import base64
import requests
import json
from tools.kitti_vis.kitti_utils import *
from tools.pykitti_eval.non_max_suppression.nms_gpu import rotate_iou_gpu_eval

TYPE_ID_CONVERSION = {
    'waymo': {
        'Car': 0,
        'Cyclist': 1,
        'Pedestrian': 2,
        'Truck': 3,
        'Tricycle': 4,
        'Bus': 5,
        'Motobike': 6,
    },
    'jdx': {
        'Car': 0,
        'Cyclist': 1,
        'Pedestrian': 2,
        'Truck': 3,
        'Tricycle': 4,
        'Bus': 5,
        'Cyclist_stopped': 6,
        'Others': 7,
    },
    'kitti': {
        'Car': 0,
        'Cyclist': 1,
        'Pedestrian': 2,
        'Van': 1,
        'Truck': 3,
        'Tram': 4,
        'Person_sitting': 5,
        'Misc': 6,
    },
    'lyft': {
        'car': 0,
        'cyclist': 1,
        'pedestrian': 2,
        'truck': 3,
        'bus': 4,
        'other_vehicle': 6,
        'bicycle': 7,
    },
}
CAMERA_TO_ID = {
    'front': 0,
    'front_left': 1,
    'front_right': 2,
    'side_left': 3,
    'side_right': 4,
}
range_list = {"kitti": [(-39.68, 39.68), (-69.12, 69.12), (-2., -2.), 0.05],
              "lyft": [(-100, 100), (-60, 60), (-2., -2.), 0.10],
              "jdx": [(-40, 40), (-0, 60), (-2., -2.), 0.10],
              "waymo": [(-60, 60), (-0, 80), (-2., -2.), 0.1]}

VEHICLE_SMALL_SCALE = 2.9
VEHICLE_LARGE_SCALE = 7
IOU_THRESHOLD = 0.9
SELF_IOU_THRESHOLD = 0.8
VEHICLE_RIOU_THRESHOLD = 0.05

# For classification
URL = "http://127.0.0.1:8001"
VEHICLE_CLASS = {0: 'Bus', 1: 'Truck'}


# Change class type of label file, according to the classification result
# the format of file name: imagename_index.txt
def change_class_type(classification_path, src_label_path, dst_label_path):
    class_type = ['Bus', 'Car', 'Misc', 'Tricycle', 'Pedestrian', 'Truck',
                  'Cyclist', 'Others', 'Cyclist_stopped']
    classification_path = classification_path + 'crop'
    src_label_path = os.path.join(src_label_path, 'training', 'label_2')
    dst_label_path = os.path.join(dst_label_path, 'label_2')
    if not os.path.exists(dst_label_path):
        os.makedirs(dst_label_path)
    for cls_type in class_type:
        sub_path = os.path.join(classification_path, cls_type)
        if not os.path.exists(sub_path):
            continue
        for line in os.listdir(sub_path):
            line = line.strip().split('.')[0]
            i = line.index('_', -4)
            image_name = line[:i]
            idx = line[i + 1:]

            src_label = os.path.join(src_label_path, image_name + '.txt')
            if not os.path.exists(src_label):
                continue
            dst_label = os.path.join(dst_label_path, image_name + '.txt')
            if os.path.exists(dst_label):
                src_label = dst_label

            annotations = KittiFormatCleaner.load_annotations(src_label)
            annotations[int(idx)]['class'] = cls_type
            KittiFormatCleaner.save_annotations(image_name, annotations,
                                                dst_label_path)


def filter_label(src_label_path, dst_label_path):
    class_type = ['Misc']
    src_label_path = os.path.join(src_label_path, 'label_2')
    dst_label_path = os.path.join(dst_label_path, 'label_2_filter')
    if not os.path.exists(dst_label_path):
        os.makedirs(dst_label_path)
    for idx, file_name in enumerate(os.listdir(src_label_path)):
        file_name = file_name.strip().split('.')[0]
        src_label = os.path.join(src_label_path, file_name + '.txt')
        if not os.path.exists(src_label):
            continue

        small_num = 0
        annotations = KittiFormatCleaner.load_annotations(src_label)
        filter_annotations = []
        for annotation in annotations:
            bbox = annotation['bbox_2d']
            # enlarge the bounding box
            box_h = bbox[3] - bbox[1] + 1
            box_w = bbox[2] - bbox[0] + 1
            if box_h < 16 or box_w < 16:
                small_num = small_num + 1
                continue

            if annotation['class'] not in class_type:
                filter_annotations.append(annotation)
        KittiFormatCleaner.save_annotations(file_name, filter_annotations,
                                            dst_label_path)
    print("Small objects: ", small_num)


# Cleanup the calibration file,  change Tr_velodyne_to_cam to Tr_velo_to_cam_0 for front camera images
def cleanup_calib(src_folder, src_folder_new):
    if not os.path.exists(src_folder_new):
        os.makedirs(src_folder_new)
    for idx, file_name in enumerate(os.listdir(src_folder)):
        if idx % 100 == 0:
            print(idx)
        file_path = os.path.join(src_folder, file_name)
        file_path_new = os.path.join(src_folder_new, file_name)
        file = open(file_path, 'r')
        file_new = open(file_path_new, 'w')
        lines = file.readlines()
        calib_keys = []
        for line in lines:

            line_sp = line.split(' ')
            if line_sp[0] in calib_keys:
                continue
            else:
                calib_keys.append(line_sp[0])

            if 'Tr_velodyne_to_cam' in line_sp[0]:
                if 'Tr_velodyne_to_cam:' in line_sp[0]:
                    line_new = line.replace('Tr_velodyne_to_cam',
                                            'Tr_velo_to_cam_0')
                else:
                    line_new = line.replace('Tr_velodyne_to_cam',
                                            'Tr_velo_to_cam')
                file_new.write(line_new)
            else:
                file_new.write(line)
        file_new.close()
        file.close()


# Cleanup the class type to Capitalize format
def cleanup_data_capitalize(src_folder, src_folder_new):
    if not os.path.exists(src_folder_new):
        os.makedirs(src_folder_new)
    for file_name in os.listdir(src_folder):
        file_path = os.path.join(src_folder, file_name)
        file_path_new = os.path.join(src_folder_new, file_name)
        file = open(file_path, 'r')
        file_new = open(file_path_new, 'w')
        lines = file.readlines()
        for line in lines:
            cls = line.strip().split()[0]
            print(cls, cls.capitalize())
            line_new = line.replace(cls, cls.capitalize())
            file_new.write(line_new)
        file_new.close()


# Calculate the number of the dataset classes
def statistic_class(root_path, dataset, type, camera_view):
    if dataset == 'waymo' and camera_view == None:
        print("Please setup the type of camera view, eg, front")
        return
    list_name = os.path.join(root_path, 'training', 'ImageSets',
                             '{}.txt'.format(type))
    sub_folder = os.path.join(root_path, 'training', 'label_2')
    if dataset == 'waymo':
        list_name = os.path.join(root_path, 'train', 'ImageSets',
                                 '{}_{}.txt'.format(type, camera_view))
        sub_folder = os.path.join(root_path, 'train', 'label_2', camera_view)
    if not os.path.exists(list_name):
        return []
    list_ptr = open(list_name, 'r')

    cls_num = {}
    for file_name in list_ptr.readlines():
        file_path = os.path.join(sub_folder, file_name.strip() + '.txt')
        file = open(file_path, 'r')
        lines = file.readlines()
        for line in lines:
            cls = line.strip().split()[0]
            if cls not in cls_num.keys():
                cls_num[cls] = 1
            else:
                cls_num[cls] = cls_num[cls] + 1
        file.close()
    print('The number of dataset {}/{}'.format(dataset, type))
    for cls in TYPE_ID_CONVERSION[dataset].keys():
        print("{:15s} {:5d}".format(cls, cls_num[cls])) if cls in cls_num.keys()\
              else print("{:15s} {:5d}".format(cls, 0))

# Calculate the average dimension, mean and std of depth of dataset classes
def statistic_dimension_and_depth(src_folder, dataset, type, camera_view, max_depth=40):
    cls_num = {}
    cls_dimension = {}
    cls_depth = []
    cls_bbox = {}

    if dataset == 'waymo':
        list_ptr = open(
            os.path.join(src_folder, type, 'ImageSets', '{}_{}.txt'.format(type, camera_view)), 'r')
        label_folder = os.path.join(src_folder, type, 'label_2', camera_view)
    elif dataset == 'jdx':
        list_ptr = open(
            os.path.join(src_folder, 'training', 'ImageSets', type + '.txt'), 'r')
        label_folder = os.path.join(src_folder, 'training', 'label_2')
    for file_name in list_ptr.readlines():
        file_path = os.path.join(label_folder, file_name.strip() + '.txt')
        annotations = KittiFormatCleaner.load_annotations(file_path)
        for annotation in annotations:
            cls = annotation['class']
            if annotation['locations'][-1] > max_depth:
                continue
            if cls not in cls_num.keys():
                cls_num[cls] = 1
                cls_dimension[cls] = {}
                cls_dimension[cls]['length'] = [annotation['dimensions'][0]]
                cls_dimension[cls]['height'] = [annotation['dimensions'][1]]
                cls_dimension[cls]['width'] = [annotation['dimensions'][2]]

                cls_bbox[cls] = {}
                cls_bbox[cls]['height'] = [annotation['bbox_2d'][3] - annotation['bbox_2d'][1]]
                cls_bbox[cls]['width'] = [annotation['bbox_2d'][2] - annotation['bbox_2d'][0]]
            else:
                cls_num[cls] = cls_num[cls] + 1
                cls_dimension[cls]['length'].append(annotation['dimensions'][0])
                cls_dimension[cls]['height'].append(annotation['dimensions'][1])
                cls_dimension[cls]['width'].append(annotation['dimensions'][2])

                cls_bbox[cls]['height'].append(annotation['bbox_2d'][3] - annotation['bbox_2d'][1])
                cls_bbox[cls]['width'].append(annotation['bbox_2d'][2] - annotation['bbox_2d'][0])
            cls_depth.append(annotation['locations'][-1])

    print("The number of each class:\n")
    for cls in cls_num.keys():
        print("{:15s} {:5d}".format(cls, cls_num[cls]))

    print("\nThe average dimension of each class:\n")
    for cls in cls_num.keys():
        print("{:15s} {:6.3f},{:6.3f},{:6.3f}".format(
             cls, np.mean(cls_dimension[cls]['length']),
             np.mean(cls_dimension[cls]['height']),
             np.mean(cls_dimension[cls]['width'])))

    print("\nThe average bbox_2d of each class:\n")
    for cls in cls_num.keys():
        print("{:15s} {:6.3f},{:6.3f}".format(
            cls, np.mean(cls_bbox[cls]['width']),
            np.mean(cls_bbox[cls]['height'])))

    print("\nThe Mean and std of Depth:{:6.3f},{:6.3f}".format(
          np.mean(cls_depth),
          np.std(cls_depth, ddof=1)))


# Generate the label list of dataset, with special classes.
# If want all labels, please setup all classes
def generate_list_with_special_class(root_path, dataset, camera_view=None,
                                     class_list=['Car', 'Pedestrian', 'Cyclist',
                                                 'Truck', 'Bus', 'Tricycle',
                                                 'Motobike']):
    print('Processing {} .....'.format(dataset))
    if dataset == 'waymo' and camera_view == None:
        print("Please setup the type of camera view, eg, front")
        return
    list_ptr = open(
        os.path.join(root_path, 'training', 'ImageSets', 'trainval.txt'), 'w')
    sub_folder = os.path.join(root_path, 'training', 'label_2')
    if dataset == 'waymo':
        list_ptr = open(os.path.join(root_path, 'train', 'ImageSets',
                                     'train_{}.txt'.format(camera_view)), 'w')
        sub_folder = os.path.join(root_path, 'train', 'label_2', camera_view)

    line_list = []
    for file_name in os.listdir(sub_folder):
        line_list.append(file_name.split('.')[0])
    line_list.sort()

    for line in line_list:
        annotations = KittiFormatCleaner.load_annotations(
            os.path.join(sub_folder, line + '.txt'))
        for annotation in annotations:
            if annotation['class'] in class_list:
                list_ptr.write(line + '\n')
                break
    list_ptr.close()


def generate_list(root_path, dataset, camera_view=None):
    print('Processing {} .....'.format(dataset))
    if dataset == 'waymo' and camera_view == None:
        print("Please setup the type of camera view, eg, front")
        return
    if not os.path.exists(os.path.join(root_path, 'training', 'ImageSets')):
        os.makedirs(os.path.join(root_path, 'training', 'ImageSets'))
    list_ptr = open(
        os.path.join(root_path, 'training', 'ImageSets', 'trainval.txt'), 'w')
    sub_folder = os.path.join(root_path, 'training', 'label_2')
    if dataset == 'waymo':
        list_ptr = open(os.path.join(root_path, 'train', 'ImageSets',
                                     'train_{}.txt'.format(camera_view)), 'w')
        sub_folder = os.path.join(root_path, 'train', 'label_2', camera_view)

    line_list = []
    for file_name in os.listdir(sub_folder):
        line_list.append(file_name.split('.')[0])
    line_list.sort()

    for line in line_list:
        list_ptr.write(line + '\n')
    list_ptr.close()
    shutil.copy(os.path.join(root_path, 'training', 'ImageSets', 'trainval.txt'),
                os.path.join(root_path, 'training', 'ImageSets', 'train.txt'))
    shutil.copy(os.path.join(root_path, 'training', 'ImageSets', 'trainval.txt'),
                os.path.join(root_path, 'training', 'ImageSets', 'val.txt'))

# transform jdx dataset with gamma correction param
def gamma_correction(src_dir, gamma=0.45):
    default_gamma = 0.8
    src_img_dir = os.path.join(src_dir, "training", "image_2")
    dst_img_dir = os.path.join(src_dir, "training", "image_2_gamma")
    assert os.path.exists(src_img_dir), "No exist image_2 folder!"
    if not os.path.exists(dst_img_dir):
        os.makedirs(dst_img_dir)

    gamma = gamma / default_gamma
    for idx, file_name in enumerate(os.listdir(src_img_dir)):
        if idx % 100 == 0:
            print(idx)
        src_path = os.path.join(src_img_dir, file_name)
        dst_path = os.path.join(dst_img_dir, file_name)
        image_src = cv2.imread(src_path)
        img = image_src.astype(np.float32) / 255
        corrected_img = np.power(img, gamma)
        corrected_img = (corrected_img * 255).astype(np.uint8)
        cv2.imwrite(dst_path, corrected_img)


def move_files(src_folder, dst_folder, list_file):
    cam_list = ['front', 'front_left', 'front_right', 'side_left', 'side_right']
    sub_list = ['calib', 'velodyne', 'image_2', 'label_2']
    ext_list = ['.txt', '.bin', '.png', '.txt']
    lines = open(list_file, 'r')
    for file_name in lines:
        file_name = file_name.strip()

        for idx, sub in enumerate(sub_list):
            if not os.path.exists(os.path.join(dst_folder, sub)):
                os.makedirs(os.path.join(dst_folder, sub))
            if not os.path.exists(
                    os.path.join(src_folder, sub, file_name + ext_list[idx])):
                continue
            print(os.path.join(src_folder, sub, file_name + ext_list[idx]))
            shutil.copy(
                os.path.join(src_folder, sub, file_name + ext_list[idx]),
                os.path.join(dst_folder, sub, file_name + ext_list[idx]))


def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    w = right_bottom[0] - left_top[0]
    h = right_bottom[1] - left_top[1]
    return h * w


def iou_of(boxes0, boxes1, criterion=-1):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
    Returns:
        iou (N): IoU values.
    """
    overlap_area = np.zeros([len(boxes0), len(boxes1)])
    area0 = np.zeros([len(boxes0), len(boxes1)])
    area1 = np.zeros([len(boxes0), len(boxes1)])
    for idx0, box0 in enumerate(boxes0):
        for idx1, box1 in enumerate(boxes1):
            overlap_left_top = [max(box0[0], box1[0]), max(box0[1], box1[1])]
            overlap_right_bottom = [min(box0[2], box1[2]),
                                    min(box0[3], box1[3])]
            if overlap_right_bottom[0] <= overlap_left_top[0] or \
                    overlap_right_bottom[1] <= overlap_left_top[1]:
                overlap_area[idx0][idx1] = 0
            else:
                overlap_area[idx0][idx1] = area_of(overlap_left_top,
                                                   overlap_right_bottom)
            area0[idx0][idx1] = area_of(box0[:2], box0[2:])
            area1[idx0][idx1] = area_of(box1[:2], box1[2:])

    if criterion == -1:
        return overlap_area / (area0 + area1 - overlap_area + 1e-5)
    elif criterion == 0:
        return overlap_area / (area0 + 1e-5)
    elif criterion == 1:
        return overlap_area / (area1 + 1e-5)


class KittiFormatCleaner(object):
    """
        Class for cleanning annotations which are not visuable in camera-view.
    """

    def __init__(self, root_path='', type='val', camera_view='front',
                 dataset_type='waymo', save_path='demo'):
        self.dataset_type = dataset_type
        self.camera_view = None if camera_view is None else camera_view
        self.type = type
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if self.dataset_type == 'waymo':
            self.velodyne_folder = os.path.join(root_path, self.type,
                                                'velodyne')
            self.calib_folder = os.path.join(root_path, self.type, 'calib')
            self.label_folder = os.path.join(root_path, self.type, 'label_2')
            self.image_folder = os.path.join(root_path, self.type, 'image_2')
        else:
            self.velodyne_folder = os.path.join(root_path, self.type,
                                                'velodyne')
            self.calib_folder = os.path.join(root_path, 'training', 'calib')
            self.label_folder = os.path.join(root_path, 'training', 'label_2')
            self.image_folder = os.path.join(root_path, 'training', 'image_2')
        if self.camera_view is not None:
            self.label_folder = os.path.join(root_path, self.type, 'label_2',
                                             self.camera_view)
            self.image_folder = os.path.join(root_path, self.type, 'image_2',
                                             self.camera_view)
        image_names = []
        for file in os.listdir(self.label_folder):
            image_names.append(file.split('.')[0])
        self.image_names = image_names

    def crop(self):
        for idx, image_name in enumerate(self.image_names):
            print(image_name)
            image_path = os.path.join(self.image_folder, image_name + '.png')
            if not os.path.exists(image_path):
                continue
            image_src = cv2.imread(image_path)
            if image_src is None:
                continue

            label_path = os.path.join(self.label_folder, image_name + '.txt')
            calib_path = os.path.join(self.calib_folder, image_name + '.txt')
            annotations = self.load_annotations(label_path)
            K, P = self.load_intrinsic_matrix(calib_path, self.camera_view)
            _, cam_to_vel = self.load_external_matrix(calib_path,
                                                       self.camera_view)

            image_vis = image_src.copy()
            for idx, annotation in enumerate(annotations):
                height, width, _ = image_src.shape
                center = self.decode_center_2d(annotation['dimensions'],
                                               annotation['locations'], K)
                bbox = annotation['bbox_2d']
                bbox[0] = max(bbox[0], 0)
                bbox[1] = max(bbox[1], 0)
                bbox[2] = min(bbox[2], width)
                bbox[3] = min(bbox[3], height)
                cv2.rectangle(image_vis, (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
                if 0 <= int(center[0]) < width and 0 <= int(center[1]) < height:
                    cv2.circle(image_vis, (int(center[0]), int(center[1])), 5,
                               (255, 0, 0), -1)

            for idx, annotation in enumerate(annotations):
                image_crop = image_vis.copy()
                height, width, _ = image_src.shape
                center = self.decode_center_2d(annotation['dimensions'],
                                               annotation['locations'], K)
                bbox = annotation['bbox_2d']
                bbox[0] = max(bbox[0], 0)
                bbox[1] = max(bbox[1], 0)
                bbox[2] = min(bbox[2], width)
                bbox[3] = min(bbox[3], height)
                cv2.rectangle(image_crop, (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])), (0, 255, 255), 2)

                # enlarge the bounding box
                box_h = bbox[3] - bbox[1] + 1
                box_w = bbox[2] - bbox[0] + 1
                if box_h < 16 or box_w < 16:
                    continue
                left_top_x = int(max(bbox[0] - box_w * 0.5, 0))
                left_top_y = int(max(bbox[1] - box_h * 0.5, 0))
                right_bottom_x = int(min(bbox[2] + box_w * 0.5, width))
                right_bottom_y = int(min(bbox[3] + box_h * 0.5, height))
                img_crop = image_crop[left_top_y:right_bottom_y,
                           left_top_x:right_bottom_x, :]

                if not os.path.exists(os.path.join(self.save_path, 'crop',
                                                   annotation['class'])):
                    os.makedirs(os.path.join(self.save_path, 'crop',
                                             annotation['class']))
                cv2.imwrite(
                    os.path.join(self.save_path, 'crop', annotation['class'],
                                 image_name + "_{}.jpg".format(idx)),
                    img_crop)

    def crop_and_classify(self):
        for idx, image_name in enumerate(self.image_names):
            print(image_name)
            image_path = os.path.join(self.image_folder, image_name + '.png')
            if not os.path.exists(image_path):
                continue
            image_src = cv2.imread(image_path)
            if image_src is None:
                continue
            height, width, _ = image_src.shape
            label_path = os.path.join(self.label_folder, image_name + '.txt')
            annotations = self.load_annotations(label_path)

            for idx, annotation in enumerate(annotations):
                print(annotation['class'])
                if annotation['class'] not in ['other_vehicle', 'Bus', 'Truck']:
                    continue

                # enlarge the bounding box
                bbox = annotation['bbox_2d']
                box_h = bbox[3] - bbox[1] + 1
                box_w = bbox[2] - bbox[0] + 1
                if box_h < 48 or box_w < 48:
                    continue
                left_top_x = int(max(bbox[0] - box_w * 0.1, 0))
                left_top_y = int(max(bbox[1] - box_h * 0.1, 0))
                right_bottom_x = int(min(bbox[2] + box_w * 0.1, width))
                right_bottom_y = int(min(bbox[3] + box_h * 0.1, height))
                img_crop = image_src[left_top_y:right_bottom_y,
                           left_top_x:right_bottom_x, :]

                # connect to the classification server
                img_encode = cv2.imencode('.jpg', img_crop)[1].tostring()
                image_b64 = base64.b64encode(img_encode)
                data = {"image": image_b64}
                result = requests.post(URL, data)
                result_dict = json.loads(result.content.decode())
                label = result_dict['label']
                save_dir = os.path.join(self.save_path, self.type, 'crop',
                                        self.camera_view, VEHICLE_CLASS[label])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(
                    os.path.join(save_dir, image_name + "_{}.jpg".format(idx)),
                    img_crop)

    def resize_and_save(self):
        vel_folder_save = os.path.join(self.save_path, self.type, 'velodyne')
        calib_folder_save = os.path.join(self.save_path, self.type, 'calib')
        label_folder_save = os.path.join(self.save_path, self.type, 'label_2')
        image_folder_save = os.path.join(self.save_path, self.type, 'image_2')
        if self.camera_view is not None:
            label_folder_save = os.path.join(self.save_path, self.type,
                                             'label_2', self.camera_view)
            image_folder_save = os.path.join(self.save_path, self.type,
                                             'image_2', self.camera_view)
        if not os.path.exists(calib_folder_save):
            os.makedirs(calib_folder_save)
        if not os.path.exists(label_folder_save):
            os.makedirs(label_folder_save)
        if not os.path.exists(image_folder_save):
            os.makedirs(image_folder_save)
        if not os.path.exists(vel_folder_save):
            os.makedirs(vel_folder_save)
        for idx, image_name in enumerate(self.image_names):
            if idx % 100 == 0:
                print(idx)

            ## copy velodyne
            vel_path = os.path.join(self.velodyne_folder, image_name + '.bin')
            shutil.copy(vel_path,
                        os.path.join(vel_folder_save, image_name + '.bin'))

            image_path = os.path.join(self.image_folder, image_name + '.png')
            if not os.path.exists(image_path):
                continue
            image_src = cv2.imread(image_path)
            if image_src is None:
                continue

            ### resize image and save
            height, width, _ = image_src.shape
            width_resize = 720
            height_resize = int(float(width_resize) / width * height)
            image_resize = cv2.resize(image_src, (width_resize, height_resize))
            cv2.imwrite(os.path.join(image_folder_save, image_name + '.png'),
                        image_resize)

            ### adjust calibration params
            calib_path = os.path.join(calib_folder_save, image_name + '.txt')
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                proj_para = 'P{}'.format(CAMERA_TO_ID[self.camera_view])
                if proj_para in line:
                    line_sp = line.strip().split(' ')
                    for idx, sp in enumerate(line_sp):
                        if idx == 0:
                            line_new = "{} ".format(sp)
                        if idx > 0 and idx <= 4:
                            sp = float(line_sp[idx]) * (
                                    float(width_resize) / width)
                            line_new = line_new + "{:e} ".format(sp)
                        elif idx > 4 and idx <= 8:
                            sp = float(line_sp[idx]) * (
                                    float(height_resize) / height)
                            line_new = line_new + "{:e} ".format(sp)
                        elif idx > 8 and idx <= 11:
                            line_new = line_new + "{} ".format(sp)
                        elif idx == 12:
                            line_new = line_new + "{}\n".format(sp)
                    lines[i] = line_new

            with open(os.path.join(calib_folder_save, image_name + '.txt'), 'w') as calib_save:
                for line in lines:
                    calib_save.write(line)

            label_path = os.path.join(self.label_folder, image_name + '.txt')
            annotations = self.load_annotations(label_path)
            for idx, annotation in enumerate(annotations):
                bbox_2d = annotation['bbox_2d']
                bbox_2d[1] = bbox_2d[1] * width_resize / width
                bbox_2d[3] = bbox_2d[3] * width_resize / width
                bbox_2d[0] = bbox_2d[0] * height_resize / height
                bbox_2d[2] = bbox_2d[2] * height_resize / height
            self.save_annotations(image_name, annotations,
                                  save_path=label_folder_save)

    def cleanup_waymo(self):
        outside_num = 0
        higher_num = 0
        bbox_num = 0
        large_vehicle_num = 0
        small_vehicle_num = 0
        occluded_num = 0
        if self.camera_view is not None:
            if not os.path.exists(os.path.join(self.save_path, 'image_visual',
                                               self.camera_view)):
                os.makedirs(os.path.join(self.save_path, 'image_visual',
                                         self.camera_view))
        else:
            if not os.path.exists(os.path.join(self.save_path, 'image_visual')):
                os.makedirs(os.path.join(self.save_path, 'image_visual'))
        points_filter = PointCloudFilter(
            side_range=range_list[self.dataset_type][0],
            fwd_range=range_list[self.dataset_type][1],
            res=range_list[self.dataset_type][-1])
        for index, image_name in enumerate(self.image_names):
            image_path = os.path.join(self.image_folder, image_name + '.png')
            image_src = cv2.imread(image_path)
            if image_src is None:
                continue
            height, width, _ = image_src.shape
            label_path = os.path.join(self.label_folder, image_name + '.txt')
            calib_path = os.path.join(self.calib_folder, image_name + '.txt')
            velodyne_path = os.path.join(self.velodyne_folder,
                                         image_name + '.bin')
            annotations = self.load_annotations(label_path)
            K, P = self.load_intrinsic_matrix(calib_path, self.camera_view)
            _, cam_to_vel = self.load_external_matrix(calib_path,
                                                       self.camera_view)
            for idx, anno in enumerate(annotations):
                center_2d = self.decode_center_2d(anno['dimensions'],
                                                  anno['locations'], K)
                annotations[idx]['center_2d'] = center_2d
                annotations[idx]['image_size'] = [width, height]

            # start to filter the label boxes
            filtered_all = []
            filtered, remained = self.filter_outside_camera_view(annotations)
            outside_num = outside_num + len(filtered)
            filtered_all.extend(filtered)

            filtered, remained = self.filter_outside_bbox_view(remained)
            bbox_num = bbox_num + len(filtered)
            filtered_all.extend(filtered)

            filtered, remained = self.filter_higher_part_of_vehicle(remained)
            higher_num = higher_num + len(filtered)
            filtered_all.extend(filtered)

            filtered, remained = self.filter_large_vehicle(remained)
            large_vehicle_num = large_vehicle_num + len(filtered)
            filtered_all.extend(filtered)

            filtered, remained = self.filter_small_vehicle(remained)
            small_vehicle_num = small_vehicle_num + len(filtered)
            filtered_all.extend(filtered)

            filtered, remained = self.filter_occlusion_camera_view(annotations)
            occluded_num = occluded_num + len(filtered)
            filtered_all.extend(filtered)

            if index % 10 == 0:
                print(index)
                image = image_src.copy()
                image_vis = draw_2d_box_on_image_gt(image, filtered_all)
                if self.camera_view is not None:
                    cv2.imwrite(
                        os.path.join(self.save_path, 'image_visual/',
                                     self.camera_view,
                                     image_name + "_filtered.jpg"),
                        image_vis)
                else:
                    cv2.imwrite(os.path.join(self.save_path, 'image_visual/',
                                             image_name + "_filtered.jpg"),
                                image_vis)

                image = image_src.copy()
                image_vis = draw_2d_box_on_image_gt(image, remained)
                if self.camera_view is not None:
                    cv2.imwrite(
                        os.path.join(self.save_path, 'image_visual/',
                                     self.camera_view,
                                     image_name + "_composed.jpg"),
                        image_vis)
                else:
                    cv2.imwrite(os.path.join(self.save_path, 'image_visual/',
                                             image_name + "_composed.jpg"),
                                image_vis)

            if self.camera_view is not None:
                self.save_annotations(image_name, remained,
                                      os.path.join(self.save_path, 'label_2',
                                                   self.camera_view))
            else:
                self.save_annotations(image_name, remained,
                                      os.path.join(self.save_path, 'label_2'))

        print("Center outside filter: ", outside_num)
        print("Bbox outside filter: ", bbox_num)
        print("Higher part filter: ", higher_num)
        print("Large vehicle filter: ", large_vehicle_num)
        print("Small vehicle filter: ", small_vehicle_num)
        print("Occlusion filter: ", occluded_num)

    def cleanup_jdx(self):
        occluded_num = 0
        if not os.path.exists(os.path.join(self.save_path, 'image_visual')):
            os.makedirs(os.path.join(self.save_path, 'image_visual'))

        for index, image_name in enumerate(self.image_names):
            image_path = os.path.join(self.image_folder, image_name + '.png')
            image_src = cv2.imread(image_path)
            if image_src is None:
                continue
            height, width, _ = image_src.shape
            label_path = os.path.join(self.label_folder, image_name + '.txt')
            calib_path = os.path.join(self.calib_folder, image_name + '.txt')
            annotations = self.load_annotations(label_path)
            K, P = self.load_intrinsic_matrix(calib_path, self.camera_view)
            _, cam_to_vel = self.load_external_matrix(calib_path,
                                                       self.camera_view)
            for idx, anno in enumerate(annotations):
                center_2d = self.decode_center_2d(anno['dimensions'],
                                                  anno['locations'], K)
                annotations[idx]['center_2d'] = center_2d
                annotations[idx]['image_size'] = [width, height]

            # start to filter the label boxes
            filtered_all = []
            filtered, remained = self.filter_occlusion_camera_view(annotations)
            occluded_num = occluded_num + len(filtered)
            filtered_all.extend(filtered)

            if index % 10 == 0:
                print(index)
                image = image_src.copy()
                image_vis_filtered = draw_2d_box_on_image_gt(image,
                                                             filtered_all)
                image = image_src.copy()
                image_vis_remained = draw_2d_box_on_image_gt(image, remained)
                if self.camera_view is not None:
                    cv2.imwrite(
                        os.path.join(self.save_path, 'image_visual/',
                                     self.camera_view,
                                     image_name + "_filtered.jpg"),
                        image_vis_filtered)
                    cv2.imwrite(
                        os.path.join(self.save_path, 'image_visual/',
                                     self.camera_view,
                                     image_name + "_composed.jpg"),
                        image_vis_remained)
                else:
                    cv2.imwrite(os.path.join(self.save_path, 'image_visual/',
                                             image_name + "_filtered.jpg"),
                                image_vis_filtered)
                    cv2.imwrite(os.path.join(self.save_path, 'image_visual/',
                                             image_name + "_composed.jpg"),
                                image_vis_remained)

            if self.camera_view is not None:
                self.save_annotations(image_name, remained,
                                      os.path.join(self.save_path, 'label_2',
                                                   self.camera_view))
            else:
                self.save_annotations(image_name, remained,
                                      os.path.join(self.save_path, 'label_2'))

        print("Occlusion filter: ", occluded_num)

    @staticmethod
    def save_annotations(image_name, annotations, save_path=None):
        if save_path is None:
            save_path = '/media/jd/data/temp/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, image_name + '.txt'),
                  'w') as filter_file:
            for idx, annotation in enumerate(annotations):
                if annotation["class"] == 'Vehicle':
                    annotation["class"] = 'Car'
                line = '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
                    annotation["class"],
                    annotation["truncation"],
                    annotation["occlusion"],
                    annotation["alpha"],
                    annotation["bbox_2d"][0],
                    annotation["bbox_2d"][1],
                    annotation["bbox_2d"][2],
                    annotation["bbox_2d"][3],
                    annotation["dimensions"][1],
                    annotation["dimensions"][2],
                    annotation["dimensions"][0],
                    annotation["locations"][0],
                    annotation["locations"][1],
                    annotation["locations"][2],
                    annotation["rot_y"])
                filter_file.write(line)

    @staticmethod
    def load_annotations(file_path):
        annotations = []
        fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin',
                      'xmax', 'ymax', 'dh', 'dw',
                      'dl', 'lx', 'ly', 'lz', 'ry']

        with open(file_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=' ',
                                    fieldnames=fieldnames)
            for line, row in enumerate(reader):
                annotations.append({
                    "class": row["type"].capitalize(),
                    "label": 0,
                    "truncation": float(row["truncated"]),
                    "occlusion": float(row["occluded"]),
                    "alpha": float(row["alpha"]),
                    "bbox_2d": [int(float(row["xmin"])),
                                int(float(row["ymin"])),
                                int(float(row["xmax"])),
                                int(float(row["ymax"]))],
                    "dimensions": [float(row['dl']), float(row['dh']),
                                   float(row['dw'])],
                    "locations": [float(row['lx']), float(row['ly']),
                                  float(row['lz'])],
                    "rot_y": float(row["ry"])
                })
        return annotations

    def decode_center_2d(self, dimention, location, K):
        l, h, w = dimention[0], dimention[1], dimention[2]
        x, y, z = location[0], location[1], location[2]

        center_3d = np.array([x, y - h / 2, z])
        proj_point = np.matmul(K, center_3d)
        center_2d = proj_point[:2] / proj_point[2]

        return center_2d

    def load_external_matrix(self, calib_file, camera_type):
        tran_type = ('Tr_velo_to_cam:' if camera_type is None
                     else 'Tr_velo_to_cam_{}:'.format(CAMERA_TO_ID[camera_type]))

        with open(calib_file, 'r') as f:
            for line in f.readlines():
                if line.split(' ')[0] == 'R0_rect:':
                    R0_rect = np.zeros((4, 4))
                    R0_rect[:3, :3] = np.array(
                        line.split('\n')[0].split(' ')[1:]).astype(
                        float).reshape(3, 3)
                    R0_rect[3, 3] = 1
                if line.split(' ')[0] == tran_type:
                    Tr_velo_to_cam = np.zeros((4, 4))
                    Tr_velo_to_cam[:3, :4] = np.array(
                        line.split('\n')[0].split(' ')[1:]).astype(
                        float).reshape(3, 4)
                    Tr_velo_to_cam[3, 3] = 1

            vel_to_cam = np.dot(R0_rect, Tr_velo_to_cam)
            cam_to_vel = np.linalg.inv(vel_to_cam)
            return vel_to_cam, cam_to_vel

    def load_intrinsic_matrix(self, calib_file, camera_view):
        proj_type = 'P2:' if camera_view is None else 'P{}:'.format(
            CAMERA_TO_ID[camera_view])
        with open(calib_file, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=' ')
            for line, row in enumerate(reader):
                if row[0] == proj_type:
                    P = row[1:]
                    P = [float(i) for i in P]
                    P = np.array(P, dtype=np.float32).reshape(3, 4)
                    K = P[:3, :3]
                    break
        return K, P

    # delete the label whose center is occluded in camera view based on 3D box
    @staticmethod
    def filter_occlusion_camera_view(annotations):
        if len(annotations) <= 1:
            return [], annotations
        filter_index = []

        boxes = []
        idxes = []
        for idx, annotation in enumerate(annotations):
            idxes.append(idx)
            boxes.append(np.array(annotation["bbox_2d"]))
        iou = iou_of(np.array(boxes), np.array(boxes), criterion=-1).astype(
            np.float64)
        iou_first = iou_of(np.array(boxes), np.array(boxes),
                           criterion=0).astype(np.float64)
        iou_second = iou_of(np.array(boxes), np.array(boxes),
                            criterion=1).astype(np.float64)

        # maybe use the rotation
        for idx_x in range(len(boxes)):
            for idx_y in range(len(boxes)):
                if iou[idx_x][idx_y] <= 0 or idx_x >= idx_y:
                    continue
                if iou[idx_x][idx_y] > IOU_THRESHOLD:
                    if (annotations[idx_x]['locations'][2] >
                            annotations[idx_y]['locations'][2] +
                            annotations[idx_y]['dimensions'][0] / 2):
                        filter_index.append(idxes[idx_x])
                    if (annotations[idx_y]['locations'][2] >
                            annotations[idx_x]['locations'][2] +
                            annotations[idx_y]['dimensions'][0] / 2):
                        filter_index.append(idxes[idx_y])
                else:
                    if (annotations[idx_x]['locations'][2] >
                            annotations[idx_y]['locations'][2] +
                            annotations[idx_y]['dimensions'][0] / 2
                            and iou_first[idx_x][idx_y] > SELF_IOU_THRESHOLD):
                        filter_index.append(idxes[idx_x])

                    if (annotations[idx_y]['locations'][2] >
                            annotations[idx_x]['locations'][2] +
                            annotations[idx_y]['dimensions'][0] / 2
                            and iou_second[idx_x][idx_y] > SELF_IOU_THRESHOLD):
                        filter_index.append(idxes[idx_y])

        filtered = []
        remained = []
        for idx, annotation in enumerate(annotations):
            if idx in filter_index:
                filtered.append(annotation)
            else:
                remained.append(annotation)

        return filtered, remained

    # delete the label whose center is out of camera view based on 3D box
    @staticmethod
    def filter_outside_bbox_view(annotations):
        filter_index = []
        for idx, annotation in enumerate(annotations):
            if annotation['bbox_2d'] == [0, 0, 0, 0]:
                filter_index.append(idx)

        filtered = []
        remained = []
        for idx, annotation in enumerate(annotations):
            if idx in filter_index:
                filtered.append(annotation)
            else:
                remained.append(annotation)

        return filtered, remained

    # delete the label whose center is out of camera view based on 3D box
    @staticmethod
    def filter_outside_camera_view(annotations):
        filter_index = []
        for idx, annotation in enumerate(annotations):
            if not ((0 <= float(annotation['center_2d'][0]) <
                     annotation['image_size'][0])
                    or not (0 <= float(annotation['center_2d'][1]) <
                            annotation['image_size'][1])):
                filter_index.append(idx)

        filtered = []
        remained = []
        for idx, annotation in enumerate(annotations):
            if idx in filter_index:
                filtered.append(annotation)
            else:
                remained.append(annotation)

        return filtered, remained

    # delete the label which is the higher part of class Vehicle
    @staticmethod
    def filter_higher_part_of_vehicle(annotations):
        if len(annotations) <= 1:
            return [], annotations
        filter_index = []
        boxes = []
        idxes = []
        for idx, annotation in enumerate(annotations):
            if annotation['class'] in ['Vehicle', 'Car']:
                idxes.append(idx)
                boxes.append(
                    [annotation["locations"][0], annotation["locations"][2],
                     annotation["dimensions"][0],
                     annotation["dimensions"][2], annotation["rot_y"]])
        riou = rotate_iou_gpu_eval(np.array(boxes), np.array(boxes),
                                   criterion=-1).astype(np.float64)
        riou_first = rotate_iou_gpu_eval(np.array(boxes), np.array(boxes),
                                         criterion=0).astype(np.float64)
        riou_second = rotate_iou_gpu_eval(np.array(boxes), np.array(boxes),
                                          criterion=1).astype(np.float64)

        for idx_x in np.arange(len(idxes)):
            for idx_y in np.arange(len(idxes)):
                if idx_x >= idx_y:
                    continue
                if riou[idx_x][idx_y] > VEHICLE_RIOU_THRESHOLD:
                    if riou_first[idx_x][idx_y] > riou_second[idx_x][idx_y]:
                        filter_index.append(idxes[idx_y])
                    else:
                        filter_index.append(idxes[idx_x])

        filtered = []
        remained = []
        for idx, annotation in enumerate(annotations):
            if idx in filter_index:
                filtered.append(annotation)
            else:
                remained.append(annotation)

        return filtered, remained

    # change some Vehicle to Bus
    @staticmethod
    def filter_large_vehicle(annotations):
        filter_index = []
        for idx, annotation in enumerate(annotations):
            if (annotation['class'] in ['Vehicle', 'Car']
                    and annotation["dimensions"][0] > VEHICLE_LARGE_SCALE):
                annotations[idx]['class'] = 'Truck'
                filter_index.append(idx)

        filtered = []
        remained = []
        for idx, annotation in enumerate(annotations):
            if idx in filter_index:
                filtered.append(annotation)
            remained.append(annotation)

        return filtered, remained

    # change some small Vehicle to motobike
    @staticmethod
    def filter_small_vehicle(annotations):
        filter_index = []
        for idx, annotation in enumerate(annotations):
            if (annotation['class'] in ['Vehicle', 'Car'] and
                    annotation["dimensions"][0] < VEHICLE_SMALL_SCALE):
                annotations[idx]['class'] = 'Motobike'
                filter_index.append(idx)

        filtered = []
        remained = []
        for idx, annotation in enumerate(annotations):
            if idx in filter_index:
                filtered.append(annotation)
            remained.append(annotation)

        return filtered, remained


def draw_2d_box_on_image_gt(image, annotations, color=(0, 0, 255)):
    if len(annotations) <= 0:
        return image
    for idx, annotation in enumerate(annotations):
        if annotation["class"] == 'Vehicle':
            annotation["class"] = 'Car'
        if annotation['bbox_2d'] == [0, 0, 0, 0]:
            continue
        bbox = np.array(annotation['bbox_2d']).astype(float)

        # opencv image
        height, width, _ = image.shape
        bbox[0] = max(bbox[0], 0)
        bbox[1] = max(bbox[1], 0)
        bbox[2] = min(bbox[2], width)
        bbox[3] = min(bbox[3], height)
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.putText(image, "{}".format(annotation["class"]),
                    (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_COMPLEX,
                    1.0, color, 2)

        center = np.array(annotation['center_2d']).astype(float)
        if 0 <= int(center[0]) < width and 0 <= int(center[1]) < height:
            cv2.circle(image, (int(center[0]), int(center[1])), 5, color, -1)
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset tool.")
    parser.add_argument("--src_path", type=str,
                        default="datasets/jdx_test/front/",
                        help="Path to the src dataset.")
    parser.add_argument("--dst_path", type=str,
                        default='demo/jdx_test/front/',
                        help='Path to save dst images.')
    parser.add_argument("--dataset", type=str, default='jdx',
                        help='Specify dataset type.')
    parser.add_argument("--type", type=str, default="val",
                        help='Specify dataset part type.')
    parser.add_argument("--camera_view", type=str, default=None,
                        help='Specify camera view.')
    parser.add_argument("--class_list", type=str,
                        default=['Car', 'Pedestrian', 'Cyclist', 'Truck', 'Bus',
                                 'Tricycle', 'Cyclist_stop'],
                        help='Specify class type list.')
    parser.add_argument("--option", type=int, default=1,
                        help='1 : statistic the number of each class.'
                             '2 : generate the label list of the dataset.'
                             '3 : clean up datasets.'
                             '4 : crop labeled part of dataset image.'
                             '5 : update labels.'
                             '6 : gamma correction.')
    args = parser.parse_args()

    if args.option == 1:
        for type in ['trainval', 'train', 'val']:
            statistic_class(args.src_path, args.dataset, type, args.camera_view)

    if args.option == 2:
        generate_list(args.src_path, args.dataset, args.camera_view)

    ################################ The flowpath of the cleanup of waymo dataset #############################
    #       1. run clean_calib(): change 'Tr_velodyne_to_cam' to 'Tr_velo_to_cam_{i}' for each camera view
    #       2. KittiFormatCleaner.cleanup(): clean up the labels which maybe outside of each camera view, or occlusion
    #       3. KittiFormatCleaner.crop_and_classify(): Connecting the classification server, classify the label which is filtered as Truck
    #       4. Change the label class according to the Bus/Truck classification result, if
    #       5. KittiFormatCleaner.resize_and_save(): Copy the source calib files and change them,
    #                                           Resize the image to 720x480 for each camera image and change calib file
    if args.option == 3:
        cleaner = KittiFormatCleaner(args.src_path, args.type,
                                     camera_view=args.camera_view,
                                     dataset_type=args.dataset,
                                     save_path=args.dst_path)
        if args.dataset == 'waymo':
            cleaner.cleanup_waymo()
        else:
            cleaner.cleanup_jdx()

    if args.option == 4:
        cleaner = KittiFormatCleaner(args.src_path, args.type,
                                     camera_view=args.camera_view,
                                     dataset_type=args.dataset,
                                     save_path=args.dst_path)
        cleaner.crop()

    if args.option == 5:
        change_class_type(args.dst_path, args.src_path, args.dst_path)
        filter_label(args.dst_path, args.dst_path)

    if args.option == 6:
        gamma_correction(args.src_path)
