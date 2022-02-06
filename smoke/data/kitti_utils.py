# -*- coding:utf8 -*-
import argparse
import os
import shutil
import cv2
import csv
import random
import numpy as np
from PIL import Image, ImageDraw

EPSILON = 1e-5
CLASS_TO_ID = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
    'Truck': 3,
    'Tricycle': 4,
    'Bus': 5,
    'Motobike': 6,
    'Cyclist_stopped': 6,
    'Others': 7,
    'Misc': 7
}

CAMERA_TO_ID = {
    'front': 0,
    'front_left': 1,
    'front_right': 2,
    'side_left': 3,
    'side_right': 4,
}

COLOR = {
    0: (0, 0, 255),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 255, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
    6: (127, 0, 127),
    7: (255, 255, 255)
}


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


def iou_2d(box_a, box_b, criterion=-1):
    """
        Return intersection-over-union (Jaccard index) of boxes.
    """
    overlap_left_top = [max(box_a[0], box_b[0]), max(box_a[1], box_b[1])]
    overlap_right_bottom = [min(box_a[2], box_b[2]),
                            min(box_a[3], box_b[3])]
    if overlap_right_bottom[0] <= overlap_left_top[0] or \
            overlap_right_bottom[1] <= overlap_left_top[1]:
        overlap_area = 0
    else:
        overlap_area = area_of(overlap_left_top,
                               overlap_right_bottom)
    area_a = area_of(box_a[:2], box_a[2:])
    area_b = area_of(box_b[:2], box_b[2:])
    if criterion == -1:
        return overlap_area / (area_a + area_b - overlap_area + EPSILON)
    elif criterion == 0:
        return overlap_area / (area_a + EPSILON)
    elif criterion == 1:
        return overlap_area / (area_b + EPSILON)


def setup_save_path(save_path):
    vel_folder_save = os.path.join(save_path, 'velodyne')
    calib_folder_save = os.path.join(save_path, 'calib')
    label_folder_save = os.path.join(save_path, 'label_2')
    image_folder_save = os.path.join(save_path, 'image_2')
    visual_folder_save = os.path.join(save_path, 'visual')
    if not os.path.exists(calib_folder_save):
        os.makedirs(calib_folder_save)
    if not os.path.exists(label_folder_save):
        os.makedirs(label_folder_save)
    if not os.path.exists(image_folder_save):
        os.makedirs(image_folder_save)
    if not os.path.exists(vel_folder_save):
        os.makedirs(vel_folder_save)
    if not os.path.exists(visual_folder_save):
        os.makedirs(visual_folder_save)


def save_annotations(image_name, annotations, save_path=None):
    assert save_path is not None, 'Please assign the path to save'
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


def load_annotation(file_path, class_list=None, depth_range=[0, 30],
                    trunction_filter=[0, 1, 2, 3, 4, 5],
                    occlusion_filter=[0, 1, 2, 3, 4, 5]):
    annotations = []
    fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin',
                  'xmax', 'ymax', 'dh', 'dw',
                  'dl', 'lx', 'ly', 'lz', 'ry']

    with open(file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=' ',
                                fieldnames=fieldnames)
        for line, row in enumerate(reader):
            type_cap = row["type"].capitalize()
            truncated = int(float(row["truncated"]))
            occluded = int(float(row["occluded"]))
            depth = float(row['lz'])
            # filter the annotation which is not satisfied
            if class_list is not None and type_cap not in class_list:
                continue
            if truncated in trunction_filter and occluded in occlusion_filter and depth > \
                    depth_range[0] and depth < depth_range[1]:
                annotations.append({
                    "class": type_cap,
                    "label": CLASS_TO_ID[type_cap],
                    "truncation": truncated,
                    "occlusion": occluded,
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

# cls_shots:      
# weight_gamma:   [0.5, 0.5]
# dimension_reference:  
                # ((4.392, 1.658, 1.910),
                #  (1.773, 1.525, 0.740),
                #  (0.505, 1.644, 0.582),
                #  (7.085, 2.652, 2.523),
                #  (2.790, 1.651, 1.201),
                #  (8.208, 2.869, 2.645))
def calculate_class_weight(cls_shots, weight_gamma, dimension_reference):
    assert np.array(cls_shots).any()
    cls_ratios = cls_shots / np.sum(cls_shots)
    cls_weights = [
        pow(1 / np.max(cls_ratios), weight_gamma[0]) if item == 0 else pow(
            1 / cls_ratios[index], weight_gamma[0])
        for index, item in enumerate(cls_shots)]
    cls_weights /= np.min(cls_weights)
    size_weight = [sum(x) for x in dimension_reference]
    size_weight = [np.power(size_weight[0] / x, weight_gamma[1]) for
                   x in size_weight]
    size_weight = np.array(size_weight, dtype="float")
    cls_weights *= size_weight
    return cls_weights


def save_calibration(image_name, intrinsic_matrix, extrinsic_matrix,
                     save_path=None):
    assert save_path is not None, 'Please assign the path to save'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    kitti_transforms = dict()
    kitti_transforms["P0"] = np.zeros((3, 4))
    kitti_transforms["P1"] = np.zeros((3, 4))
    kitti_transforms["P2"] = intrinsic_matrix
    kitti_transforms["P3"] = np.zeros((3, 4))
    kitti_transforms["R0_rect"] = np.identity(3)
    kitti_transforms["Tr_velo_to_cam"] = extrinsic_matrix[:3, :]
    kitti_transforms["Tr_imu_to_velo"] = np.zeros((3, 4))
    with open(os.path.join(save_path, image_name + '.txt'), "w") as f:
        for (key, val) in kitti_transforms.items():
            val = val.flatten()
            val_str = "%.12e" % val[0]
            for v in val[1:]:
                val_str += " %.12e" % v
            f.write("%s: %s\n" % (key, val_str))


def save_calibration_aug(image_name, intrinsic_matrix_list, extrinsic_matrix,
                         save_path=None):
    assert save_path is not None, 'Please assign the path to save'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    kitti_transforms = dict()
    kitti_transforms["P0"] = np.zeros((3, 4))
    kitti_transforms["P1"] = np.zeros((3, 4))
    # kitti_transforms["P2"] = intrinsic_matrix_list
    kitti_transforms["P3"] = np.zeros((3, 4))
    kitti_transforms["R0_rect"] = np.identity(3)
    kitti_transforms["Tr_velo_to_cam"] = extrinsic_matrix[:3, :]
    kitti_transforms["Tr_imu_to_velo"] = np.zeros((3, 4))
    with open(os.path.join(save_path, image_name + '.txt'), "w") as f:
        for (key, val) in kitti_transforms.items():
            val = val.flatten()
            val_str = "%.12e" % val[0]
            for v in val[1:]:
                val_str += " %.12e" % v
            f.write("%s: %s\n" % (key, val_str))
        for intrinsic_matrix in intrinsic_matrix_list:
            val = intrinsic_matrix.flatten()
            val_str = "%.12e" % val[0]
            for v in val[1:]:
                val_str += " %.12e" % v
            f.write("P2: %s\n" % (val_str))


def load_intrinsic_matrix(calib_file, camera_type=None):
    proj_type = 'P2:' if camera_type is None else 'P{}:'.format(
        CAMERA_TO_ID[camera_type])
    with open(os.path.join(calib_file), 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for line, row in enumerate(reader):
            if row[0] == proj_type:
                P = row[1:]
                P = [float(i) for i in P]
                P = np.array(P, dtype=np.float32).reshape(3, 4)
                K = P[:3, :3]
                break
    return K, P


def load_external_matrix(calib_file, camera_type=None):
    tran_type = 'Tr_velo_to_cam:' if camera_type is None else 'Tr_velo_to_cam_{}:'.format(
        CAMERA_TO_ID[camera_type])
    with open(calib_file, 'r') as f:
        for line in f.readlines():
            if (line.split(' ')[0] == 'R0_rect:'):
                R0_rect = np.zeros((4, 4))
                R0_rect[:3, :3] = np.array(
                    line.split('\n')[0].split(' ')[1:]).astype(float).reshape(3,
                                                                              3)
                R0_rect[3, 3] = 1
            if (line.split(' ')[0] == tran_type):
                Tr_velo_to_cam = np.zeros((4, 4))
                Tr_velo_to_cam[:3, :4] = np.array(
                    line.split('\n')[0].split(' ')[1:]).astype(float).reshape(3,
                                                                              4)
                Tr_velo_to_cam[3, 3] = 1

        vel_to_cam = np.dot(R0_rect, Tr_velo_to_cam)
        cam_to_vel = np.linalg.inv(vel_to_cam)
        return vel_to_cam, cam_to_vel


def decode_bbox_3d(dimention, location, rotation_y, P):
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, h, w = dimention[0], dimention[1], dimention[2]
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    corners_3d = corners_3d.transpose(1, 0)
    corners_3d_homo = np.concatenate(
        [corners_3d, np.ones((corners_3d.shape[0], 1), dtype=np.float32)],
        axis=1)
    corners_2d = np.dot(P, corners_3d_homo.transpose(1, 0)).transpose(1, 0)
    corners_2d = corners_2d[:, :2] / corners_2d[:, 2:]

    return corners_2d


def decode_center_2d(dimention, location, K):
    l, h, w = dimention[0], dimention[1], dimention[2]
    x, y, z = location[0], location[1], location[2]

    center_3d = np.array([x, y - h / 2, z])
    proj_point = np.matmul(K, center_3d)
    center_2d = proj_point[:2] / proj_point[2]

    return center_2d


def draw_box_3d(image, corners, color=(0, 255, 0)):
    thickness = 1
    face_idx = [[0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7]]
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            cv2.line(image, (int(corners[f[j], 0]), int(corners[f[j], 1])),
                     (int(corners[f[(j + 1) % 4], 0]),
                      int(corners[f[(j + 1) % 4], 1])), color, thickness,
                     lineType=cv2.LINE_AA)
    if ind_f == 0:
        cv2.line(image, (int(corners[f[0], 0]), int(corners[f[0], 1])),
                 (int(corners[f[2], 0]), int(corners[f[2], 1])), color,
                 thickness, lineType=cv2.LINE_AA)
        cv2.line(image, (int(corners[f[1], 0]), int(corners[f[1], 1])),
                 (int(corners[f[3], 0]), int(corners[f[3], 1])), color,
                 thickness, lineType=cv2.LINE_AA)
    return image


def draw_2d_box_on_image_gt(image, label_file, color=(0, 0, 255)):
    thickness = 1
    with open(label_file) as f:
        for line in f.readlines():
            line_list = line.split('\n')[0].split(' ')
            object_type = line_list[0]
            bbox = np.array(line_list[4:8]).astype(float)
            # opencv image
            height, width, _ = image.shape
            bbox[0] = max(bbox[0], 0)
            bbox[1] = max(bbox[1], 0)
            bbox[2] = min(bbox[2], width)
            bbox[3] = min(bbox[3], height)

            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])),
                          COLOR[CLASS_TO_ID[object_type]], thickness,
                          lineType=cv2.LINE_AA)
            cv2.putText(image, '{}'.format(line_list[0]),
                        (int(bbox[0]), int(bbox[1])),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        COLOR[CLASS_TO_ID[object_type]], thickness,
                        lineType=cv2.LINE_AA)
    return image


def draw_3d_box_on_image_gt(image, label_file, calib_file, camera_type,
                            color=(0, 0, 255)):
    K, P = load_intrinsic_matrix(calib_file, camera_type)
    with open(label_file) as f:
        for line in f.readlines():
            line_list = line.split('\n')[0].split(' ')
            object_type = line_list[0]
            # change the order of dimention as [length, height, width]
            dimention = np.array(
                [line_list[10], line_list[8], line_list[9]]).astype(float)
            location = np.array(line_list[11:14]).astype(float)
            rotation_y = float(line_list[14])
            bbox_3d = decode_bbox_3d(dimention, location, rotation_y, P)
            center = decode_center_2d(dimention, location, K)
            # opencv image
            height, width, _ = image.shape
            if 0 <= int(center[0]) < width and 0 <= int(center[1]) < height:
                image = draw_box_3d(image, bbox_3d,
                                    COLOR[CLASS_TO_ID[object_type]])
    return image


def draw_box_on_bev_image_gt(image, points_filter, calib_path, label_path=None,
                             camera_type='front', color=(0, 0, 255)):
    thickness = 1
    _, cam_to_vel = get_transform_matrix(calib_path, camera_type)
    with open(label_path) as f:
        for line in f.readlines():
            line_list = line.split('\n')[0].split(' ')
            class_name = line_list[0]
            dimensions = np.array(line_list[8:11]).astype(float)
            location = np.array(line_list[11:14]).astype(float)
            rotation_y = float(line_list[14])
            corner_points = get_object_corners_in_lidar(cam_to_vel, dimensions,
                                                        location, rotation_y)
            x_image, y_image = points_filter.pcl2xy_plane(corner_points[:, 0],
                                                          corner_points[:, 1])
            for i in range(len(x_image)):
                cv2.line(image, (x_image[0], y_image[0]),
                         (x_image[1], y_image[1]), color, thickness,
                         lineType=cv2.LINE_AA)
                cv2.line(image, (x_image[0], y_image[0]),
                         (x_image[2], y_image[2]), color, thickness,
                         lineType=cv2.LINE_AA)
                cv2.line(image, (x_image[1], y_image[1]),
                         (x_image[3], y_image[3]), color, thickness,
                         lineType=cv2.LINE_AA)
                cv2.line(image, (x_image[2], y_image[2]),
                         (x_image[3], y_image[3]), color, thickness,
                         lineType=cv2.LINE_AA)
            cv2.putText(image, "{}_{}".format(class_name[:3], int(location[2])),
                        (min(x_image) + 1, min(y_image) + 1),
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, color, thickness)
    return image


def draw_center_on_image_gt(image, label_file, calib_file, camera_type):
    thickness = 1
    K, P = load_intrinsic_matrix(calib_file, camera_type)
    lines = open(label_file).readlines()
    for line in lines:
        line_list = line.split('\n')[0].split(' ')
        object_type = line_list[0]
        # change the order of dimention as [length, height, width]
        dimention = np.array(
            [line_list[10], line_list[8], line_list[9]]).astype(float)
        location = np.array(line_list[11:14]).astype(float)
        center = decode_center_2d(dimention, location, K)
        height, width, _ = image.shape
        if 0 <= int(center[0]) < width and 0 <= int(center[1]) < height:
            cv2.circle(image, (int(center[0]), int(center[1])), 2,
                       COLOR[CLASS_TO_ID[object_type]], thickness,
                       lineType=cv2.LINE_AA)
    return image


def draw_center_on_image(image, annotations, K, color=(0, 255, 0)):
    thickness = 1
    for annotation in annotations:
        dimention = annotation['dimensions']
        location = annotation['locations']
        center = decode_center_2d(dimention, location, K)
        height, width, _ = image.shape
        if 0 <= int(center[0]) < width and 0 <= int(center[1]) < height:
            cv2.circle(image, (int(center[0]), int(center[1])), 2, color,
                       thickness, lineType=cv2.LINE_AA)
    return image


# show the predictions using the changed image and P2
def draw_3d_box_on_image(image, annotations, P, color=(0, 0, 255)):
    for annotation in annotations:
        dimention = annotation['dimensions']
        location = annotation['locations']
        rotation_y = annotation['rot_y']
        bbox_3d = decode_bbox_3d(dimention, location, rotation_y, P)
        image = draw_box_3d(image, bbox_3d, color)
    return image


def draw_center_on_image_aug(image, annotations, P_list, color=(0, 255, 0)):
    thickness = 1
    for idx, annotation in enumerate(annotations):
        dimention = annotation['dimensions']
        location = annotation['locations']
        center = decode_center_2d(dimention, location, P_list[idx][:3, :3])
        height, width, _ = image.shape
        if 0 <= int(center[0]) < width and 0 <= int(center[1]) < height:
            cv2.circle(image, (int(center[0]), int(center[1])), 2, color,
                       thickness, lineType=cv2.LINE_AA)
    return image


# show the predictions using the changed image and P2
def draw_3d_box_on_image_aug(image, annotations, P_list, color=(0, 0, 255)):
    for idx, annotation in enumerate(annotations):
        dimention = annotation['dimensions']
        location = annotation['locations']
        rotation_y = annotation['rot_y']
        center = decode_center_2d(dimention, location, P_list[idx][:3, :3])
        height, width, _ = image.shape
        if 0 <= int(center[0]) < width and 0 <= int(center[1]) < height:
            cv2.circle(image, (int(center[0]), int(center[1])), 2, color,
                       1, lineType=cv2.LINE_AA)
        bbox_3d = decode_bbox_3d(dimention, location, rotation_y, P_list[idx])
        image = draw_box_3d(image, bbox_3d, color)
    return image


def draw_2d_box_on_image(image, annotations, color=(0, 0, 255)):
    thickness = 1
    for annotation in annotations:
        bbox = annotation['bbox_2d']
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])),
                      color, thickness, lineType=cv2.LINE_AA)
    return image
