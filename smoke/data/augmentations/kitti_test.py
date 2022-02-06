# -*- coding:utf8 -*-
import argparse
import os
import shutil
import cv2
import csv
import random
import numpy as np
from PIL import Image, ImageDraw

from smoke.data.kitti_utils import *
from smoke.data.augmentations.augmentations import *


class KittiObject3D(object):
    """
        Class for KITTI 3D object annotations.
    """

    def __init__(self, root_path='', camera_type=None, save_path='./'):
        self.camera_type = None if camera_type is None else camera_type
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.velodyne_folder = os.path.join(root_path, 'training',
                                            'velodyne')
        self.calib_folder = os.path.join(root_path, 'training', 'calib')
        self.label_folder = os.path.join(root_path, 'training', 'label_2')
        self.image_folder = os.path.join(root_path, 'training', 'image_2')
        if self.camera_type is not None:
            self.label_folder = os.path.join(root_path, 'label_2',
                                             self.camera_type)
            self.image_folder = os.path.join(root_path, 'image_2',
                                             self.camera_type)
        image_names = []
        for file in os.listdir(self.label_folder):
            image_names.append(file.split('.')[0])
        self.image_names = image_names

    def augmentation(self, target_size, class_list):
        # augment = RandomHSV()
        # augment = RandomHorizontallyFlip(1.0)
        # augment = RandomAffineTransformation(0.5, [0.2, 0.2], target_size, class_list)
        # augment = RandomCutMixup(0.5, '/media/jd/data/monocular3d/demo/jdx_fusion_cutmixup_640x480/front', target_size)
        augment = RandomMixup(0.5, '/media/jd/data/dataset/jdx_simu_217/front/training/', target_size)
        setup_save_path(self.save_path)
        for idx, image_name in enumerate(self.image_names):
            if idx % 100 != 0:
                continue
            image_path = os.path.join(self.image_folder, image_name + '.jpg')
            if not os.path.exists(image_path):
                continue
            image = Image.open(image_path)
            if image is None:
                continue
            label_path = os.path.join(self.label_folder, image_name + '.txt')
            annotations = load_annotation(label_path)
            calib_path = os.path.join(self.calib_folder,
                                      image_name + '.txt')
            K, P = load_intrinsic_matrix(calib_path, None)
            extrinsic_matrix, _ = load_external_matrix(calib_path)

            image_aug, anno_aug, P_aug = augment(image, annotations, P)

            # Save image
            image_path_save = os.path.join(self.save_path, 'image_2',
                                           image_name + '.jpg')
            image_aug.save(image_path_save)

            img_cv = cv2.cvtColor(np.asarray(image_aug), cv2.COLOR_RGB2BGR)
            # img_cv = draw_2d_box_on_image(img_cv, anno_aug)
            img_cv = draw_3d_box_on_image(img_cv, anno_aug, P_aug)
            # img_cv = draw_3d_box_on_image_aug(img_cv, anno_aug, P_aug)
            # Save image
            image_visual_save = os.path.join(self.save_path, 'visual',
                                             image_name + '.jpg')
            cv2.imwrite(image_visual_save, img_cv)

            cv2.namedWindow('Visual', cv2.WINDOW_KEEPRATIO)
            cv2.imshow('Visual', img_cv)
            cv2.waitKey(0)

            save_calibration_aug(image_name, P_aug,
                             extrinsic_matrix,
                             save_path=os.path.join(self.save_path, 'calib'))

            ## Copy velodyne
            vel_path = os.path.join(self.velodyne_folder,
                                    image_name + '.bin')
            shutil.copy(vel_path,
                        os.path.join(self.save_path, 'velodyne',
                                     image_name + '.bin'))

    def generate_cutmix_dataset(self, target_size, class_list):
        for cls in class_list:
            setup_save_path(os.path.join(self.save_path, cls))
        for idx, image_name in enumerate(self.image_names):
            print(idx)
            image_path = os.path.join(self.image_folder, image_name + '.jpg')
            if not os.path.exists(image_path):
                continue
            image = Image.open(image_path)
            if image is None:
                continue

            label_path = os.path.join(self.label_folder, image_name + '.txt')
            annotations = load_annotation(label_path)
            calib_path = os.path.join(self.calib_folder,
                                      image_name + '.txt')
            K, P = load_intrinsic_matrix(calib_path, None)
            extrinsic_matrix, _ = load_external_matrix(calib_path)
            size = np.array([i for i in image.size], dtype=np.float32)
            center = np.array([i / 2 for i in size], dtype=np.float32)
            trans_affine = get_transfrom_matrix([center, size], target_size)
            trans_affine_inv = np.linalg.inv(trans_affine)
            image = image.transform(
                (int(target_size[0]), int(target_size[1])),
                method=Image.AFFINE,
                data=trans_affine_inv.flatten()[:6],
                resample=Image.BILINEAR)

            # Adjust and save Projection params
            P[:3, :3] = np.matmul(trans_affine, P[:3, :3])
            for idx, annotation in enumerate(annotations):
                if not annotation['class'] in class_list:
                    continue
                anno_save = []
                bbox_2d = np.array(annotation['bbox_2d'])
                bbox_2d[:2] = affine_transform(
                    bbox_2d[:2], trans_affine)
                bbox_2d[2:] = affine_transform(
                    bbox_2d[2:], trans_affine)
                bbox_2d[[0, 2]] = bbox_2d[
                    [0, 2]].clip(0, target_size[0] - 1)
                bbox_2d[[1, 3]] = bbox_2d[
                    [1, 3]].clip(0, target_size[1] - 1)
                annotation['bbox_2d'] = bbox_2d
                if (bbox_2d[2] - bbox_2d[0]) < 32 or (
                        bbox_2d[3] - bbox_2d[1]) < 32:
                    continue
                anno_save.append(annotation)
                image_cv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                # # img_cv = draw_3d_box_on_image(image_cv, anno_save, P)
                # image_cv = draw_2d_box_on_image(image_cv, anno_save)
                # cv2.namedWindow('Visual', cv2.WINDOW_KEEPRATIO)
                # cv2.imshow('Visual', image_cv)
                # cv2.waitKey(0)
                image_crop = image_cv[bbox_2d[1]:bbox_2d[3], bbox_2d[0]:bbox_2d[2]]
                image_path_save = os.path.join(self.save_path,
                                               annotation['class'], 'image_2',
                                               '{}_{}.jpg'.format(image_name,
                                                                  idx))
                cv2.imwrite(image_path_save, image_crop)

                save_calibration('{}_{}'.format(image_name, idx), P,
                                 extrinsic_matrix,
                                 save_path=os.path.join(self.save_path,
                                                        annotation['class'],
                                                        'calib'))
                save_annotations(
                    '{}_{}'.format(image_name, idx), anno_save,
                    save_path=os.path.join(self.save_path, annotation['class'],
                                           'label_2'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset tool.")
    parser.add_argument("--src_path", type=str,
                        default="datasets/jdx_test/front/",
                        help="Path to the src dataset.")
    parser.add_argument("--dst_path", type=str,
                        default='demo/jdx_test/front/',
                        help='Path to save dst images.')
    parser.add_argument("--camera_type", type=str, default=None,
                        help='Specify camera view.')
    parser.add_argument("--class_list", type=str,
                        default=['Pedestrian', 'Cyclist', 'Truck', 'Bus',
                                 'Tricycle'],
                        help='Specify class type list.')
    parser.add_argument("--option", type=int, default=4,
                        help='1 : statistic the number of each class.')
    args = parser.parse_args()

    kitti = KittiObject3D(args.src_path, args.camera_type,
                          args.dst_path)
    kitti.augmentation([720, 540], args.class_list)

    # kitti.generate_cutmix_dataset([640, 480], args.class_list)
