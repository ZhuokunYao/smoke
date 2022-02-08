import os
import cv2
import argparse
import pprint
import numpy as np
from kitti_utils import *

parser = argparse.ArgumentParser('Description: visualize pred & gt of dataset in KITTI format on bird eye view')
parser.add_argument('--data_root_path', type=str, default="datasets/jdx", help='the root path of dataset')
parser.add_argument('--pred_path', type=str, default="demo/pred", help='the path of pred files')
parser.add_argument('--output_path', type=str, default="demo/bev/", help='output path')
parser.add_argument('--dataset_type', type=str, default="jdx", help='dataset type')
args = parser.parse_args()

dataset_root = os.path.join(args.data_root_path, 'training')
velodyne_path = os.path.join(dataset_root, "velodyne")
label_path = os.path.join(dataset_root, "label_2")
calib_path = os.path.join(dataset_root, "calib")
output_path = args.output_path
pred_path = args.pred_path
dataset_type = args.dataset_type
if not os.path.exists(output_path):
    os.makedirs(output_path)

if __name__ == '__main__':
    for label in os.listdir(pred_path):
        full_label_name = os.path.join(label_path, label.split('.')[0] + ".txt")
        full_pred_name = os.path.join(pred_path, label)
        full_velodyne_name = os.path.join(velodyne_path, label.split('.')[0] + ".bin")
        full_calib_name = os.path.join(calib_path, label.split('.')[0] + ".txt")
        full_bev_image_name = os.path.join(output_path, label.split('.')[0] + ".jpg")
        
        vel_to_cam, cam_to_vel = KittiCalibration.get_transform_matrix(full_calib_name)
        range_list = {"kitti": [(-39.68, 39.68), (0, 69.12), (-2., -2.)],
                      "lyft": [(-100, 100), (-60, 60), (-2., -2.)],
                      "jdx": [(-100, 100), (-60, 60), (-2., -2.)]}
        points_filter = PointCloudFilter(side_range=range_list[dataset_type][0], fwd_range=range_list[dataset_type][1])
        bev_image = points_filter.get_bev_image(full_velodyne_name)

        # Ground Truth
        with open(full_label_name) as f:
            for line in f.readlines():
                line_list = line.split('\n')[0].split(' ')
                object_type = line_list[0]
                occluded = line_list[2]
                dimensions = np.array(line_list[8:11]).astype(float)
                location = np.array(line_list[11:14]).astype(float)
                rotation_y = float(line_list[14])
                corner_points = get_object_corners_in_lidar(cam_to_vel, dimensions, location, rotation_y)
                x_img,y_img = points_filter.pcl2xy_plane(corner_points[:, 0], corner_points[:, 1])
                for i in range(len(x_img)):
                    red = (0, 0, 255)
                    cv2.line(bev_image, (x_img[0], y_img[0]), (x_img[1], y_img[1]), red, 2)
                    cv2.line(bev_image, (x_img[0], y_img[0]), (x_img[2], y_img[2]), red, 2)
                    cv2.line(bev_image, (x_img[1], y_img[1]), (x_img[3], y_img[3]), red, 2)
                    cv2.line(bev_image, (x_img[2], y_img[2]), (x_img[3], y_img[3]), red, 2)
        # Prediction
        with open(full_pred_name) as f:
            for line in f.readlines():
                line_list = line.split('\n')[0].split(' ')
                object_type = line_list[0]
                dimensions = np.array(line_list[8:11]).astype(float)
                location = np.array(line_list[11:14]).astype(float)
                rotation_y = float(line_list[14])
                score = float(line_list[15])
                if score < 0.3:
                    continue
                corner_points = get_object_corners_in_lidar(cam_to_vel, dimensions, location, rotation_y)
                x_img, y_img = points_filter.pcl2xy_plane(corner_points[:, 0], corner_points[:, 1])
                for i in range(len(x_img)):
                    green = (0, 255, 0)
                    cv2.line(bev_image, (x_img[0], y_img[0]), (x_img[1], y_img[1]), green, 2)
                    cv2.line(bev_image, (x_img[0], y_img[0]), (x_img[2], y_img[2]), green, 2)
                    cv2.line(bev_image, (x_img[1], y_img[1]), (x_img[3], y_img[3]), green, 2)
                    cv2.line(bev_image, (x_img[2], y_img[2]), (x_img[3], y_img[3]), green, 2)
        if dataset_type == "lyft":
            bev_image = bev_image[:, :int(0.5*bev_image.shape[1])]
            bev_image = rotate_bound(bev_image, 90)

        cv2.imwrite(full_bev_image_name, bev_image)
        print(full_bev_image_name + " saved ...")
