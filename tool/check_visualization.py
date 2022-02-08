import argparse
import csv
import cv2
from PIL import Image

from tools.kitti_vis.kitti_utils import *

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

def load_intrinsic_matrix(calib_file, camera_type):
    proj_type = 'P2:' if camera_type is None else 'P{}:'.format(CAMERA_TO_ID[camera_type])
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

def get_transform_matrix(calib_file, camera_type):
    tran_type = 'Tr_velo_to_cam:' if camera_type is None else 'Tr_velo_to_cam_{}:'.format(CAMERA_TO_ID[camera_type])
    with open(calib_file, 'r') as f:
        for line in f.readlines():
            if (line.split(' ')[0] == 'R0_rect:'):
                R0_rect = np.zeros((4, 4))
                R0_rect[:3, :3] = np.array(line.split('\n')[0].split(' ')[1:]).astype(float).reshape(3,3)
                R0_rect[3, 3] = 1
            if (line.split(' ')[0] == tran_type):
                Tr_velo_to_cam = np.zeros((4, 4))
                Tr_velo_to_cam[:3, :4] = np.array(line.split('\n')[0].split(' ')[1:]).astype(float).reshape(3,4)
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
        [corners_3d, np.ones((corners_3d.shape[0], 1), dtype=np.float32)], axis=1)
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
    face_idx = [[0,1,5,4],
              [1,2,6, 5],
              [2,3,7,6],
              [3,0,4,7]]
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            cv2.line(image, (int(corners[f[j], 0]), int(corners[f[j], 1])),
                   (int(corners[f[(j+1)%4], 0]), int(corners[f[(j+1)%4], 1])), color, thickness, lineType=cv2.LINE_AA)
    if ind_f == 0:
        cv2.line(image, (int(corners[f[0], 0]), int(corners[f[0], 1])),
               (int(corners[f[2], 0]), int(corners[f[2], 1])), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(image, (int(corners[f[1], 0]), int(corners[f[1], 1])),
               (int(corners[f[3], 0]), int(corners[f[3], 1])), color, thickness, lineType=cv2.LINE_AA)
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

            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                          COLOR[CLASS_TO_ID[object_type]], thickness, lineType=cv2.LINE_AA)
            cv2.putText(image, '{}'.format(line_list[0]), (int(bbox[0]), int(bbox[1])),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, COLOR[CLASS_TO_ID[object_type]], thickness, lineType=cv2.LINE_AA)
    return image

def draw_3d_box_on_image_gt(image, label_file, calib_file, camera_type, color=(0, 0, 255)):
    K, P = load_intrinsic_matrix(calib_file, camera_type)
    with open(label_file) as f:
        for line in f.readlines():
            line_list = line.split('\n')[0].split(' ')
            object_type = line_list[0]
            # change the order of dimention as [length, height, width]
            dimention = np.array([line_list[10], line_list[8], line_list[9]]).astype(float)
            location = np.array(line_list[11:14]).astype(float)
            rotation_y = float(line_list[14])
            bbox_3d = decode_bbox_3d(dimention, location, rotation_y, P)
            center = decode_center_2d(dimention, location, K)
            # opencv image
            height, width, _ = image.shape
            if 0 <= int(center[0]) < width and 0 <= int(center[1]) < height:
                image = draw_box_3d(image, bbox_3d, COLOR[CLASS_TO_ID[object_type]])
    return image

def draw_box_on_bev_image_gt(image, points_filter, calib_path, label_path=None, camera_type = 'front', color=(0, 0, 255)):
    thickness = 1
    _, cam_to_vel = get_transform_matrix(calib_path, camera_type)
    with open(label_path) as f:
        for line in f.readlines():
            line_list = line.split('\n')[0].split(' ')
            class_name = line_list[0]
            dimensions = np.array(line_list[8:11]).astype(float)
            location = np.array(line_list[11:14]).astype(float)
            rotation_y = float(line_list[14])
            corner_points = get_object_corners_in_lidar(cam_to_vel, dimensions, location, rotation_y)
            x_image, y_image = points_filter.pcl2xy_plane(corner_points[:, 0], corner_points[:, 1])
            for i in range(len(x_image)):
                cv2.line(image, (x_image[0], y_image[0]), (x_image[1], y_image[1]), color, thickness, lineType=cv2.LINE_AA)
                cv2.line(image, (x_image[0], y_image[0]), (x_image[2], y_image[2]), color, thickness, lineType=cv2.LINE_AA)
                cv2.line(image, (x_image[1], y_image[1]), (x_image[3], y_image[3]), color, thickness, lineType=cv2.LINE_AA)
                cv2.line(image, (x_image[2], y_image[2]), (x_image[3], y_image[3]), color, thickness, lineType=cv2.LINE_AA)
            cv2.putText(image, "{}_{}".format(class_name[:3], int(location[2])), (min(x_image) + 1, min(y_image) + 1),
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
        dimention = np.array([line_list[10], line_list[8], line_list[9]]).astype(float)
        location = np.array(line_list[11:14]).astype(float)
        center = decode_center_2d(dimention, location, K)
        height, width, _ = image.shape
        if 0 <= int(center[0]) < width and 0 <= int(center[1]) < height:
            cv2.circle(image, (int(center[0]), int(center[1])), 2, COLOR[CLASS_TO_ID[object_type]], thickness, lineType=cv2.LINE_AA)
    return image

range_list = {'kitti': [(-39.68, 39.68), (0, 69.12), (-2., -2.), 0.05],
            'jdx': [(-40, 40), (-60, 60), (-2., -2.), 0.10],
            'waymo': [(-60, 60), (-0, 80), (-2., -2.), 0.1]}


def run_visualization(root_dir, output_dir, dataset_type, camera_type=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if dataset_type == 'waymo':
        labels_dir = os.path.join(root_dir, 'label_2', camera_type)
        images_dir = os.path.join(root_dir, 'image_2', camera_type)
    else:
        camera_type = None
        labels_dir = os.path.join(root_dir, 'label_2')
        images_dir = os.path.join(root_dir, 'image_2')
    calibs_dir = os.path.join(root_dir, 'calib')
    velodynes_dir = os.path.join(root_dir, 'velodyne')

    for i, label in enumerate(os.listdir(labels_dir)):
        image_name = os.path.splitext(label)[0]
        print(image_name)
        image_path = os.path.join(images_dir, image_name + '.jpg') if os.path.exists(
            os.path.join(images_dir, image_name + '.jpg')) else os.path.join(images_dir, image_name + '.png')
        calib_path = os.path.join(calibs_dir, image_name+'.txt')
        label_path = os.path.join(labels_dir, image_name+'.txt')
        velodyne_path = os.path.join(velodynes_dir, image_name + '.bin')
        img = Image.open(image_path)

        img_cv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        image_3d = draw_center_on_image_gt(img_cv, label_path, calib_path, camera_type)
        # image_3d = draw_2d_box_on_image_gt(image_3d, label_path)
        image_3d = draw_3d_box_on_image_gt(img_cv, label_path, calib_path, camera_type)

        points_filter = PointCloudFilter(side_range=range_list[dataset_type][0], fwd_range=range_list[dataset_type][1], res=range_list[dataset_type][-1])
        image_bev = points_filter.get_bev_image(velodyne_path)
        image_bev = draw_box_on_bev_image_gt(image_bev, points_filter, calib_path, label_path, camera_type, color=(0, 0, 255))

        # adjust the bev visualization image and save
        bev_size = (int(image_3d.shape[1]), int(image_3d.shape[1]/image_bev.shape[1]*image_bev.shape[0]))
        image_bev = cv2.resize(image_bev, bev_size, interpolation=cv2.INTER_AREA)

        # combine the 3d and bev visualization images
        image_composed = np.vstack([image_3d, image_bev])
        cv2.imwrite(os.path.join(output_dir, image_name+'_composed.jpg'), image_composed)
        cv2.namedWindow('COMPOSED', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('COMPOSED', image_composed)
        cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser(description='SMOKE visualization')
    parser.add_argument('--root_dir', type=str,
                        help='Specify a root dir of dataset',
                        default='datasets/jdx_test/front/training/')
    parser.add_argument('--output_dir', default='demo/jdx_test_vis/front/', type=str,
                        help='Specify a dir to save')
    parser.add_argument('--dataset_type', default='jdx', type=str,
                        help='Specify dataset type')
    parser.add_argument('--camera_type', default=None, type=str,
                        help='Specify camera type')

    args = parser.parse_args()

    run_visualization(root_dir=args.root_dir, output_dir=args.output_dir,
                      dataset_type=args.dataset_type,
                      camera_type=args.camera_type)

if __name__ == '__main__':
    main()
