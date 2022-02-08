import argparse
import csv
import torch
from PIL import Image

from smoke.modeling.detector import build_detection_model
from smoke.config import cfg
from smoke.data.transforms import build_transforms
from smoke.engine.inference import generate_kitti_3d_detection
from smoke.utils.model_serialization import load_state_dict
from smoke.modeling.heatmap_coder import get_transfrom_matrix
from smoke.structures.params_3d import ParamsList
import warnings
from tools.kitti_vis.kitti_utils import *

warnings.filterwarnings('ignore')

CLASS_TO_ID = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
    'Truck': 3,
    'Tricycle': 4,
    'Bus': 5,
    'Motobike': 6,
    'Cyclist_stopped': 6
}
ID_TO_CLASS = {
    0: 'Car',
    1: 'Cyclist',
    2: 'Pedestrian',
    3: 'Truck',
    4: 'Tricycle',
    5: 'Bus',
    6: 'Cyclist_stopped'
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
    4: (255, 0, 255),
    5: (255, 255, 0),
    6: (127, 0, 127)
}

range_list = {'kitti': [(-39.68, 39.68), (0, 69.12), (-2., -2.), 0.05],
              'jdx': [(-30, 30), (-40, 40), (-2., -2.), 0.10],
              'waymo720': [(-40, 40), (0, 80), (-2., -2.), 0.1]}


def load_intrinsic_matrix(calib_file, camera_type):
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


def decode_bbox_3d(dimension, location, rotation_y, P):
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, h, w = dimension[0], dimension[1], dimension[2]
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


def decode_center_2d(dimension, location, K):
    l, h, w = dimension[0], dimension[1], dimension[2]
    x, y, z = location[0], location[1], location[2]

    center_3d = np.array([x, y - h / 2, z])
    proj_point = np.matmul(K, center_3d)
    center_2d = proj_point[:2] / proj_point[2]

    return center_2d


def get_transform_matrix(calib_file, camera_type):
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


def draw_3d_box_on_image_gt(image, label_file, calib_file, camera_type,
                            color=(0, 0, 255)):
    K, P = load_intrinsic_matrix(calib_file, camera_type)
    with open(label_file) as f:
        for line in f.readlines():
            line_list = line.split('\n')[0].split(' ')
            object_type = line_list[0]
            # change the order of dimension as [length, height, width]
            dimension = np.array(
                [line_list[10], line_list[8], line_list[9]]).astype(float)
            location = np.array(line_list[11:14]).astype(float)
            rotation_y = float(line_list[14])
            bbox_3d = decode_bbox_3d(dimension, location, rotation_y, P)
            center = decode_center_2d(dimension, location, K)
            # opencv image
            height, width, _ = image.shape
            if 0 <= int(center[0]) < width and 0 <= int(center[1]) < height:
                image = draw_box_3d(image, bbox_3d,
                                    COLOR[CLASS_TO_ID[object_type]])
    return image


def draw_2d_box_on_image_gt(image, label_file, color=(0, 255, 0)):
    thickness = 1
    with open(label_file) as f:
        for line in f.readlines():
            line_list = line.split('\n')[0].split(' ')
            class_name = line_list[0]
            bbox = [float(line_list[4]), float(line_list[5]),
                    float(line_list[6]), float(line_list[7])]
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])),
                          color, thickness, lineType=cv2.LINE_AA)
            cv2.putText(image, "{}".format(class_name[:3]),
                        (int(bbox[0]), int(bbox[1])),
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, color, thickness,
                        lineType=cv2.LINE_AA)
    return image


# show the predictions using the changed image and P2
def draw_3d_box_on_image(img, prediction, P, color=(0, 255, 0)):
    thickness = 1
    image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    for p in prediction:
        p = p.numpy()
        p = p.round(4)
        object_type = int(p[0])
        dimension = [float(p[8]), float(p[6]), float(p[7])]
        location = [float(p[9]), float(p[10]), float(p[11])]
        rotation_y = float(p[12])
        bbox_3d = decode_bbox_3d(dimension, location, rotation_y, P)
        image = draw_box_3d(image, bbox_3d, COLOR[int(p[0])])
        cv2.putText(image, "{}_{}".format(ID_TO_CLASS[object_type][:3],
                                          int(location[2])),
                    (int(min(bbox_3d[:, 0])), int(min(bbox_3d[:, 1]))),
                    cv2.FONT_HERSHEY_COMPLEX, 0.3, color, thickness,
                    lineType=cv2.LINE_AA)
    return image


def draw_2d_box_on_image(img, prediction, color=(0, 0, 255)):
    thickness = 1
    image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    for p in prediction:
        p = p.numpy()
        p = p.round(4)
        bbox = [float(p[2]), float(p[3]), float(p[4]), float(p[5])]
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])),
                      COLOR[int(p[0])], thickness, lineType=cv2.LINE_AA)
    return image


def draw_box_on_bev_image_gt(img, points_filter, label_path, calib_path,
                             camera_type='front', color=(0, 255, 0), ):
    thickness = 1
    _, cam_to_vel = get_transform_matrix(calib_path, camera_type)
    with open(label_path) as f:
        for line in f.readlines():
            line_list = line.split('\n')[0].split(' ')
            class_name = line_list[0]
            dimensions = np.array(line_list[8:11]).astype(float)
            location = np.array(line_list[11:14]).astype(float)
            rotation_y = float(line_list[14])
            if not dimensions.any():
                continue
            corner_points = get_object_corners_in_lidar(cam_to_vel, dimensions,
                                                        location, rotation_y)
            x_img, y_img = points_filter.pcl2xy_plane(corner_points[:, 0],
                                                      corner_points[:, 1])
            for i in range(len(x_img)):
                cv2.line(img, (x_img[0], y_img[0]), (x_img[1], y_img[1]), color,
                         thickness, lineType=cv2.LINE_AA)
                cv2.line(img, (x_img[0], y_img[0]), (x_img[2], y_img[2]), color,
                         thickness, lineType=cv2.LINE_AA)
                cv2.line(img, (x_img[1], y_img[1]), (x_img[3], y_img[3]), color,
                         thickness, lineType=cv2.LINE_AA)
                cv2.line(img, (x_img[2], y_img[2]), (x_img[3], y_img[3]), color,
                         thickness, lineType=cv2.LINE_AA)
            cv2.putText(img, "{}_{}".format(class_name[:3], int(location[2])),
                        (min(x_img) + 1, min(y_img) + 1),
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, color, thickness)
    return img


def draw_box_on_bev_image(img, points_filter, prediction, calib_path,
                          camera_type='front', color=(0, 0, 255)):
    thickness = 1
    _, cam_to_vel = get_transform_matrix(calib_path, camera_type)
    for p in prediction:
        p = p.numpy()
        p = p.round(4)
        dim = np.array([float(p[6]), float(p[7]), float(p[8])])
        location = np.array([float(p[9]), float(p[10]), float(p[11])])
        rotation_y = float(p[12])
        corner_points = get_object_corners_in_lidar(cam_to_vel, dim, location,
                                                    rotation_y)
        x_img, y_img = points_filter.pcl2xy_plane(corner_points[:, 0],
                                                  corner_points[:, 1])
        for i in range(len(x_img)):
            cv2.line(img, (x_img[0], y_img[0]), (x_img[1], y_img[1]), color,
                     thickness, lineType=cv2.LINE_AA)
            cv2.line(img, (x_img[0], y_img[0]), (x_img[2], y_img[2]), color,
                     thickness, lineType=cv2.LINE_AA)
            cv2.line(img, (x_img[1], y_img[1]), (x_img[3], y_img[3]), color,
                     thickness, lineType=cv2.LINE_AA)
            cv2.line(img, (x_img[2], y_img[2]), (x_img[3], y_img[3]), color,
                     thickness, lineType=cv2.LINE_AA)

    return img


@torch.no_grad()
def run_demo(cfg, model_path, validation_dir, output_dir, dataset_type,
             camera_type=None):
    class_to_id = {}
    id_to_class = {}
    for idx, cls in enumerate(cfg.DATASETS.DETECT_CLASSES):
        id_to_class[idx] = cls
        class_to_id[cls] = idx
    print(class_to_id)
    output_image_dir = os.path.join(output_dir, 'image')
    output_pred_dir = os.path.join(output_dir, 'prediction')
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_pred_dir):
        os.makedirs(output_pred_dir)

    val_list_path = os.path.join(validation_dir, 'ImageSets/val.txt')
    labels_dir = os.path.join(validation_dir, 'label_2')
    images_dir = os.path.join(validation_dir, 'image_2')
    calibs_dir = os.path.join(validation_dir, 'calib')
    velodynes_dir = os.path.join(validation_dir, 'velodyne')
    if 'waymo720' in dataset_type:
        val_list_path = os.path.join(validation_dir, 'ImageSets',
                                     'val_{}.txt'.format(camera_type))
        images_dir = os.path.join(validation_dir, 'image_2', camera_type)
        labels_dir = os.path.join(validation_dir, 'label_2', camera_type)

    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)
    if 'backbone_channels' in checkpoint.keys() and len(
            checkpoint['backbone_channels']) != 0:
        cfg.MODEL.BACKBONE.CHANNELS = checkpoint['backbone_channels']
        cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS = \
        checkpoint['backbone_channels'][-1]

    input_width = cfg.INPUT.WIDTH_TEST
    input_height = cfg.INPUT.HEIGHT_TEST
    output_width = input_width // cfg.MODEL.BACKBONE.DOWN_RATIO
    output_height = input_height // cfg.MODEL.BACKBONE.DOWN_RATIO
    device = torch.device(cfg.MODEL.DEVICE)
    model = build_detection_model(cfg)
    model = model.to(device)
    model.eval()

    load_state_dict(model, checkpoint['model'])
    transforms = build_transforms(cfg, is_train=False)
    list_lines = open(val_list_path, 'r').readlines()
    for i, line in enumerate(list_lines):
        print(i, len(list_lines))
        image_name = line.strip()
        image_path = os.path.join(images_dir,
                                  image_name + '.jpg') if os.path.exists(
            os.path.join(images_dir, image_name + '.jpg')) else os.path.join(
            images_dir, image_name + '.png')
        calib_path = os.path.join(calibs_dir, image_name + '.txt')
        label_path = os.path.join(labels_dir, image_name + '.txt')
        velodyne_path = os.path.join(velodynes_dir, image_name + '.bin')

        K, P = load_intrinsic_matrix(calib_path, camera_type)
        P_src = P.copy()
        K_src = K.copy()
        img_src = Image.open(image_path)
        image = img_src.copy()

        if cfg.INPUT.TEST_AFFINE_TRANSFORM:
            center = np.array([i / 2 for i in image.size], dtype=np.float32)
            size = np.array([i for i in image.size], dtype=np.float32)
            center_size = [center, size]
            trans_affine = get_transfrom_matrix(center_size,
                                                [input_width, input_height])
            trans_affine_inv = np.linalg.inv(trans_affine)
            image = image.transform(
                (input_width, input_height),
                method=Image.AFFINE,
                data=trans_affine_inv.flatten()[:6],
                resample=Image.BILINEAR)
        else:
            # Resize the image and change the instric params
            src_width, src_height = image.size
            image = image.resize((input_width, input_height), Image.BICUBIC)
            K[0] = K[0] * input_width / src_width
            K[1] = K[1] * input_height / src_height
            center = np.array([i / 2 for i in image.size], dtype=np.float32)
            size = np.array([i for i in image.size], dtype=np.float32)
            center_size = [center, size]

        trans_mat = get_transfrom_matrix(center_size,
                                         [output_width, output_height])
        # Because resize input image, so need the source image size and intrinsic matrix for 2d bbox decoding
        target = ParamsList(image_size=img_src.size, is_train=False)
        target.add_field('K_src', K_src)
        target.add_field('trans_mat', trans_mat)
        target.add_field('K', K)

        image_input, target = transforms(image, target)
        with torch.no_grad():
            prediction = model(image_input.unsqueeze(0).to(device),
                               [target]).cpu()

        # image_3d = draw_2d_box_on_image(img_src, prediction)
        image_3d = draw_3d_box_on_image(img_src, prediction, P_src)
        image_3d = draw_2d_box_on_image_gt(image_3d, label_path)

        points_filter = PointCloudFilter(side_range=range_list[dataset_type][0],
                                         fwd_range=range_list[dataset_type][1],
                                         res=range_list[dataset_type][-1])
        image_bev = points_filter.get_bev_image(velodyne_path)
        image_bev = draw_box_on_bev_image(image_bev, points_filter, prediction,
                                          calib_path, camera_type)
        image_bev = draw_box_on_bev_image_gt(image_bev, points_filter,
                                             label_path, calib_path,
                                             camera_type)
        # adjust the bev visualization image and save
        bev_size = (int(image_3d.shape[1]), int(
            image_3d.shape[1] / image_bev.shape[1] * image_bev.shape[0]))
        image_bev = cv2.resize(image_bev, bev_size,
                               interpolation=cv2.INTER_AREA)

        # combine the 3d and bev visualization images
        image_composed = np.vstack([image_3d, image_bev])
        cv2.imwrite(
            os.path.join(output_image_dir, image_name + '_composed.jpg'),
            image_composed)
        cv2.namedWindow('COMPOSED', cv2.WINDOW_NORMAL)
        cv2.imshow('COMPOSED', image_composed)
        cv2.waitKey()

        predict_txt = os.path.join(output_pred_dir, image_name + '.txt')
        generate_kitti_3d_detection(prediction, predict_txt, id_to_class)


def main():
    parser = argparse.ArgumentParser(description='SMOKE Demo')
    parser.add_argument('--config_file', type=str, help='Path to config file',
                        default='configs/smoke_jdx_resnet18_640x480.yaml')
    parser.add_argument('--model_path', type=str,
                        help='Path to the trained checkpoint',
                        default='path/to/ur/checkpoint.pth')
    parser.add_argument('--validation_dir', type=str,
                        help='Path to the dataset',
                        default='datasets/jdx_test/front/training/')
    parser.add_argument('--output_dir', default='demo/jdx_test/', type=str,
                        help='Path of saved dir')
    parser.add_argument('--dataset_type', default='jdx', type=str,
                        help='Dataset type.')
    parser.add_argument('--camera_type', default=None, type=str,
                        help='default None for kitti and jdx')
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)

    run_demo(cfg=cfg,
             model_path=args.model_path,
             validation_dir=args.validation_dir,
             output_dir=args.output_dir,
             dataset_type=args.dataset_type,
             camera_type=args.camera_type)


if __name__ == '__main__':
    main()
