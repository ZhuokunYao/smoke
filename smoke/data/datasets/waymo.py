import os
import csv
import logging
import random
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

from smoke.modeling.heatmap_coder import (
    get_transfrom_matrix,
    affine_transform,
    gaussian_radius,
    draw_umich_gaussian,
)
from smoke.modeling.smoke_coder import encode_label
from smoke.structures.params_3d import ParamsList

TYPE_ID_CONVERSION = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
    'Truck': 3,
    'Bus': 4,
    'Motobike': 5,
}

CAMERA_TO_ID = {
    'front': 0,
    'front_left': 1,
    'front_right': 2,
    'side_left': 3,
    'side_right': 4,
}

class WAYMODataset(Dataset):
    def __init__(self, cfg, root, split, is_train=True, transforms=None, camera='Front', sample_ratio = 0):
        super(WAYMODataset, self).__init__()
        self.root = root
        self.camera_type = camera
        self.image_dir = os.path.join(root, "image_2", camera)
        self.label_dir = os.path.join(root, "label_2", camera)
        self.calib_dir = os.path.join(root, "calib")

        self.split = split
        self.is_train = is_train
        self.transforms = transforms
        if self.split in ["trainval", "train", "val", "test"]:
            imageset_txt = os.path.join(root, "ImageSets", "{}_{}.txt".format(self.split, camera))
        else:
            raise ValueError("Invalid split!")

        image_files = []
        for line in open(imageset_txt, "r"):
            base_name = line.replace("\n", "")
            image_name = base_name + ".png"
            image_files.append(image_name)
        if sample_ratio != 0:
            image_files = image_files[0:-1:sample_ratio]
        self.image_files = image_files
        self.label_files = [i.replace(".png", ".txt") for i in self.image_files]
        self.num_samples = len(self.image_files)
        self.classes = cfg.DATASETS.DETECT_CLASSES

        self.flip_prob = cfg.INPUT.FLIP_PROB_TRAIN if is_train else 0
        self.aug_prob = cfg.INPUT.SHIFT_SCALE_PROB_TRAIN if is_train else 0
        self.shift_scale = cfg.INPUT.SHIFT_SCALE_TRAIN
        self.num_classes = len(self.classes)

        self.input_width = cfg.INPUT.WIDTH_TRAIN
        self.input_height = cfg.INPUT.HEIGHT_TRAIN
        self.output_width = self.input_width // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_height = self.input_height // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.max_objs = cfg.DATASETS.MAX_OBJECTS

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing WAYMO {} set with {} files loaded".format(self.split, self.num_samples))
        self.debug = False

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # load default parameter here
        original_idx = self.label_files[idx].replace(".txt", "")
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path)
        anns, K = self.load_annotations(idx)

        if self.debug:
            img_src = img.copy()

        center = np.array([i / 2 for i in img.size], dtype=np.float32)
        size = np.array([i for i in img.size], dtype=np.float32)

        """
        Horizontal flip, and affine augmentation are performed here.
        since it is complicated to compute heatmap w.r.t transform.
        """
        flipped = False
        if (self.is_train) and (random.random() < self.flip_prob):
            flipped = True
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            center[0] = size[0] - center[0] - 1
            K[0, 2] = size[0] - K[0, 2] - 1

        affine = False
        if (self.is_train) and (random.random() < self.aug_prob):
            affine = True
            shift, scale = self.shift_scale[0], self.shift_scale[1]
            shift_ranges = np.arange(-shift, shift + 0.1, 0.1)
            center[0] += size[0] * random.choice(shift_ranges)
            center[1] += size[1] * random.choice(shift_ranges)

            scale_ranges = np.arange(1 - scale, 1 + scale + 0.1, 0.1)
            size *= random.choice(scale_ranges)
        center_size = [center, size]
        trans_affine = get_transfrom_matrix(center_size, [self.input_width, self.input_height])
        trans_affine_inv = np.linalg.inv(trans_affine)
        img = img.transform(
            (self.input_width, self.input_height),
            method=Image.AFFINE,
            data=trans_affine_inv.flatten()[:6],
            resample=Image.BILINEAR,
        )

        trans_mat = get_transfrom_matrix(
            center_size,
            [self.output_width, self.output_height]
        )

        if not self.is_train:
            # for inference we parametrize with original size
            target = ParamsList(image_size=size, is_train=self.is_train)
            target.add_field("trans_mat", trans_mat)
            target.add_field("K", K)
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target, original_idx

        heat_map = np.zeros([self.num_classes, self.output_height, self.output_width], dtype=np.float32)
        regression = np.zeros([self.max_objs, 3, 8], dtype=np.float32)
        cls_ids = np.zeros([self.max_objs], dtype=np.int32)
        proj_points = np.zeros([self.max_objs, 2], dtype=np.int32)
        p_offsets = np.zeros([self.max_objs, 2], dtype=np.float32)
        dimensions = np.zeros([self.max_objs, 3], dtype=np.float32)
        locations = np.zeros([self.max_objs, 3], dtype=np.float32)
        rotys = np.zeros([self.max_objs], dtype=np.float32)
        reg_mask = np.zeros([self.max_objs], dtype=np.uint8)
        flip_mask = np.zeros([self.max_objs], dtype=np.uint8)

        if self.debug:
            import cv2
            from utils import compute_box_3d, project_to_image, draw_box_3d
            img_src_cv = cv2.cvtColor(np.asarray(img_src), cv2.COLOR_RGB2BGR)
            img_cv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        for i, a in enumerate(anns):
            a = a.copy()
            cls = a["label"]
            dims = np.array(a["dimensions"])
            locs = np.array(a["locations"])
            rot_y = np.array(a["rot_y"])
            if flipped:
                locs[0] *= -1
                rot_y *= -1

            point, box2d, box3d = encode_label(K, rot_y, dims, locs)
            point_affine = affine_transform(point, trans_mat)
            box2d[:2] = affine_transform(box2d[:2], trans_mat)
            box2d[2:] = affine_transform(box2d[2:], trans_mat)
            box2d[[0, 2]] = box2d[[0, 2]].clip(0, self.output_width - 1)
            box2d[[1, 3]] = box2d[[1, 3]].clip(0, self.output_height - 1)
            h, w = box2d[3] - box2d[1], box2d[2] - box2d[0]

            if self.debug:
                cv2.circle(img_src_cv, (int(point[0]), int(point[1])), 10, (255, 0, 0), -1)
                box2d_gt = [int(float(a["xmin"])), int(float(a["ymin"])), int(float(a["xmax"])), int(float(a["ymax"]))]
                cv2.rectangle(img_src_cv, (int(box2d_gt[0]), int(box2d_gt[1])), (int(box2d_gt[2]), int(box2d_gt[3])),
                              (0, 0, 255), 1)
                cv2.circle(img_cv, (int(point_affine[0]), int(point_affine[1])), 5, (255, 0, 0), -1)

            if (0 < point_affine[0] < self.output_width) and (0 < point_affine[1] < self.output_height):
                point_int = point_affine.astype(np.int32)
                p_offset = point_affine - point_int
                radius = gaussian_radius(h, w)
                radius = max(0, int(radius))
                heat_map[cls] = draw_umich_gaussian(heat_map[cls], point_int, radius)

                cls_ids[i] = cls
                regression[i] = box3d
                proj_points[i] = point_int
                p_offsets[i] = p_offset
                dimensions[i] = dims
                locations[i] = locs
                rotys[i] = rot_y
                reg_mask[i] = 1 if not affine else 0
                flip_mask[i] = 1 if not affine and flipped else 0
        if self.debug:
            cv2.imwrite('demo/' + original_idx + "_src.jpg", img_src_cv)
            cv2.namedWindow('SRC', cv2.WINDOW_NORMAL)
            cv2.imshow('SRC', img_src_cv)
            cv2.imwrite('demo/' + original_idx + "_affine.jpg", img_cv)
            cv2.namedWindow('AFFINE', cv2.WINDOW_NORMAL)
            cv2.imshow('AFFINE', img_cv)
            cv2.waitKey()


        target = ParamsList(image_size=img.size, is_train=self.is_train)
        target.add_field("hm", heat_map)
        target.add_field("reg", regression)
        target.add_field("cls_ids", cls_ids)
        target.add_field("proj_p", proj_points)
        target.add_field("dimensions", dimensions)
        target.add_field("locations", locations)
        target.add_field("rotys", rotys)
        target.add_field("trans_mat", trans_mat)
        target.add_field("K", K)
        target.add_field("reg_mask", reg_mask)
        target.add_field("flip_mask", flip_mask)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, original_idx

    def load_annotations(self, idx):
        annotations = []
        file_name = self.label_files[idx]
        fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                      'dl', 'lx', 'ly', 'lz', 'ry']

        if self.is_train:
            with open(os.path.join(self.label_dir, file_name), 'r') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)

                for line, row in enumerate(reader):
                    if row["type"] in self.classes:
                        annotations.append({
                            "class": row["type"],
                            "label": TYPE_ID_CONVERSION[row["type"]],
                            "truncation": float(row["truncated"]),
                            "occlusion": float(row["occluded"]),
                            "alpha": float(row["alpha"]),
                            "dimensions": [float(row['dl']), float(row['dh']), float(row['dw'])],
                            "locations": [float(row['lx']), float(row['ly']), float(row['lz'])],
                            "rot_y": float(row["ry"]),
                            "xmin": row["xmin"],
                            "ymin": row["ymin"],
                            "xmax": row["xmax"],
                            "ymax": row["ymax"]
                        })

        # get camera intrinsic matrix K
        proj_type = 'P{}:'.format(CAMERA_TO_ID[self.camera_type])
        with open(os.path.join(self.calib_dir, file_name), 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=' ')
            for line, row in enumerate(reader):
                if row[0] == proj_type:
                    P = row[1:]
                    P = [float(i) for i in P]
                    P = np.array(P, dtype=np.float32).reshape(3, 4)
                    K = P[:3, :3]
        return annotations, K
