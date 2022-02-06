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
    'Tricycle': 4,
    'Bus': 5,
    'Motobike': 6
}

CAMERA_TO_ID = {
    'front': 0,
    'front_left': 1,
    'front_right': 2,
    'side_left': 3,
    'side_right': 4,
}


class WAYMO720Dataset(Dataset):
    def __init__(self, cfg, root, split, is_train=True, transforms=None, camera='front', sample_ratio=0,
                 distance_threshold=80):
        super(WAYMO720Dataset, self).__init__()
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
        self.distance_threshold = distance_threshold

        ### filter label file without object
        filered_label_files = []
        filered_image_files = []
        for idx in range(len(self.image_files)):
            anns, K = self.load_annotations(idx)
            if anns:
                filered_image_files.append(self.image_files[idx])
                filered_label_files.append(self.label_files[idx])
            else:
                pass

        total_num = len(self.image_files)
        reduced_num = len(self.image_files) - len(filered_image_files)
        self.image_files = filered_image_files
        self.label_files = filered_label_files
        self.num_samples = len(self.image_files)

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "Initializing WAYMO 720 {} set with {} files loaded,   valid: {},  invalid: {}".format(self.split,
                                                                                                   total_num,
                                                                                                   self.num_samples,
                                                                                                   reduced_num))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # load default parameter here
        original_idx = self.label_files[idx].replace(".txt", "")
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path)
        anns, K= self.load_annotations(idx)
        K_src = K.copy()

        # Resize the image and change the instric params
        src_width, src_height = img.size
        img = img.resize((self.input_width, self.input_height), Image.BICUBIC)
        K[0] = K[0] * self.input_width / src_width
        K[1] = K[1] * self.input_height / src_height
        size = np.array([i for i in img.size], dtype=np.float32)
        center = np.array([i / 2 for i in size], dtype=np.float32)
        center_size = [center, size]

        if not self.is_train:
            trans_mat = get_transfrom_matrix(center_size, [self.output_width, self.output_height])
            # for inference we parametrize with original size
            target = ParamsList(image_size=[src_width, src_height], is_train=self.is_train)
            target.add_field("K_src", K_src)
            target.add_field("trans_mat", trans_mat)
            target.add_field("K", K)
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target, original_idx

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
        if self.is_train:
            trans_affine = get_transfrom_matrix(center_size, [self.input_width, self.input_height])
            trans_affine_inv = np.linalg.inv(trans_affine)
            img = img.transform((self.input_width, self.input_height),
                                method=Image.AFFINE,
                                data=trans_affine_inv.flatten()[:6],
                                resample=Image.BILINEAR)
        trans_mat = get_transfrom_matrix(center_size, [self.output_width, self.output_height])

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

        with open(os.path.join(self.label_dir, file_name), 'r') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)

            for line, row in enumerate(reader):
                if row["type"] in self.classes and float(row['lz']) < self.distance_threshold:
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
