import os
import csv
import logging
import random
import numpy as np
import copy
from PIL import Image, ImageDraw

from torch.utils.data import Dataset

from smoke.modeling.heatmap_coder import (
    get_transfrom_matrix,
    affine_transform,
    gaussian_radius,
    draw_umich_gaussian,
    gaussian2D,
)
from smoke.modeling.smoke_coder import encode_label
from smoke.structures.params_3d import ParamsList
from smoke.data.augmentations import build_augmentations
from smoke.data.kitti_utils import *

TYPE_ID_CONVERSION = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
    'Truck': 3,
    'Tricycle': 4,
    'Bus': 5,
    'Cyclist_stopped': 6
}


class JDXDataset(Dataset):
    def __init__(self, cfg, root, split, is_train=True, transforms=None):
        super(JDXDataset, self).__init__()
        self.logger = logging.getLogger(__name__)
        # datasets/......../training
        self.root = root
        self.image_dir = os.path.join(root, "image_2")
        self.label_dir = os.path.join(root, "label_2")
        self.calib_dir = os.path.join(root, "calib")
        # train
        self.split = split
        self.is_train = is_train
        self.transforms = transforms
        if self.split in ["trainval", "train", "val", "test"]:
            self.imageset_txt = os.path.join(root, "ImageSets",
                                             "{}.txt".format(self.split))
        else:
            raise ValueError("Invalid split!")
        self.classes = cfg.DATASETS.DETECT_CLASSES
        # ["Car", "Cyclist", "Pedestrian", "Truck", "Tricycle", "Bus"]
        self.num_classes = len(self.classes)
        self.depth_range = cfg.DATASETS.DEPTH_RANGE
        # [0, 30]
        self.trunction = cfg.DATASETS.TRUNCATION
        # [-1, 0, 1, 2, 3, 4, 5]
        self.occlusion = cfg.DATASETS.OCCLUSION
        # [-1, 0, 1, 2, 3, 4, 5]
        self.cls_shots = np.zeros([self.num_classes], dtype=np.int32)
        # [0, 0, 0, 0, 0, 0]
        self.cls_weights = np.ones([self.num_classes], dtype=np.float32)
        # [1, 1, 1, 1, 1, 1]
        self.weight_gamma = [cfg.MODEL.SMOKE_HEAD.CLS_REWEIGHT_SHOT_GAMMA,
                             cfg.MODEL.SMOKE_HEAD.CLS_REWEIGHT_SIZE_GAMMA]
        # [0.5, 0.5]
        self.dimension_reference = cfg.MODEL.SMOKE_HEAD.DIMENSION_REFERENCE
        # ((4.392, 1.658, 1.910),
        #  (1.773, 1.525, 0.740),
        #  (0.505, 1.644, 0.582),
        #  (7.085, 2.652, 2.523),
        #  (2.790, 1.651, 1.201),
        #  (8.208, 2.869, 2.645))
        self.image_files = []
        self.anno_dicts = []
        self.P_dicts = []
        self.load_annotations()
        """
        after this
        self.image_files:   image pth list
        self.anno_dicts:    list of dict
                            one image per dict
                                 {"class": type_cap,                                  Car
                                 "label": CLASS_TO_ID[type_cap],                      0
                                 "truncation": truncated,                             -1
                                 "occlusion": occluded,                               -1
                                 "alpha": float(row["alpha"]),                        0.65
                                 "bbox_2d": [int(float(row["xmin"])),                 177.85
                                             int(float(row["ymin"])),                 279.78
                                             int(float(row["xmax"])),                 255.61
                                             int(float(row["ymax"]))],                302.81
                                 "dimensions": [float(row['dl']), float(row['dh']),   [4.85, 1.6, 2.63]
                                                float(row['dw'])],
                                 "locations": [float(row['lx']), float(row['ly']),    [-10.64, 1.68, 33.27]  of the top car!!!
                                               float(row['lz'])],
                                 "rot_y": float(row["ry"])}                           0.29
        self.P_dicts:   3x4 matrix kist,  3x3 of which is instrict matrix
        self.cls_shots:   class nums list
        """
        self.num_samples = len(self.image_files)
        if self.is_train:
            self.cls_weights = calculate_class_weight(
                self.cls_shots, self.weight_gamma, self.dimension_reference)
        #self.cls_weights:  cls_weight is depend on 1. cls_shot 2. dimension_reference
        #   cls_shot bigger cls_weight smaller
        #   dimension_reference bigger cls_weight smaller
        self.infer_width = cfg.INPUT.WIDTH_TRAIN   # 640
        self.infer_height = cfg.INPUT.HEIGHT_TRAIN # 480
        self.output_width = self.infer_width // cfg.MODEL.BACKBONE.DOWN_RATIO  #  160
        self.output_height = self.infer_height // cfg.MODEL.BACKBONE.DOWN_RATIO#  120
        self.max_objs = cfg.DATASETS.MAX_OBJECTS   # 100
        #  use for what??
        self.radius_iou = cfg.INPUT.RADIUS_IOU     # 0.5
        """
        input: image, annotation_dict_list, P    output: image, annotation_dict_list, P
        RandomHSV:                  change image
        RandomHorizontallyFlip:     change image & annotation['bbox_2d'] & P
        RandomAffineTransformation: change image & annotation['bbox_2d'] & P
        """
        self.augmentations = build_augmentations(cfg) if self.is_train else None
        # False
        self.debug_visual = cfg.INPUT.DEBUG_VISUAL
        # None
        self.debug_dir = os.path.join(cfg.OUTPUT_DIR,
                                      'debug') if self.debug_visual else None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # load default parameter here
        original_idx = self.image_files[idx][:-4]
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img_src = Image.open(img_path)
        src_width, src_height = img_src.size
        annos_src, P_src = self.anno_dicts[idx], self.P_dicts[idx]
        K_src = P_src[:3, :3]

        if not self.is_train:
            # Resize the image and change the instric params
            img = img_src.resize((self.infer_width, self.infer_height),
                                 Image.BICUBIC)
            K = K_src.copy()
            # resize causing the K changed
            K[0] = K[0] * self.infer_width / src_width
            K[1] = K[1] * self.infer_height / src_height
            size = np.array([i for i in img.size], dtype=np.float32)
            center = np.array([i / 2 for i in size], dtype=np.float32)
            center_size = [center, size]
            # transform from origin_img to the output 120*160 downsampled image
            trans_mat = get_transfrom_matrix(center_size, [self.output_width,
                                                           self.output_height])
            # for inference we parametrize with original size
            target = ParamsList(image_size=[src_width, src_height],
                                is_train=self.is_train)
            target.add_field("K_src", K_src)
            target.add_field("trans_mat", trans_mat)
            target.add_field("K", K)
            #transforms:  input:img & target    output:BGR,255 img & origin target
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target, original_idx
        """
        Test mode:
        return:
                   img:  origin image resized to 480*640, then BGR,255,torch tensor
                   target:  
                       attri:  image_size=[src_width, src_height]  (origin)
                       field:  K_src    origin K  in the annotation file
                       trans_mat:  trans matrix from 480*640 to 120*160
                       K:      changed K because of resize
                   original_idx: image name without .jpg or .png
        """
        # For training
        img = img_src.copy()
        annos = copy.deepcopy(annos_src)
        P = P_src.copy()
        img_aug, annos, P = self.augmentations(img, annos, P)
        K = P[:3, :3]
        size = np.array([i for i in img_aug.size], dtype=np.float32)
        center = np.array([i / 2 for i in size], dtype=np.float32)
        center_size = [center, size]
        # E  matrix
        trans_affine = get_transfrom_matrix(center_size, [self.infer_width,
                                                          self.infer_height])
        trans_affine_inv = np.linalg.inv(trans_affine)
        img = img_aug.transform((self.infer_width, self.infer_height),
                                method=Image.AFFINE,
                                data=trans_affine_inv.flatten()[:6],
                                resample=Image.BILINEAR)
        # down sample matrix from 480*640 to 120*160
        trans_mat = get_transfrom_matrix(center_size, [self.output_width,
                                                       self.output_height])
        #  6, 120, 160
        heat_map = np.zeros(
            [self.num_classes, self.output_height, self.output_width],
            dtype=np.float32)
        #  100, 3, 8
        regression = np.zeros([self.max_objs, 3, 8], dtype=np.float32)
        #  100
        cls_ids = np.zeros([self.max_objs], dtype=np.int32)
        #  100, 2
        proj_points = np.zeros([self.max_objs, 2], dtype=np.int32)
        #  100, 2
        p_offsets = np.zeros([self.max_objs, 2], dtype=np.float32)
        #  100, 3
        dimensions = np.zeros([self.max_objs, 3], dtype=np.float32)
        #  100, 3
        locations = np.zeros([self.max_objs, 3], dtype=np.float32)
        #  100
        rotys = np.zeros([self.max_objs], dtype=np.float32)
        #  100
        reg_mask = np.zeros([self.max_objs], dtype=np.uint8)
        
        target_dict = {}

        for i, a in enumerate(annos):
            if a["class"] not in self.classes:
                continue
            cls = a["label"]
            dims = np.array(a["dimensions"])
            locs = np.array(a["locations"])
            rot_y = np.array(a["rot_y"])
            # proj_point: 3D center projected into 2D
            # box2d: xmin, ymin, xmax, ymax
            # corners_3d: 3*8
            point, box2d, box3d = encode_label(K, rot_y, dims, locs)
            # trans to 120*160
            point = affine_transform(point, trans_mat)
            box2d[:2] = affine_transform(box2d[:2], trans_mat)
            box2d[2:] = affine_transform(box2d[2:], trans_mat)
            box2d[[0, 2]] = box2d[[0, 2]].clip(0, self.output_width - 1)
            box2d[[1, 3]] = box2d[[1, 3]].clip(0, self.output_height - 1)
            h, w = box2d[3] - box2d[1], box2d[2] - box2d[0]

            if (0 < point[0] < self.output_width) and (
                    0 < point[1] < self.output_height):
                point_int = point.astype(np.int32)
                # p_offset:  float diff caused by downsample!!!!!
                p_offset = point - point_int
                radius = gaussian_radius(h, w, thresh_min=self.radius_iou)
                radius = max(0, int(radius))
                # max, not add!!!
                heat_map[cls] = draw_umich_gaussian(heat_map[cls], point_int, radius)
                ###cls_ids[i] = cls
                ###regression[i] = box3d
                ###proj_points[i] = point_int
                ###p_offsets[i] = p_offset
                ###dimensions[i] = dims
                ###locations[i] = locs
                ###rotys[i] = rot_y
                ###reg_mask[i] = 1

                all_point = []
                scores = []
                gauss_square = gaussian2D((2*radius+1,2*radius+1), (2*radius+1)/6)
                for gauss_h in range(2*radius+1):
                    for gauss_w in range(2*radius+1):
                        offset_h = gauss_h - radius
                        offset_w = gauss_w - radius
                        gauss_score = gauss_square[gauss_h, gauss_w]
                        if(gauss_score > 0.1):
                            all_point.append((point_int[0]+offset_w, point_int[1]+offset_h))
                            scores.append(gauss_score)

                #all_point, scores = find_point_in_gaussian(point_int, radius)
                for POINT,score in zip(all_point,scores):
                    if ( POINT[0]<=0 or POINT[0]>=self.output_width or POINT[1]<=0 or POINT[1]>=self.output_height):
                        continue
                    if ( POINT not in target_dict or score>target_dict[POINT]["score"]):
                        point_dict = {}
                        point_dict["cls_ids"] = cls
                        point_dict["regression"] = box3d
                        #point_dict["proj_points"] = (POINT[0],POINT[1])
                        point_dict["dimensions"] = dims
                        point_dict["locations"] = locs
                        point_dict["rotys"] = rot_y
                        point_dict["p_offsets"] = point - np.array(POINT,dtype=np.float32)
                        target_dict[POINT] = point_dict
        print(f"number of regression targets: {target_dict.size()}")
        target_idx = 0
        for POINT,anno_dict in target_dict.items():
            cls_ids[target_idx] = anno_dict["cls_ids"]
            regression[target_idx] = anno_dict["regression"]
            p_offsets[target_idx] = anno_dict["p_offsets"]
            dimensions[target_idx] = anno_dict["dimensions"]
            locations[target_idx] = anno_dict["locations"]
            rotys[target_idx] = anno_dict["rotys"]
            proj_points[target_idx] = np.array(POINT,dtype=np.float32)
            reg_mask[target_idx] = 1
            ###cls_ids[i] = cls
            ###regression[i] = box3d
            ###proj_points[i] = point_int
            ###p_offsets[i] = p_offset
            ###dimensions[i] = dims
            ###locations[i] = locs
            ###rotys[i] = rot_y
            ###reg_mask[i] = 1
        # Debug and Visual
        if self.debug_visual:
            img_cv = cv2.cvtColor(np.asarray(img_aug), cv2.COLOR_RGB2BGR)
            self.visualize(img_cv, annos, P, self.image_files[idx])
        target = ParamsList(image_size=img.size, is_train=self.is_train)
        target.add_field("cls_weights", self.cls_weights)
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
        target.add_field("p_offsets",p_offsets)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, original_idx
        """
        Train mode:
        return:
                   img:  origin image augemented(including resized to 480*640), then BGR,255,torch tensor
                   target:  
                       attri:  image_size=[640, 480]
                       field:  invalid is set to 0!!!!!!!
                       cls_weights (6):    cls_weight is depend on 1. cls_shot 2. dimension_reference
                                       cls_shot bigger cls_weight smaller
                                       dimension_reference bigger cls_weight smaller
                       hm (6, 120, 160): gaussed heatmap
                       reg (100, 3, 8): 3d box
                       cls_ids (100): class id
                       proj_p (100, 2): 2d center in (int) in 120*160 image plane
                       dimensions (100, 3): dims
                       locations (100, 3): 3d locations of the top(!!!!!) car
                       rotys (100): rotys
                       trans_mat: down sample matrix from 480*640 to 120*160 
                       K: changed K because of augmentation
                       reg_mask (100): 0/1
                   original_idx: image name without .jpg or .png
        """
        
    def get_annotations(self):
        return self.anno_dicts

    def load_annotations(self):
        # Filter label file without object
        lines = open(self.imageset_txt, "r").readlines()
        for line in lines:
            # image_2
            image_name = line.strip() + ".png" if os.path.exists(
                os.path.join(self.image_dir,
                             line.strip() + ".png")) else line.strip() + ".jpg"
            # label_2
            label_file = os.path.join(self.label_dir, line.strip() + ".txt")
            """
                    {"class": type_cap,                                  Car
                    "label": CLASS_TO_ID[type_cap],                      0
                    "truncation": truncated,                             -1
                    "occlusion": occluded,                               -1
                    "alpha": float(row["alpha"]),                        0.65
                    "bbox_2d": [int(float(row["xmin"])),                 177.85
                                int(float(row["ymin"])),                 279.78
                                int(float(row["xmax"])),                 255.61
                                int(float(row["ymax"]))],                302.81
                    "dimensions": [float(row['dl']), float(row['dh']),   [4.85, 1.6, 2.63]
                                   float(row['dw'])],
                    "locations": [float(row['lx']), float(row['ly']),    [-10.64, 1.68, 33.27]
                                  float(row['lz'])],
                    "rot_y": float(row["ry"])}                           0.29
            """  
            # list of dict
            annotation = load_annotation(label_file, self.classes,
                                         self.depth_range, self.trunction,
                                         self.occlusion)
            calib_file = os.path.join(self.calib_dir, line.strip() + ".txt")
            # load_intrinsic_matrix: return: K,P   K 3x3  P 3x4
            _, P = load_intrinsic_matrix(calib_file)
            if annotation:
                self.image_files.append(image_name)
                self.anno_dicts.append(annotation)
                self.P_dicts.append(P)

        total_num = len(lines)
        valid_num = len(self.image_files)
        self.logger.info(
            "Initializing JDX [{}] => load: {}, valid: {}, invalid: {}".format(
                self.root, total_num, valid_num, total_num - valid_num))

        # statistcs the number each class
        if self.is_train:
            for anno in self.anno_dicts:
                for a in anno:
                    if a["class"] in self.classes:
                        self.cls_shots[CLASS_TO_ID[a["class"]]] += 1
            self.logger.info("Class_shots: {}".format(self.cls_shots))

    def visualize(self, img, annos, P, img_name):
        img = draw_center_on_image(img, annos, P[:3, :3])
        img = draw_3d_box_on_image(img, annos, P)

        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
        cv2.imwrite(os.path.join(self.debug_dir, img_name), img)
