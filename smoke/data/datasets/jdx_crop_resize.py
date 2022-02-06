import os
import csv
import logging
import random
import numpy as np
from PIL import Image
import cv2

from torch.utils.data import Dataset

from smoke.modeling.heatmap_coder import (
    get_transfrom_matrix,
    affine_transform,
    gaussian_radius,
    draw_umich_gaussian,
)
from smoke.modeling.heatmap_coder_oval import (
    gaussian_radius_oval,
    draw_umich_gaussian_oval,
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
    'Cyclist_stopped': 6
}


class JDXDataset(Dataset):
    def __init__(self, cfg, root, split, is_train=True, transforms=None, camera='front', crop_size=[8, 156, 8, 0]):
        super(JDXDataset, self).__init__()
        self.root = root
        self.image_dir = os.path.join(root, "image_2")
        self.label_dir = os.path.join(root, "label_2")
        self.calib_dir = os.path.join(root, "calib")

        self.camera = camera
        self.split = split
        self.is_train = is_train
        self.transforms = transforms
        if self.split in ["trainval", "train", "val", "test"]:
            self.imageset_txt = os.path.join(root, "ImageSets",
                                             "{}.txt".format(self.split))
        else:
            raise ValueError("Invalid split!")
        self.classes = cfg.DATASETS.DETECT_CLASSES
        self.num_classes = len(self.classes)
        self.image_files = []
        self.label_files = []
        self.anno_dicts = []
        self.K_dicts = []
        self.cls_shots = np.zeros([self.num_classes], dtype=np.int32)
        self.cls_weights = np.ones([self.num_classes], dtype=np.float32)
        total_num = 0
        # Filter label file without object
        for line in open(self.imageset_txt, "r"):
            total_num += 1
            image_name = line.strip() + ".png" if os.path.exists(
                os.path.join(self.image_dir, line.strip() + ".png")) else line.strip() + ".jpg"
            label_file = line.strip() + ".txt"
            anns, K = self.load_annotations(label_file)
            if anns:
                self.image_files.append(image_name)
                self.label_files.append(label_file)
                self.anno_dicts.append(anns)
                self.K_dicts.append(K)
                ### statistic object class num
                for i, a in enumerate(anns):
                    cls = a["label"]
                    self.cls_shots[cls] += 1

        cls_ratios = self.cls_shots / np.sum(self.cls_shots)
        for index, item in enumerate(self.cls_shots):
            if item != 0:
                self.cls_weights[index] = pow(1 / cls_ratios[index], cfg.MODEL.SMOKE_HEAD.CLS_REWEIGHT_SHOT_GAMMA)
        for index, item in enumerate(self.cls_shots):
            if item == 0:
                self.cls_weights[index] = np.max(self.cls_weights)
        self.cls_weights /= self.cls_weights[0]

        size_weight = [sum(x) for x in cfg.MODEL.SMOKE_HEAD.DIMENSION_REFERENCE]
        size_weight = [np.power(size_weight[0] / x, cfg.MODEL.SMOKE_HEAD.CLS_REWEIGHT_SIZE_GAMMA) for x in size_weight]
        size_weight = np.array(size_weight, dtype="float")

        self.cls_weights *= size_weight
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "This datasets class_shots: {},               clas_weights: {}".format(self.cls_shots, self.cls_weights))
        #############################################################

        self.num_samples = len(self.image_files)
        self.infer_width = cfg.INPUT.WIDTH_TRAIN
        self.infer_height = cfg.INPUT.HEIGHT_TRAIN
        self.output_width = self.infer_width // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_height = self.infer_height // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.max_objs = cfg.DATASETS.MAX_OBJECTS
        self.flip_prob = cfg.INPUT.FLIP_PROB_TRAIN if is_train else 0
        self.aug_prob = cfg.INPUT.SHIFT_SCALE_PROB_TRAIN if is_train else 0
        self.shift_scale = cfg.INPUT.SHIFT_SCALE_TRAIN

        # The croped pixels in [left, up, right, below]
        self.crop_size = crop_size
        self.radius_iou = cfg.INPUT.RADIUS_IOU
        self.heatmap_methord = cfg.INPUT.HEATMAP_METHORD
        self.gamma_correction = cfg.DATASETS.JDX_GAMMA_CORRECTION

        invalid_num = total_num - self.num_samples
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "Initializing JDX {} set with {} files loaded,   valid: {},  invalid: {}".format(self.split,
                                                                                             total_num,
                                                                                             self.num_samples,
                                                                                             invalid_num))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # load default parameter here
        original_idx = self.label_files[idx].replace(".txt", "")
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path)
        anns, K_src = self.anno_dicts[idx], self.K_dicts[idx]
        K = K_src.copy()

        ### gamma correction,default FLIR gamma=0.8, expect is 0.45
        if self.gamma_correction:
            gamma = 0.45 / 0.8
            img = np.asarray(img).astype(np.float32) / 255
            img = np.power(img, gamma)
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)

        if not self.is_train:
            # Resize the image and change the instric params
            src_width, src_height = img.size
            img = img.resize((self.infer_width, self.infer_height), Image.BICUBIC)
            K[0] = K[0] * self.infer_width / src_width
            K[1] = K[1] * self.infer_height / src_height
            size = np.array([i for i in img.size], dtype=np.float32)
            center = np.array([i / 2 for i in size], dtype=np.float32)
            center_size = [center, size]

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
                crop height randomly, resize, horizontal flip, and affine augmentation are performed here.
                since it is complicated to compute heatmap w.r.t transform.
                """
        img_width, img_height = img.size
        flipped = False
        if (random.random() < self.flip_prob):
            flipped = True
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            K[0, 0] = -K[0, 0]
            K[0, 2] = (img_width - 1) - K[0, 2]



        img_mask = np.ones([img_height, img_width], dtype=np.float32)
        img_mask = Image.fromarray(img_mask)
        size = np.array([i for i in img.size], dtype=np.float32)
        center = np.array([i / 2 for i in size], dtype=np.float32)
        zoom_affine = False

        if random.random() > self.aug_prob:
            resize_ratio = 1.0
            size = np.array([self.infer_width, self.infer_height]) * resize_ratio
            x0 = random.choice(np.arange(0.0, img.size[0] - size[0], 1.0))
            if (img.size[1] - size[1]) > 1.0:
                y0 = random.choice(np.arange(0.0, img.size[1] - size[1], 1.0))
            else:
                y0 = 0.0
            center = np.array([x0 + size[0] / 2, y0 + size[1] / 2])
        else:
            zoom_affine = True
            shift, scale = self.shift_scale[0], self.shift_scale[1]
            max_ratio = min((img.size[0] - 1) / self.infer_width, (img.size[1] - 1) / self.infer_height)
            min_ratio = 1 - scale
            resize_ratio = random.choice(np.arange(min_ratio, max_ratio, 0.001))
            size = np.array([self.infer_width, self.infer_height]) * resize_ratio
            x0 = random.choice(np.arange(0.0, img.size[0] - size[0], 1.0))
            y0 = random.choice(np.arange(0.0, img.size[1] - size[1], 1.0))
            center = np.array([x0 + size[0] / 2, y0 + size[1] / 2])


        center_size = [center, size]
        trans_affine = get_transfrom_matrix(center_size, [self.infer_width, self.infer_height])
        trans_affine_inv = np.linalg.inv(trans_affine)
        img = img.transform((self.infer_width, self.infer_height),
                            method=Image.AFFINE,
                            data=trans_affine_inv.flatten()[:6],
                            resample=Image.BILINEAR)
        ### lyb: add, using img as new image with modified K ################################
        K = np.matmul(trans_affine, K)
        size = np.array([i for i in img.size], dtype=np.float32)
        center = np.array([i / 2 for i in size], dtype=np.float32)
        center_size = [center, size]
        trans_mat = get_transfrom_matrix(center_size, [self.output_width, self.output_height])

        ### lyb:add for heat_map_mask ########################################################
        img_mask = img_mask.transform((self.infer_width, self.infer_height),
                                      method=Image.AFFINE,
                                      data=trans_affine_inv.flatten()[:6],
                                      resample=Image.BILINEAR)
        trans_mat_inv = np.linalg.inv(trans_mat)
        img_mask = img_mask.transform((self.output_width, self.output_height),
                                      method=Image.AFFINE,
                                      data=trans_mat_inv.flatten()[:6],
                                      resample=Image.BILINEAR)
        img_mask = np.array(img_mask)
        heat_map_mask = np.expand_dims(img_mask, axis=0)
        heat_map_mask = np.repeat(heat_map_mask, self.num_classes, axis=0)
        #######################################################################################

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

        img_save = np.asarray(img)
        img_save = cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR)
        for i, a in enumerate(anns):
            a = a.copy()
            cls = a["label"]
            dims = np.array(a["dimensions"])
            locs = np.array(a["locations"])
            rot_y = np.array(a["rot_y"])

            point, box2d, box3d = encode_label(K, rot_y, dims, locs)

            ### lyb:added ###############################################################
            box2d_temp = box2d.copy()
            box2d_temp[[0, 2]] = box2d_temp[[0, 2]].clip(0, self.infer_width - 1)
            box2d_temp[[1, 3]] = box2d_temp[[1, 3]].clip(0, self.infer_height - 1)
            cv2.rectangle(img_save, (int(box2d_temp[0]), int(box2d_temp[1])), (int(box2d_temp[2]), int(box2d_temp[3])), color=(0, 0, 255),
                          thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(img_save, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)
            #############################################################################

            point_affine = affine_transform(point, trans_mat)
            box2d[:2] = affine_transform(box2d[:2], trans_mat)
            box2d[2:] = affine_transform(box2d[2:], trans_mat)
            box2d[[0, 2]] = box2d[[0, 2]].clip(0, self.output_width - 1)
            box2d[[1, 3]] = box2d[[1, 3]].clip(0, self.output_height - 1)
            h, w = box2d[3] - box2d[1], box2d[2] - box2d[0]

            if w <= 1.5 or h <= 1.5:
                continue

            if (0 < point_affine[0] < self.output_width) and (0 < point_affine[1] < self.output_height):
                point_int = point_affine.astype(np.int32)
                p_offset = point_affine - point_int

                if self.heatmap_methord == "round":
                    radius = gaussian_radius(h, w, thresh_min=self.radius_iou)
                    radius = max(0, int(radius))
                    heat_map[cls] = draw_umich_gaussian(heat_map[cls], point_int, radius)
                elif self.heatmap_methord == "oval":
                    radius = gaussian_radius_oval(h, w, thresh_min=self.radius_iou)
                    radius = (max(0, int(radius[0])), max(0, int(radius[1])))
                    heat_map[cls] = draw_umich_gaussian_oval(heat_map[cls], point_int, radius)
                else:
                    print("heatmap error")

                cls_ids[i] = cls
                regression[i] = box3d
                proj_points[i] = point_int
                p_offsets[i] = p_offset
                dimensions[i] = dims
                locations[i] = locs
                rotys[i] = rot_y


                reg_mask[i] = 1
                flip_mask[i] = 1 if flipped else 0
                ####################################################

        heat_map = heat_map_mask * heat_map
        # ### lyb: save img ##################################################################
        # heat_map_temp = np.sum(heat_map, axis=0) * 255
        # heat_map_temp=Image.fromarray(heat_map_temp)
        # heat_map_temp = heat_map_temp.transform((self.infer_width, self.infer_height),
        #                     method=Image.AFFINE,
        #                     data=trans_mat.flatten()[:6],
        #                     resample=Image.BILINEAR)
        # heat_map_temp = np.array(heat_map_temp)
        # heat_map_temp = np.expand_dims(heat_map_temp, axis=2)
        # heat_map_temp = np.repeat(heat_map_temp, 3, axis=2)
        # img_save = np.hstack([img_save, heat_map_temp])
        #
        # save_dir = "/media/york/H/jdx_img_temp/img_debug43"
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # if z_bad_flag:
        #     cv2.imwrite(os.path.join(save_dir, self.image_files[idx]), img_save)
        # ####################################################################################

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
        target.add_field("flip_mask", flip_mask)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, original_idx

    def get_annotations(self):
        return self.anno_dicts

    def load_annotations(self, file_name):
        annotations = []
        fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                      'dl', 'lx', 'ly', 'lz', 'ry']

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
                        "rot_y": float(row["ry"])
                    })

        # get camera intrinsic matrix K
        with open(os.path.join(self.calib_dir, file_name), 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=' ')
            for line, row in enumerate(reader):
                if row[0] == 'P2:':
                    P = row[1:]
                    P = [float(i) for i in P]
                    P = np.array(P, dtype=np.float32).reshape(3, 4)
                    K = P[:3, :3]
                    break

        return annotations, K
