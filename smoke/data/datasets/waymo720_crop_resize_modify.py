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
    gaussian2D,
)
from smoke.modeling.heatmap_coder_oval import (
    gaussian_radius_oval,
    draw_umich_gaussian_oval,
)

from smoke.modeling.smoke_coder import encode_label,encode_label_iou_box
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
    def __init__(self, cfg, root, split, is_train=True, transforms=None, camera='Front', sample_ratio=0,
                 crop_size=[0, 0, 0, 0], distance_threshold=60):
        super(WAYMO720Dataset, self).__init__()
        print('\n\n\n using fixed dataset code !!! \n\n\n')
        self.root = root
        self.image_dir = os.path.join(root, "image_2", camera)
        self.label_dir = os.path.join(root, "label_2", camera)
        self.calib_dir = os.path.join(root, "calib")

        self.distance_threshold = distance_threshold
        self.camera = camera
        self.split = split
        self.is_train = is_train
        self.transforms = transforms
        if self.split in ["trainval", "train", "val", "test"] and self.camera in ["front", "rear", "left", "right"]:
            imageset_txt = os.path.join(root, "ImageSets", "{}_{}.txt".format(self.split, camera))
        else:
            raise ValueError("Invalid split!")
        #imageset_txt = os.path.join(root, "ImageSets", "verify.txt")
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
        for line in open(imageset_txt, "r"):
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

        if sample_ratio != 0:
            self.image_files = self.image_files[0:-1:sample_ratio]
            self.label_files = self.label_files[0:-1:sample_ratio]
            self.anno_dicts = self.anno_dicts[0:-1:sample_ratio]
            self.K_dicts = self.K_dicts[0:-1:sample_ratio]

        self.num_samples = len(self.image_files)
        if is_train:
            self.infer_width = cfg.INPUT.WIDTH_TRAIN
            self.infer_height = cfg.INPUT.HEIGHT_TRAIN
            self.output_width = self.infer_width // cfg.MODEL.BACKBONE.DOWN_RATIO
            self.output_height = self.infer_height // cfg.MODEL.BACKBONE.DOWN_RATIO
        else:
            self.infer_width = cfg.INPUT.WIDTH_TEST
            self.infer_height = cfg.INPUT.HEIGHT_TEST
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

        invalid_num = total_num - self.num_samples
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "Initializing WAYMO 720 {} set with {} files loaded,   valid: {},  invalid: {}".format(self.split,
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
        #img = np.array(img)[:,:,::-1]
        #img = Image.fromarray(img)
        
        #print(np.max(np.array(img)))

        anns, K_src = self.anno_dicts[idx], self.K_dicts[idx]
        K = K_src.copy()
        flipped = False

        if not self.is_train:
            # Resize the image and change the instric params
            src_width, src_height = img.size
            img = img.resize((self.infer_width, self.infer_height), Image.BICUBIC)
            K[0] = K[0] * self.infer_width / src_width
            K[1] = K[1] * self.infer_height / src_height
            size = np.array([i for i in img.size], dtype=np.float32)
            center = np.array([i / 2 for i in size], dtype=np.float32)
            center_size = [center, size]
            ### infer img to output heatmap
            trans_mat = get_transfrom_matrix(center_size, [self.output_width, self.output_height])

            # # for inference we parametrize with original size
            # target = ParamsList(image_size=[src_width, src_height], is_train=self.is_train)
            # target.add_field("K_src", K_src)
            # target.add_field("trans_mat", trans_mat)
            # target.add_field("K", K)
            # if self.transforms is not None:
            #     img, target = self.transforms(img, target)
            #
            # return img, target, original_idx

        """
                crop height randomly, resize, horizontal flip, and affine augmentation are performed here.
                since it is complicated to compute heatmap w.r.t transform.
                """
        if self.is_train:
            img_width, img_height = img.size
            if (random.random() < self.flip_prob):
                flipped = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                K[0, 0] = -K[0, 0]
                K[0, 2] = (img_width - 1) - K[0, 2]

            img_mask = np.ones([img_height, img_width], dtype=np.float32)
            img_mask = Image.fromarray(img_mask)
            size = np.array([i for i in img.size], dtype=np.float32)
            center = np.array([i / 2 for i in size], dtype=np.float32)

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
                shift, scale = self.shift_scale[0], self.shift_scale[1]

                ### 1. statistic sparse class(truck,bus.cyclist) box info
                box_long_tail = [img_width - 1, img_height - 1, 0, 0]
                box_all = [img_width - 1, img_height - 1, 0, 0]
                long_tail_cls = False
                for i, a in enumerate(anns):
                    dims = np.array(a["dimensions"])
                    locs = np.array(a["locations"])
                    rot_y = np.array(a["rot_y"])
                    point, box2d, box3d, _ = encode_label(K, rot_y, dims, locs)
                    box2d[[0, 2]] = box2d[[0, 2]].clip(0, img_width - 1)
                    box2d[[1, 3]] = box2d[[1, 3]].clip(0, img_height - 1)
                    if a["class"] == "truck" or a["class"] == "bus" or a["class"] == "Cyclist":
                        long_tail_cls = True
                        box_long_tail[0] = min(box_long_tail[0], box2d[0])
                        box_long_tail[1] = min(box_long_tail[1], box2d[1])
                        box_long_tail[2] = max(box_long_tail[2], box2d[2])
                        box_long_tail[3] = max(box_long_tail[3], box2d[3])
                    box_all[0] = min(box_all[0], box2d[0])
                    box_all[1] = min(box_all[1], box2d[1])
                    box_all[2] = max(box_all[2], box2d[2])
                    box_all[3] = max(box_all[3], box2d[3])

                    if box2d[0] > box2d[2] or box2d[1] > box2d[3]:
                        print("box2d:", box2d)
                if long_tail_cls:
                    xmin, ymin, xmax, ymax = box_long_tail[0], box_long_tail[1], box_long_tail[2], box_long_tail[3]
                else:
                    xmin, ymin, xmax, ymax = box_all[0], box_all[1], box_all[2], box_all[3]

                ### 2.compute crop roi by (xmin, ymin, xmax, ymax)
                if int(ymax - ymin + 1) >= img_height:
                    size = np.array([i for i in img.size], dtype=np.float32)
                    center = np.array([i / 2 for i in size], dtype=np.float32)
                else:
                    min_ratio = (ymax - ymin + 1) / self.infer_height
                    min_ratio = max(min_ratio, 1 - scale)
                    max_ratio = min((img.size[0] - 1) / self.infer_width, (img.size[1] - 1) / self.infer_height)
                    if min_ratio >= max_ratio:
                        resize_ratio = max_ratio
                    else:
                        resize_ratio = random.choice(np.arange(min_ratio, max_ratio, 0.001))
                    size = np.array([self.infer_width, self.infer_height]) * resize_ratio

                    if size[0] <= (xmax - xmin + 1):
                        x0 = random.choice(np.arange(0.0, img.size[0] - size[0], 1.0))
                    else:
                        min_x0 = max(0, xmax - size[0])
                        max_x0 = min(xmin, img_width - size[0])
                        x0 = random.choice(np.arange(min_x0, max_x0 + 1.0, 1.0))
                    min_y0 = max(0, ymax - size[1])
                    max_y0 = min(ymin, img_height - size[1])
                    y0 = random.choice(np.arange(min_y0, max_y0 + 1.0, 1.0))
                    center = np.array([x0 + size[0] / 2, y0 + size[1] / 2])

            center_size = [center, size]
            trans_affine = get_transfrom_matrix(center_size, [self.infer_width, self.infer_height])
            trans_affine_inv = np.linalg.inv(trans_affine)
            ### update img and K, center_size
            img = img.transform((self.infer_width, self.infer_height),
                                method=Image.AFFINE,
                                data=trans_affine_inv.flatten()[:6],
                                resample=Image.BILINEAR)
            K = np.matmul(trans_affine, K)

            size = np.array([i for i in img.size], dtype=np.float32)
            center = np.array([i / 2 for i in size], dtype=np.float32)
            center_size = [center, size]
            ### infer img to output heatmap
            trans_mat = get_transfrom_matrix(center_size, [self.output_width, self.output_height])

            ### lyb:add for heat_map_mask #########################################################
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
        #######  yao add  #######
        iou_boxs = np.zeros([self.max_objs, 3, 8], dtype=np.float32)
        target_conner_2d = np.zeros([self.max_objs, 2, 8], dtype=np.float32)
        #######  yao add  #######
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
        bound_box_flag = False
        
        
        
        target_dict = {}
        
        for i, a in enumerate(anns):
            a = a.copy()
            cls = a["label"]
            dims = np.array(a["dimensions"])
            locs = np.array(a["locations"])
            rot_y = np.array(a["rot_y"])
            #point, box2d, box3d = encode_label(K, rot_y, dims, locs)
            point, box2d, box3d, conner_2d = encode_label(K, rot_y, dims, locs)

            #######  yao add  #######
            iou_box = encode_label_iou_box(dims, locs)
            #######  yao add  #######

            ### lyb:added for show###############################################################
            box2d_temp = box2d.copy()
            box2d_temp[[0, 2]] = box2d_temp[[0, 2]].clip(0, self.infer_width - 1)
            box2d_temp[[1, 3]] = box2d_temp[[1, 3]].clip(0, self.infer_height - 1)
            cv2.rectangle(img_save, (int(box2d_temp[0]), int(box2d_temp[1])), (int(box2d_temp[2]), int(box2d_temp[3])),
                          color=(0, 0, 255),
                          thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(img_save, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            str_txt = str(a["dimensions"][1]) + " " + str(a["locations"][2])
            cv2.putText(img_save, str_txt, (int(box2d_temp[0]), int(box2d_temp[1] + 10)), font, 0.5, (255, 255, 255),
                        1)  ### 参数分别为：原图，文字，文字坐标，字体，字体大小，字体颜色，字体粗细
            #############################################################################

            point = affine_transform(point, trans_mat)
            box2d[:2] = affine_transform(box2d[:2], trans_mat)
            box2d[2:] = affine_transform(box2d[2:], trans_mat)
            box2d[[0, 2]] = box2d[[0, 2]].clip(0, self.output_width - 1)
            box2d[[1, 3]] = box2d[[1, 3]].clip(0, self.output_height - 1)
            h, w = box2d[3] - box2d[1], box2d[2] - box2d[0]

            if w <= 1.5 or h <= 1.5:
                continue

            if (0 < point[0] < self.output_width) and (0 < point[1] < self.output_height):
                point_int = point.astype(np.int32)
                p_offset = point - point_int
                
                radius = gaussian_radius(h, w, thresh_min=self.radius_iou)
                radius = max(0, int(radius))
                heat_map[cls] = draw_umich_gaussian(heat_map[cls], point_int, radius)
                
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
                            
                for POINT,score in zip(all_point,scores):
                    if ( POINT[0]<=0 or POINT[0]>=self.output_width or POINT[1]<=0 or POINT[1]>=self.output_height):
                        continue
                    if ( POINT not in target_dict or score>target_dict[POINT]["score"]):
                        point_dict = {}
                        point_dict["cls_ids"] = cls
                        point_dict["regression"] = box3d
                        #point_dict["proj_points"] = (POINT[0],POINT[1])
                        point_dict["dimensions"] = dims
                        point_dict["score"] = score
                        point_dict["locations"] = locs
                        point_dict["rotys"] = rot_y
                        point_dict["p_offsets"] = point - np.array(POINT,dtype=np.float32)
                        point_dict["iou_box"] = iou_box
                        point_dict["conner_2d"] = conner_2d
                        target_dict[POINT] = point_dict
        target_idx = 0
        for POINT,anno_dict in target_dict.items():
            cls_ids[target_idx] = anno_dict["cls_ids"]
            regression[target_idx] = anno_dict["regression"]
            p_offsets[target_idx] = anno_dict["p_offsets"]
            dimensions[target_idx] = anno_dict["dimensions"]
            locations[target_idx] = anno_dict["locations"]
            rotys[target_idx] = anno_dict["rotys"]
            proj_points[target_idx] = np.array(POINT,dtype=np.float32)
            iou_boxs[target_idx] = anno_dict["iou_box"]
            target_conner_2d[target_idx] = anno_dict["conner_2d"]
            reg_mask[target_idx] = 1
            flip_mask[target_idx] = 1 if flipped else 0
            target_idx += 1
            

        if self.is_train:
            # heat_map = heat_map_mask * heat_map
            # ### lyb: save img ##################################################################
            # heat_map_temp = np.sum(heat_map, axis=0) * 255
            # heat_map_temp = Image.fromarray(heat_map_temp)
            # heat_map_temp = heat_map_temp.transform((self.infer_width, self.infer_height),
            #                                         method=Image.AFFINE,
            #                                         data=trans_mat.flatten()[:6],
            #                                         resample=Image.BILINEAR)
            # heat_map_temp = np.array(heat_map_temp)
            # heat_map_temp = np.expand_dims(heat_map_temp, axis=2)
            # heat_map_temp = np.repeat(heat_map_temp, 3, axis=2)
            # img_save = np.hstack([img_save, heat_map_temp])
            #
            # save_dir = "/media/york/H/waymo_img_temp/img_debug115_round_riou0.7_train_laplace1.0"
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            # # if bound_box_flag:
            # cv2.imwrite(os.path.join(save_dir, self.image_files[idx]), img_save)
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
            target.add_field("p_offsets",p_offsets)
            target.add_field("iou_box", iou_boxs)
            target.add_field("conner_2d", target_conner_2d)

            if self.transforms is not None:
                # BGR 255 image
                img, target = self.transforms(img, target)
            # print(np.max(np.array(img)))
            return img, target, original_idx
        else:
            # for inference we parametrize with original size
            target = ParamsList(image_size=[src_width, src_height], is_train=self.is_train)
            target.add_field("K_src", K_src)
            target.add_field("trans_mat", trans_mat)
            target.add_field("K", K)
            ###  yao add '#'
            #target.add_field("proj_p", proj_points)
            #target.add_field("reg_mask", reg_mask)
            #target.add_field("locations", locations)
            #target.add_field("cls_ids", cls_ids)
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
                if row["type"] in self.classes and float(row['lz']) < self.distance_threshold:
                    ### lyb:rename object'height car to truck
                    error_car = False
                    if row["type"] == "Car" and float(row['dh']) > 2.7:
                        error_car = True
                        row["type"] = "Truck"

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
                        "ymax": row["ymax"],
                        "error_car": error_car
                    })

        # get camera intrinsic matrix K
        proj_type = 'P{}:'.format(CAMERA_TO_ID[self.camera])
        with open(os.path.join(self.calib_dir, file_name), 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=' ')
            for line, row in enumerate(reader):
                ### lyb:add
                if not row:
                    continue

                if row[0] == proj_type:
                    P = row[1:]
                    P = [float(i) for i in P]
                    P = np.array(P, dtype=np.float32).reshape(3, 4)
                    K = P[:3, :3]
        return annotations, K
