import math
import copy
import random
import numpy as np
from PIL import Image

from smoke.modeling.heatmap_coder import get_transfrom_matrix, affine_transform
from smoke.data.kitti_utils import *


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, annotations, P):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            self.PIL2Numpy = True

        for augmentation in self.augmentations:
            img, annotations, P = augmentation(img, annotations, P)

        if self.PIL2Numpy:
            img = np.array(img)

        return img, annotations, P


class RandomHSV(object):
    #  0.5, (0.15, 0.7, 0.4)
    def __init__(self, prob, hgain=0.5, sgain=0.5, vgain=0.5):
        super(RandomHSV, self).__init__()
        self.prob = prob
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, img, annotations, P):
        if self.prob <= 0:
            return img, annotations, P
        if random.random() < self.prob:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain,
                                               self.vgain] + 1  # random gains
            hue, sat, val = cv2.split(
                cv2.cvtColor(np.array(img), cv2.COLOR_BGR2HSV))
            x = np.arange(0, 256, dtype=np.int16)
            lut_hue = ((x * r[0]) % 180).astype(hue.dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(hue.dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(hue.dtype)
            img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat),
                                 cv2.LUT(val, lut_val))).astype(hue.dtype)
            img = Image.fromarray(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR))

        return img, annotations, P


class RandomHorizontallyFlip(object):
    # 0.5
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, img, annotations, P):
        if self.prob <= 0:
            return img, annotations, P
        if random.random() < self.prob:
            # flip image
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_w, img_h = img.size

            # flip labels
            for idx, annotation in enumerate(annotations):
                # flip box2d
                w = annotation['bbox_2d'][2] - annotation['bbox_2d'][0]
                annotation['bbox_2d'][0] = img_w - annotation['bbox_2d'][2] - 1
                annotation['bbox_2d'][2] = annotation['bbox_2d'][0] + w

            # mofify P if flip
            P[0, 0] = -P[0, 0]
            P[0, 2] = (img_w - 1) - P[0, 2]

            return img, annotations, P
        else:
            return img, annotations, P


class RandomAffineTransformation(object):
    #  0.3   （0.2,0.4）  [cfg.INPUT.WIDTH_TRAIN,cfg.INPUT.HEIGHT_TRAIN] classes except car
    def __init__(self, prob, shift_scale, dst_size, aug_class):
        self.prob = prob
        self.shift_scale = shift_scale
        self.dst_size = dst_size
        self.aug_class = aug_class

    def __call__(self, img, annotations, P):
        if self.prob <= 0:
            return img, annotations, P
        if random.random() < self.prob:
            # Crop image with fixed size which is equal to target image
            # return center, size=self.dst_size
            center_size = self.get_random_crop_center_size(img.size)
        else:
            # Crop image with random size which is larger than the target image
            # return center, size=self.dst_size * ratio(0.6, max_ratio)
            center_size = self.get_random_resize_center_size(img.size)
        if center_size is None:
            return img, annotations, P
        center, size = center_size
        width_crop, height_crop = size
        bbox_crop = [center[0] - width_crop / 2 + 1,   #xmin
                     center[1] - height_crop / 2 + 1,  #ymin
                     center[0] + width_crop / 2 + 1,   #xmax
                     center[1] + height_crop / 2 + 1]  #ymax
        trans_affine = get_transfrom_matrix([center, size], self.dst_size)
        trans_affine_inv = np.linalg.inv(trans_affine)
        image_crop = img.transform(
            (int(self.dst_size[0]), int(self.dst_size[1])),
            method=Image.AFFINE,
            data=trans_affine_inv.flatten()[:6],
            resample=Image.BILINEAR)

        # Adjust and save Projection params
        K_src = P.copy()[:3, :3]
        P[:3, :3] = np.matmul(trans_affine, P[:3, :3])

        # Filter and save the annotations
        remained = []
        for annotation in annotations:
            # bottle of the car     -h/2
            center_2d = decode_center_2d(annotation['dimensions'],
                                         annotation['locations'],
                                         K_src)
            # why or?  not and?
            if ((bbox_crop[0] <= center_2d[0] < bbox_crop[2])
                    or (bbox_crop[1] <= center_2d[1] < bbox_crop[3])):
                bbox_2d = np.array(annotation['bbox_2d'])
                bbox_2d[:2] = affine_transform(
                    bbox_2d[:2], trans_affine)
                bbox_2d[2:] = affine_transform(
                    bbox_2d[2:], trans_affine)
                bbox_2d[[0, 2]] = bbox_2d[
                    [0, 2]].clip(0, width_crop - 1)
                bbox_2d[[1, 3]] = bbox_2d[
                    [1, 3]].clip(0, height_crop - 1)
                annotation['bbox_2d'] = bbox_2d
                remained.append(annotation)

        return image_crop, remained, P

    # Crop image with fixed size which is equal to target image
    def get_random_crop_center_size(self, src_size):
        size = np.array([self.dst_size[0], self.dst_size[1]])
        x0 = random.choice(np.arange(0.0, src_size[0] - size[0], 1.0))
        if (src_size[1] - size[1]) > 1.0:
            y0 = random.choice(np.arange(0.0, src_size[1] - size[1], 1.0))
        else:
            y0 = 0.0
        center = np.array([x0 + size[0] / 2, y0 + size[1] / 2])
        return [center, size]

    # Crop image with random size which is larger than the target image
    def get_random_resize_center_size(self, src_size):
        shift, scale = self.shift_scale[0], self.shift_scale[1]
        max_ratio = min(src_size[0] / self.dst_size[0],
                        src_size[1] / self.dst_size[1])
        resize_ratio = random.choice(np.arange(1 - scale, max_ratio, 0.01))
        size = np.array(self.dst_size) * resize_ratio
        x0 = random.choice(np.arange(0.0, src_size[0] - size[0], 1.0))
        y0 = random.choice(np.arange(0.0, src_size[1] - size[1], 1.0))
        center = np.array([x0 + size[0] / 2, y0 + size[1] / 2])

        return [center, size]

    # Crop image with random size which is larger than the special size, tightly
    # including all the lang-tail classes
    def get_random_center_size_tight(self, src_size, annotations):
        ### crop image with equal aspect ratio of target size
        width, height = src_size
        ratio_base = min(self.dst_size[0] / width, self.dst_size[1] / height)
        bbox_base = None
        for idx, annotation in enumerate(annotations):
            if annotation["class"] in self.aug_class:
                if bbox_base is None:
                    bbox_base = annotation["bbox_2d"]
                else:
                    bbox_base[0] = min(annotation["bbox_2d"][0], bbox_base[0])
                    bbox_base[1] = min(annotation["bbox_2d"][1], bbox_base[1])
                    bbox_base[2] = max(annotation["bbox_2d"][2], bbox_base[2])
                    bbox_base[3] = max(annotation["bbox_2d"][3], bbox_base[3])
        if bbox_base is None:
            return None
        center_base = [(bbox_base[0] + bbox_base[2]) / 2,
                       (bbox_base[1] + bbox_base[3]) / 2]
        width_base, height_base = bbox_base[2] - bbox_base[0], bbox_base[
            3] - bbox_base[1]
        ratio_base = max(ratio_base, max(width_base / width,
                                         height_base / height))
        if ratio_base == 1:
            return None
        ratio_crop = random.choice(np.arange(ratio_base, 1, 0.01))
        center_crop = center_base
        height_crop, width_crop = ratio_crop * height, ratio_crop * width
        # rectify the center of crop bbox
        if center_base[0] - width_crop / 2 <= 0:
            center_crop[0] = center_crop[0] + (
                    width_crop / 2 - center_base[0])
        if center_base[0] + width_crop / 2 >= width:
            center_crop[0] = center_crop[0] - (
                    center_base[0] + width_crop / 2 - width_crop)
        if center_base[1] - height_crop / 2 <= 0:
            center_crop[1] = center_crop[1] + (
                    height_crop / 2 - center_base[1])
        if center_base[1] + height_crop / 2 >= height:
            center_crop[1] = center_crop[1] - (
                    center_base[1] + height_crop / 2 - height_crop)
        bbox_crop = [max(0, center_crop[0] - width_crop / 2),
                     min(0, center_crop[1] - height_crop / 2),
                     min(width, center_crop[0] + width_crop / 2),
                     min(height, center_crop[1] + height_crop / 2)]
        # Calcuate the shift range to [left, top, right, bottom]
        border_crop_src = [bbox_crop[0], bbox_crop[1],
                           width - bbox_crop[2] - 1,
                           height - bbox_crop[3] - 1]
        border_crop_base = [bbox_base[0] - bbox_crop[0],
                            bbox_base[1] - bbox_crop[1],
                            bbox_crop[2] - bbox_base[2],
                            bbox_crop[3] - bbox_base[3]]
        border_range = [min(border_crop_base[2], border_crop_src[0]),
                        min(border_crop_base[3], border_crop_src[1]),
                        min(border_crop_base[0], border_crop_src[2]),
                        min(border_crop_base[1], border_crop_src[3])]
        shift_x, shift_y = 0, 0
        if (border_range[0] + border_range[2]) == 0:
            shift_x = 0
        elif (border_range[1] + border_range[3]) == 0:
            shift_y = 0
        else:
            shift_x = random.choice(np.arange(
                -border_range[0] / (border_range[0] + border_range[2]),
                border_range[2] / (border_range[0] + border_range[2]),
                0.01)) * (
                              border_range[0] + border_range[2])
            shift_y = random.choice(np.arange(
                -border_range[1] / (border_range[1] + border_range[3]),
                border_range[3] / (border_range[1] + border_range[3]),
                0.01)) * (
                              border_range[1] + border_range[3])
        center = np.array(
            [center_base[0] + shift_x, center_base[1] + shift_y])
        size = [width_crop, height_crop]

        return [center, size]


class RandomCutMixup(object):
    def __init__(self, prob, aug_path=None,
                 dst_size=[640, 480],
                 paste_range=[2, 8],
                 aug_class=["Cyclist", "Truck", "Tricycle", "Bus"]):
        super(RandomCutMixup, self).__init__()
        self.prob = prob
        self.paste_range = paste_range
        self.aug_path = aug_path
        self.dst_size = dst_size
        self.aug_class = aug_class
        self.aug_dataset = self.setup_augment_dataset() if self.prob > 0 else None
        self.iou_thresh = 0.3

    def setup_augment_dataset(self):
        if self.aug_path is None or not os.path.exists(self.aug_path):
            print("CutMix path is not exist!")
        aug_list = {}
        for cls in self.aug_class:
            aug_list[cls] = [line[:-4] for line in os.listdir(
                os.path.join(self.aug_path, cls, 'image_2'))]
        return aug_list

    def random_select_cutmix_object(self):
        cls_type = random.choice(self.aug_class)
        image_name = random.choice(self.aug_dataset[cls_type])
        img = Image.open(os.path.join(self.aug_path, cls_type, 'image_2',
                                      image_name + '.jpg'))
        anno = load_annotation(os.path.join(self.aug_path, cls_type, 'label_2',
                                            image_name + '.txt'))
        K, P = load_intrinsic_matrix(
            os.path.join(self.aug_path, cls_type, 'calib', image_name + '.txt'))
        return img, anno, P

    def __call__(self, img, annotations, P):
        if self.prob <= 0:
            P_list = [P for i in np.arange(len(annotations))]
            return img, annotations, P_list

        if random.random() >= self.prob:
            P_list = [P for i in np.arange(len(annotations))]
            return img, annotations, P_list

        size = np.array([i for i in img.size], dtype=np.float32)
        center = np.array([i / 2 for i in size], dtype=np.float32)
        trans_affine = get_transfrom_matrix([center, size], self.dst_size)
        trans_affine_inv = np.linalg.inv(trans_affine)
        img = img.transform(
            (int(self.dst_size[0]), int(self.dst_size[1])),
            method=Image.AFFINE,
            data=trans_affine_inv.flatten()[:6],
            resample=Image.BILINEAR)

        # Adjust Projection params, extend to list
        P[:3, :3] = np.matmul(trans_affine, P[:3, :3])
        P_list = [P for i in np.arange(len(annotations))]

        # Adjust annotations
        for annotation in annotations:
            bbox_2d = np.array(annotation['bbox_2d'])
            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans_affine)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans_affine)
            bbox_2d[[0, 2]] = bbox_2d[[0, 2]].clip(0, self.dst_size[0] - 1)
            bbox_2d[[1, 3]] = bbox_2d[[1, 3]].clip(0, self.dst_size[1] - 1)
            annotation['bbox_2d'] = bbox_2d

        max_paste = random.randint(self.paste_range[0], self.paste_range[1])
        for idx in range(max_paste):
            img_cut, anno_cut, P_cut = self.random_select_cutmix_object()
            target = np.array(anno_cut[0]['bbox_2d'])
            iou_max = 0.0
            for annotation in annotations:
                ref_bbox = annotation['bbox_2d']
                iou_ref = iou_2d(target, ref_bbox, criterion=1)
                if iou_ref > iou_max:
                    iou_max = iou_ref
            if iou_max > self.iou_thresh:
                continue
            else:
                img.paste(img_cut, (target[0], target[1]))
                annotations.append(anno_cut[0])
                P_list.append(P_cut)

        return img, annotations, P_list


class RandomMixup(object):
    def __init__(self, prob, aug_path=None,
                 aug_class=["Cyclist", "Truck", "Tricycle", "Bus"]):
        super(RandomMixup, self).__init__()
        self.prob = prob
        self.aug_path = aug_path
        self.aug_class = aug_class
        self.aug_dataset = self.setup_augment_dataset() if self.prob > 0 else None

    def __call__(self, img, annotations, P):
        if self.prob <= 0:
            return img, annotations, P
        if random.random() < self.prob:
            img_retrieve, anno_retrieve, P_retrieve = self.random_select_mixup_image()
            img_mixup = Image.blend(img, img_retrieve, 0.5)
            annotations.extend(anno_retrieve)

            return img_mixup, annotations, P
        else:
            return img, annotations, P

    def setup_augment_dataset(self):
        if self.aug_path is None or not os.path.exists(self.aug_path):
            print("Mixup path is not exist!")
        aug_list = []
        for line in os.listdir(os.path.join(self.aug_path, 'label_2')):
            annotaions = load_annotation(
                os.path.join(self.aug_path, 'label_2', line), self.aug_class)
            if len(annotaions) > 0:
                aug_list.append(line[:-4])
        return aug_list

    def random_select_mixup_image(self):
        image_name = random.choice(self.aug_dataset)
        img = Image.open(os.path.join(self.aug_path, 'image_2',
                                      image_name + '.jpg'))
        anno = load_annotation(os.path.join(self.aug_path, 'label_2',
                                            image_name + '.txt'))
        K, P = load_intrinsic_matrix(
            os.path.join(self.aug_path, 'calib', image_name + '.txt'))
        return img, anno, P
