# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2021 Toyota Research Institute.  All rights reserved.
import copy
import logging
from typing import List, Union

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as d2_utils
from detectron2.data import transforms as T

from tridet.data.augmentations import build_augmentation
from tridet.data.transform_utils import annotations_to_instances, transform_instance_annotations
from tridet.structures.pose import Pose
from tridet.utils.tasks import TaskManager

LOG = logging.getLogger(__name__)

__all__ = ["DefaultDatasetMapper"]


class DefaultDatasetMapper:
    """
    This is adapted from:
        https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/dataset_mapper.py

    The changes from the original:
        - It uses a custom `build_augmentation()`, which is different from its original by
            1) applying a expanded version of `ResizeTransform` (and therefore `ResizeShortestEdge`).
            2) disabling `RandomFlip()` transform for now; if one wants to use it, one needs to implement methods
            for handling intrinsics and 3D box.
        - In the `__call__()`, the (crop / resize) transformation is applied to intrinsics and 3D box (TODO).
        - It uses custom versions of `trainsform_instance_annotations` and `annotations_to_instances`.
        - It takes `TaskManager` object as input to handle various settings of multitask.

    ===================================================================================================================
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """
    @configurable
    def __init__(
        self,
        is_train: bool,
        task_manager,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
    ):
        """
        NOTE: 'configurable' interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        # yapf: disable
        self.is_train                         = is_train
        self.task_manager                     = task_manager
        self.augmentations                    = T.AugmentationList(augmentations)
        self.image_format                     = image_format
        # yapf: enable

        LOG.info("Augmentations used in training: " + str(augmentations))
    # @configurable will first do this
    # the return "ret" will use to initialize DefaultDatasetMapper
    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        # build augmentation from cfg.INPUT
        augs = build_augmentation(cfg, is_train)
        # define box2d, box3d, depth is opened or not
        #        yes    yes    no
        tm = TaskManager(cfg)
        ret = {
            "is_train": is_train,
            "task_manager": tm,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,  #BGR
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # dataset_dict:
        #       intrinsics: 9-meta vec
        #       file_name:  img path
        #       width & height:
        #       image_id:   000000_camera_2
        #       sample_id:  000000
        #       extrinsics: {'wxyz':..., 'tvec':...}    # from camera_2 frame to Velodyne frame
        #       annotations:  list[dict]   each dict represent a object
        #           category_id:0
        #           instance_id:000000_1(obj_idx)
        #           bbox3d: 10-meta-vec  quat(4), tvec(3), dim(3-width, length, height)
        #           distance: l2 norm of center car xyz(tvec)
        #           bbox=[l, t, r, b]   left_up & right_bottom, (xmin,ymin),(xmax,ymax) in image axis
        #           bbox_mode=BoxMode.XYXY_ABS
        #       raw_kitti_annotations: raw kitti label
        # add & modify:
        #       image: tensor(C,H,W)   auged image!!!
        #       instances: filled with augmented annotations
        #                Instances(image_size)
        #                .gt_boxes  .gt_classes  .gt_boxes3d
        #       intrinsics: 3*3 & augmented     
        #       inv_intrinsics: 3*3 & augmented 
        #       extrinsics: Pose('wxyz', 'tvec')   # from camera_2 frame to Velodyne frame
        # example for instance:
        """
        Instances(num_instances=5, image_height=416, image_width=1378, 
        fields=[gt_boxes: Boxes(tensor([[733.6352, 193.9669, 770.4152, 219.2375],
                                        [138.8207, 101.7370, 525.9478, 280.9498],
                                        [759.4422, 191.0161, 814.2072, 227.7018],
                                        [  0.0000, 214.7226, 428.4892, 414.8907],
                                        [548.5927, 201.7212, 585.2395, 227.7129]])), 
        gt_classes: tensor([0, 4, 0, 0, 0]), 
        gt_boxes3d: <tridet.structures.boxes3d.Boxes3D object at 0x7f0070d50c70>])
                .quat           :[num_instances, 4]
                .proj_ctr       :[num_instances, 2]       uv
                .depth          :[num_instances, 1]       z
                .size           :[num_instances, 2]       width, length, height
                .inv_intrinsics :[num_instances, 3, 3]
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = d2_utils.read_image(dataset_dict["file_name"], format=self.image_format)
        #print('\n\n\n')
        #print(f'origin shape: {image.shape[:2]}')
        d2_utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic seglmentation.
        semseg_gt, depth_gt, intrinsics, pose = None, None, None, None
        if "sem_seg_file_name" in dataset_dict:  # False
            semseg_gt = d2_utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)

        aug_input = T.AugInput(image, sem_seg=semseg_gt)
        # this line 'aug_input.image' is not augmented

        # (dennis.park) `self.augmentations` is [(RandomCrop,) ResizeShortestEdge, (RandomFlip), (ColorJitter)]
        transforms = self.augmentations(aug_input)

        image, semseg2d_gt = aug_input.image, aug_input.sem_seg
        #print(f'auged shape: {image.shape[:2]}')
        #print('\n\n\n')
        # image: auged image,  semseg2d_gt:None
        # transforms: a class for image augmentation


        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if semseg2d_gt is not None:
            dataset_dict["semseg2d"] = torch.as_tensor(semseg2d_gt.astype("long"))

        # `transforms` apply the following transformations in order:
        #   1. (optional) CropTransform
        #   2. ResizeTransform
        #   3. (optional) FlipTransform
        #   4. (optional) color jittering tranforms (brightness, saturation, contrast).
        # See `multitask.data.augmentations.build.py`
        # If you add other transformations, then consider if they have to support intrinsics, 3D bbox, depth. etc.
        # See crop_transform.py, resize_transform.py, flip_transform.py for examples.

        if "depth_file_name" in dataset_dict: # False
            depth_gt = np.load(dataset_dict.pop("depth_file_name"))['data']
            depth_gt = transforms.apply_depth(depth_gt)
            dataset_dict["depth"] = torch.as_tensor(depth_gt)

        intrinsics = None
        if "intrinsics" in dataset_dict:
            intrinsics = np.reshape(
                dataset_dict["intrinsics"],
                (3, 3),
            ).astype(np.float32)
            intrinsics = transforms.apply_intrinsics(intrinsics)
            dataset_dict["intrinsics"] = torch.as_tensor(intrinsics)
            dataset_dict["inv_intrinsics"] = torch.as_tensor(np.linalg.inv(intrinsics))

        if "pose" in dataset_dict: # False
            pose = Pose(wxyz=np.float32(dataset_dict["pose"]["wxyz"]), tvec=np.float32(dataset_dict["pose"]["tvec"]))
            dataset_dict["pose"] = pose
            # NOTE: no transforms affect global pose.

        if "extrinsics" in dataset_dict:
            extrinsics = Pose(
                wxyz=np.float32(dataset_dict["extrinsics"]["wxyz"]),
                tvec=np.float32(dataset_dict["extrinsics"]["tvec"])
            )
            dataset_dict["extrinsics"] = extrinsics

        if not self.task_manager.has_detection_task:
            dataset_dict.pop("annotations", None)

        if "annotations" in dataset_dict:

            for anno in dataset_dict["annotations"]:
                if not self.task_manager.has_detection_task:
                    anno.pop("bbox", None)
                    anno.pop("bbox_mode", None)
                if not self.task_manager.box3d_on:
                    anno.pop("bbox3d", None)

            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                ) for obj in dataset_dict["annotations"] if obj.get("iscrowd", 0) == 0
            ]

            if annos and 'bbox3d' in annos[0]:
                # Remove boxes with negative z-value for center.
                annos = [anno for anno in annos if anno['bbox3d'][6] > 0]

            instances = annotations_to_instances(
                annos,
                image_shape,
                intrinsics=intrinsics,
            )

            if self.is_train:
                instances = d2_utils.filter_empty_instances(instances)
            dataset_dict["instances"] = instances

        return dataset_dict
