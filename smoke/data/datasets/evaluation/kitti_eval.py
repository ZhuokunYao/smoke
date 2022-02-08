import os
import logging
from tool.pykitti_eval import kitti_common as kitti
from tool.pykitti_eval.eval import evaluate_mAP


def kitti_evaluation(cfg, eval_type, dataset_name):
    logger = logging.getLogger(__name__)
    gt_label_path = "datasets/kitti/training/label_2/"
    if "detection" in eval_type:
        logger.info("performing KITTI detection evaluation: ")
        pred_label_path = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        pred_annos, image_ids = kitti.get_label_annos(pred_label_path, return_ids=True)
        gt_annos = kitti.get_label_annos(gt_label_path, image_ids=image_ids)
        assert len(pred_annos) == len(gt_annos)
        return evaluate_mAP(gt_annos, pred_annos, cfg.DATASETS.DETECT_CLASSES, cfg.TEST.DEPTH_RANGES,
                            cfg.TEST.IOU_THRESHOLDS)
