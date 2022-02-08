import os
import logging
from tool.pykitti_eval import kitti_common as kitti
from tool.pykitti_eval.eval import evaluate_mAP
from smoke.utils.imports import import_file


def waymo_evaluation(cfg, eval_type, dataset_name):
    logger = logging.getLogger(__name__)
    path_catalog = import_file("smoke.config.paths_catalog", cfg.PATHS_CATALOG, True)
    DatasetCatalog = path_catalog.DatasetCatalog
    data_dir = DatasetCatalog.DATA_DIR
    data_attrs = DatasetCatalog.DATASETS[dataset_name]
    gt_label_path = os.path.join(data_dir, data_attrs['root'], "label_2", data_attrs['camera'])

    if "detection" in eval_type:
        logger.info("performing WAYMO detection evaluation: ")
        pred_label_path = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        pred_annos, image_ids = kitti.get_label_annos(pred_label_path, return_ids=True)
        gt_annos = kitti.get_label_annos(gt_label_path, image_ids=image_ids)
        #annos:
        #list of (per image)
        # for each object
        #[all in list] name,truncated,occluded,alpha,bbox(4),dimensions(l,h,w),location(3)
        #[all in list] rotation_y,score,index(one of 0,1,2,...numobject-1),group_ids(same with index)
        #[all in list] image_idx

        #image_ids:
        #list of image_index (per image)

        assert len(pred_annos) == len(gt_annos)
        return evaluate_mAP(gt_annos, pred_annos, cfg.DATASETS.DETECT_CLASSES, cfg.TEST.DEPTH_RANGES,
                            cfg.TEST.IOU_THRESHOLDS)
