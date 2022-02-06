import os


class DatasetCatalog():
    DATA_DIR = "datasets"
    DATASETS = {
        "kitti_train": {
            "root": "kitti/training/",
            "split": "train",
        },
        "kitti_val": {
            "root": "kitti/training/",
            "split": "val",
        },
        "kitti_trainval": {
            "root": "kitti/training/",
            "split": "trainval",
        },
        "kitti_test": {
            "root": "kitti/testing/",
            "split": "test",
        },
        "jdx_sod_pesudo_front": {
            "root": "sod_pesudo/front/training/",
            "split": "train",
            "camera": "front",
        },
        "jdx2020_tracker": {
            "root": "jdx2020_tracker/training/",
            "split": "train",
        },
        "jdx2021_fusion": {
            "root": "jdx2021_fusion/training/",
            "split": "train",
        },
        "jdx2021_tracker": {
            "root": "jdx2021_tracker/training/",
            "split": "train",
        },
        "jdx_fusion_front": {
            "root": "jdx_fusion/front/training/",
            "split": "train",
        },
        "jdx_fusion_rear": {
            "root": "jdx_fusion/rear/training/",
            "split": "train",
        },
        "jdx_fusion_left": {
            "root": "jdx_fusion/left/training/",
            "split": "train",
        },
        "jdx_simu_front": {
            "root": "jdx_simu/front/training/",
            "split": "val",
        },
        "jdx_simu_rear": {
            "root": "jdx_simu/rear/training/",
            "split": "val",
        },
        "jdx_test_front": {
            "root": "jdx_test/front/training/",
            "split": "val",
        },
        "jdx_test_rear": {
            "root": "jdx_test/rear/training/",
            "split": "val",
        },
        "waymofront_train": {
            "root": "waymo_front/training/",
            "split": "train",
            "camera": "front",
            "sample_ratio": 1,
        },
        "waymofront_val": {
            "root": "waymo_front/training/",
            "split": "val",
            "camera": "front",
            "sample_ratio": 1,
        },
        "waymo720front_train": {
            "root": "waymo720_front/train/",
            "split": "train",
            "camera": "front",
            "sample_ratio": 1,
            "distance_threshold": 80,
        },
        "waymo720front_val": {
            "root": "waymo720_front/val/",
            "split": "val",
            "camera": "front",
            "sample_ratio": 1,
            "distance_threshold": 80,
        },
        "waymo720_train_front": {
            "root": "waymo720/train/",
            "split": "train",
            "camera": "front",
            "sample_ratio": 1,
            "distance_threshold": 80,
        },
        "waymo720_val_front": {
            "root": "waymo720/val/",
            "split": "val",
            "camera": "front",
            "sample_ratio": 1,
            "distance_threshold": 80,
        },
        "waymo720_train_front_left": {
            "root": "waymo720/train/",
            "split": "train",
            "camera": "front_left",
            "sample_ratio": 1,
            "distance_threshold": 80,
        },
        "waymo720_val_front_left": {
            "root": "waymo720/val/",
            "split": "val",
            "camera": "front_left",
            "sample_ratio": 1,
            "distance_threshold": 80,
        },
        "waymo720_train_front_right": {
            "root": "waymo720/train/",
            "split": "train",
            "camera": "front_right",
            "sample_ratio": 1,
            "distance_threshold": 80,
        },
        "waymo720_val_front_right": {
            "root": "waymo720/val/",
            "split": "val",
            "camera": "front_right",
            "sample_ratio": 1,
            "distance_threshold": 80,
        },
        "waymo720_train_side_left": {
            "root": "waymo720/train/",
            "split": "train",
            "camera": "side_left",
            "sample_ratio": 1,
            "distance_threshold": 80,
        },
        "waymo720_val_side_left": {
            "root": "waymo720/val/",
            "split": "val",
            "camera": "side_left",
            "sample_ratio": 1,
            "distance_threshold": 80,
        },
        "waymo720_train_side_right": {
            "root": "waymo720/train/",
            "split": "train",
            "camera": "side_right",
            "sample_ratio": 1,
            "distance_threshold": 80,
        },
        "waymo720_val_side_right": {
            "root": "waymo720/val/",
            "split": "val",
            "camera": "side_right",
            "sample_ratio": 1,
            "distance_threshold": 80,
        },
    }

    @staticmethod
    def get(name):
        if "kitti" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["root"]),
                split=attrs['split'],
            )
            return dict(
                factory="KITTIDataset",
                args=args,
            )
        if "jdx" in name:
            data_dir = DatasetCatalog.DATA_DIR
            # "datasets"
            attrs = DatasetCatalog.DATASETS[name]
            # "jdx2020_tracker": {
            # "root": "jdx2020_tracker/training/",
            # "split": "train",
            # }
            args = dict(
                root=os.path.join(data_dir, attrs["root"]),
                split=attrs['split'],
            )
            return dict(
                factory="JDXDataset",
                args=args,
            )
        if "waymo720" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["root"]),
                split=attrs['split'],
                # front
                camera=attrs['camera'],
                # 1
                sample_ratio=attrs['sample_ratio'],
                # 80
                distance_threshold=attrs['distance_threshold']
            )
            return dict(
                factory="WAYMO720Dataset",
                #factory="JDXDataset",
                args=args,
            )
        if "waymo" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["root"]),
                split=attrs['split'],
                camera=attrs['camera'],
                sample_ratio=attrs['sample_ratio']
            )
            return dict(
                factory="WAYMODataset",
                #factory="JDXDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))
