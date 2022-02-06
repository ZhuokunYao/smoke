from smoke.data import datasets

from smoke.data.datasets.evaluation.kitti_eval import kitti_evaluation
from smoke.data.datasets.evaluation.jdx_eval import jdx_evaluation
from smoke.data.datasets.evaluation.waymo_eval import waymo_evaluation

def evaluate(cfg, eval_type, dataset, dataset_name):
    """evaluate dataset using different methods based on dataset type.
    Args:
        eval_type:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(cfg=cfg, eval_type=eval_type, dataset_name = dataset_name)
    if isinstance(dataset, datasets.KITTIDataset):
        return kitti_evaluation(**args)
    if isinstance(dataset, datasets.JDXDataset):
        return jdx_evaluation(**args)
    if isinstance(dataset, datasets.WAYMODataset):
        return waymo_evaluation(**args)
    if isinstance(dataset, datasets.WAYMO720Dataset):
        return waymo_evaluation(**args)
    else:
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset.__class__.__name__))