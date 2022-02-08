import argparse
import os
from tools.pykitti_eval import kitti_common as kitti
from tools.pykitti_eval.eval import evaluate_mAP

DATASET_CLASS = {
    'kitti': ['Car', 'Cyclist', 'Pedestrian'],
    'jdx': ['Car', 'Cyclist', 'Pedestrian', 'Truck', 'Tricycle', 'Bus', 'Cyclist_stopped'],
    'waymo': ['Car', 'Cyclist', 'Pedestrian', 'Truck', 'Tricycle', 'Bus', 'Cyclist_stopped'],
}

parser = argparse.ArgumentParser(description='Evaluate mAP in KITTI format.')
parser.add_argument('--dataset_type', type=str, default='jdx', help='Specify the dataset type')
parser.add_argument('--gt_label_path', type=str, help='The path of ground truth label',
                    default='datasets/jdx_test/training/label_2/')
parser.add_argument('--pred_label_path', type=str, help='The path of prediction label',
                    default='demo/jdx_test/prediction/')
args = parser.parse_args()


def evaluate_kitti_mAP(gt_label_path, pred_label_path, class_name):
    if not os.path.exists(gt_label_path):
        print('GroundTruth label path not found')
    if not os.path.exists(pred_label_path):
        print('Prediction label path not found')
    pred_annos, image_ids = kitti.get_label_annos(pred_label_path, return_ids=True)
    gt_annos = kitti.get_label_annos(gt_label_path, image_ids=image_ids)
    print('Prediction_annos: ', len(pred_annos))
    print('GroundTruth annos: ', len(gt_annos))
    assert len(pred_annos) == len(gt_annos)
    return evaluate_mAP(gt_annos, pred_annos, class_name)


if __name__ == '__main__':
    gt_label_path = args.gt_label_path
    pred_label_path = args.pred_label_path
    dataset_type = args.dataset_type

    result_dict = evaluate_kitti_mAP(gt_label_path, pred_label_path, DATASET_CLASS[dataset_type])
    print(result_dict['print'])
