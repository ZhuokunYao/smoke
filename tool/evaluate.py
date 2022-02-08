import os
import argparse
import torch
from smoke.modeling.detector import build_detection_model
from smoke.config import cfg
from smoke.utils.model_serialization import load_state_dict
from smoke.data import build_test_loader
from smoke.engine.inference import inference
#from smoke.utils import comm
from smoke.utils.miscellaneous import mkdir


def evaluate(cfg, model):
    eval_types = ('detection',)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    # ["jdx_simu_front", "jdx_test_front"]
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, 'inference', dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = build_test_loader(cfg)
    result_dict = {}
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        result = inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            eval_types=eval_types,
            device=cfg.MODEL.DEVICE,
            output_folder=output_folder,
        )
        #comm.synchronize()
        if result is not None:
            result_dict[dataset_name] = result
    return result_dict


def main():
    parser = argparse.ArgumentParser(description='SMOKE Model Evaluation')
    parser.add_argument('--config_file', type=str, help='Path to config file',
                        default='configs/smoke_jdx_resnet18_640x480.yaml')
    parser.add_argument('--model_path', type=str, help='Path to trained checkpoint',
                        default='path/to/ur/checkpoint.pth')
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    if 'backbone_channels' in checkpoint.keys() and len(checkpoint['backbone_channels']) > 0:
        cfg.MODEL.BACKBONE.CHANNELS = checkpoint['backbone_channels']
        cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS = checkpoint['backbone_channels'][-1]

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()
    load_state_dict(model, checkpoint['model'])

    result = evaluate(cfg, model)
    for test_set in cfg.DATASETS.TEST:
        print("Evaluation result of dataset [{}]: \n".format(test_set))
        print(result[test_set]["print"])

if __name__ == '__main__':
    main()