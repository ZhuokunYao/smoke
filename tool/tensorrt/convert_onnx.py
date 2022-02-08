import argparse
import numpy as np
import torch
from smoke.modeling.detector import build_detection_model
from smoke.config import cfg
from smoke.utils.model_serialization import load_state_dict

import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Convert Pytorch model to ONNX file ...')
parser.add_argument('--cfg_path', type=str, help='The path of config file',
                    default='configs/smoke_jdx_resnet18_640x480.yaml')
parser.add_argument('--model_path', type=str, help='The path of Pytorch model',
                    default='path/to/ur/checkpoint.pth')
parser.add_argument('--onnx_path', type=str, help='The path to save ONNX model',
                    default='path/to/ur/checkpoint.onnx')


args = parser.parse_args()

model_path = args.model_path
cfg_path = args.cfg_path
onnx_path = args.onnx_path

input_names = ['input']
output_names = ['heatmap', 'regression']


def check_onnx(dummy_input):
    import onnxruntime
    session = onnxruntime.InferenceSession(onnx_path)

    for input in session.get_inputs():
        print(input.name)
    for output in session.get_outputs():
        print(output.name)

    onnx_output = session.run(None, {input_names[0]: dummy_input.numpy()})
    return onnx_output


def convert_pth_to_onnx():
    cfg.merge_from_file(cfg_path)
    input_height, input_width = cfg.INPUT.HEIGHT_TEST, cfg.INPUT.WIDTH_TEST
    cfg.CONVERT_ONNX = True
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    if 'backbone_channels' in checkpoint.keys() and len(checkpoint['backbone_channels']) != 0:
        cfg.MODEL.BACKBONE.CHANNELS = checkpoint['backbone_channels']
        cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS = checkpoint['backbone_channels'][-1]

    model = build_detection_model(cfg)
    model.eval()
    load_state_dict(model, checkpoint['model'])

    dummy_input = torch.randn(1, 3, input_height, input_width, device='cpu')
    model.batch_size_onnx = 1
    model.export_to_onnx_mode = True
    with torch.no_grad():
        torch.onnx.export(model, dummy_input, onnx_path, verbose=True,
                          input_names=input_names, output_names=output_names)
        torch_output = model(dummy_input)

    onnx_output = check_onnx(dummy_input)

    np.testing.assert_almost_equal(onnx_output[0], torch_output[0], decimal=2)

    np.testing.assert_almost_equal(onnx_output[1], torch_output[1], decimal=2)


if __name__ == '__main__':
    convert_pth_to_onnx()
