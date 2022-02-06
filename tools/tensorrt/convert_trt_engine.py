import argparse
import os
from PIL import Image
import numpy as np
import csv
import cv2
from tqdm import tqdm
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from torchvision.transforms import functional as F
import torch

from smoke.config import cfg
from smoke.modeling.heads.smoke_head.post_processor import make_smoke_post_processor
from smoke.modeling.heatmap_coder import get_transfrom_matrix
from smoke.structures.params_3d import ParamsList
from tools.utils import compute_box_3d, project_to_image, draw_box_3d

TRT_DATA_TYPE = {
    'fp32': trt.DataType.FLOAT,
    'fp16': trt.DataType.HALF
}
ID_TYPE_CONVERSION = {
    0: 'Car',
    1: 'Cyclist',
    2: 'Pedestrian',
    3: 'Truck',
    4: 'Tricycle',
    5: 'Bus',
    6: 'Cyclist_stopped',
}
CAMERA_TO_ID = {
    'front': 0,
    'front_left': 1,
    'front_right': 2,
    'side_left': 3,
    'side_right': 4,
}


parser = argparse.ArgumentParser(description='Convert ONNX model to TensorRT file ...')
parser.add_argument('--cfg_path', type=str, help='The path of config file',
                    default='configs/smoke_jdx_resnet18_640x480.yaml')
parser.add_argument('--onnx_path', type=str, help='The path of ONNX model',
                    default='path/to/ur/checkpoint.onnx')
parser.add_argument('--engine_path', type=str, help='The path of TensorRT engine',
                    default='path/to/ur/checkpoint.engine')
parser.add_argument('--dataset_type', type=str, help='Specify a dataset type', default='jdx')
parser.add_argument('--camera_type', type=str, help='Specify the camera view, default is None for kitti and jdx',
                    default=None)
parser.add_argument('--trt_data_type', type=str, help='Specify a TensorRT precision', default='fp16')
parser.add_argument('--validation_dir', type=str, help='The path of dataset', default='datasets/jdx_test/front/training/')
parser.add_argument('--output_dir', type=str, help='Specify a dir to save results', default='demo/jdx_test_trt/')

args = parser.parse_args()
onnx_path = args.onnx_path
engine_path = args.engine_path
dataset_type = args.dataset_type
validation_dir = args.validation_dir
camera_type = args.camera_type
trt_data_type = TRT_DATA_TYPE[args.trt_data_type]
cfg.merge_from_file(args.cfg_path)
cfg.ENABLE_TENSORRT = True

# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return 'Host:\n' + str(self.host) + '\nDevice:\n' + str(self.device)

    def __repr__(self):
        return self.__str__()


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        # print(engine.get_binding_name())
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def create_trt_engine(onnx_path, engine_path='', data_type=trt.DataType.HALF):
    '''If the sereized engine is existed， load and run; else create tensorrt engine and save it.'''

    def build_engine(data_type=trt.DataType.HALF):
        '''Takes an ONNX file and creates a TensorRT engine to run inference with'''
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network() as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30  # 1GB
            builder.max_batch_size = 1
            if data_type == trt.DataType.HALF and builder.platform_has_fast_fp16:
                builder.fp16_mode = True

            # pass the onnx file
            if not os.path.exists(onnx_path):
                print('ONNX file {} not found, please run convert_to_onnx.py first to generate it.'.format(onnx_path))
                exit(0)

            print('Loading ONNX file from path {}...'.format(onnx_path))
            with open(onnx_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
                err = parser.get_error(0)
                if err is not None:
                    print('[ERROR] {}'.format(err))
                    raise IOError('Failed to parse ONNX file')
            print('Completed parsing of ONNX file')

            print('Building an engine from file {}; this may take a while...'.format(onnx_path))
            engine = builder.build_cuda_engine(network)
            print('Completed creating Engine')

            if engine is None:
                print('Can not create Engine')
            else:
                with open(engine_path, 'wb') as f:
                    f.write(engine.serialize())
            return engine

    if os.path.exists(engine_path):
        # If you have created the TensorRT engine, plz load and run.
        print('Loading engine from file {}'.format(engine_path))
        with open(engine_path, 'rb') as f, \
                trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(data_type=data_type)


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    # [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def load_intrinsic_matrix(calib_file, camera_type):
    proj_type = 'P2:' if camera_type is None else 'P{}:'.format(CAMERA_TO_ID[camera_type])
    with open(os.path.join(calib_file), 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for line, row in enumerate(reader):
            if row[0] == proj_type:
                P = row[1:]
                P = [float(i) for i in P]
                P = np.array(P, dtype=np.float32).reshape(3, 4)
                K = P[:3, :3]
                break
    return K, P


def draw_3d_box_on_image(img, prediction, P):
    image = np.asarray(img)
    for p in prediction:
        p = p.numpy()
        p = p.round(4)
        dim = [float(p[6]), float(p[7]), float(p[8])]
        location = [float(p[9]), float(p[10]), float(p[11])]
        rotation_y = float(p[12])
        box_3d = compute_box_3d(dim, location, rotation_y)
        box_2d = project_to_image(box_3d, P)
        image = draw_box_3d(image, box_2d)
    return image


def generate_kitti_3d_detection(prediction, predict_txt):
    with open(predict_txt, 'w', newline='') as f:
        w = csv.writer(f, delimiter=' ', lineterminator='\n')
        if len(prediction) == 0:
            w.writerow([])
        else:
            for p in prediction:
                p = p.numpy()
                p = p.round(4)
                type = ID_TYPE_CONVERSION[int(p[0])]
                row = [type, 0, 0] + p[1:].tolist()
                w.writerow(row)


def run_demo(engine, output_dir):
    output_image_dir = os.path.join(output_dir, 'image')
    output_pred_dir = os.path.join(output_dir, 'prediction')
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_pred_dir):
        os.makedirs(output_pred_dir)

    input_width = cfg.INPUT.WIDTH_TEST
    input_height = cfg.INPUT.HEIGHT_TEST
    output_width, output_height = int(input_width / cfg.MODEL.BACKBONE.DOWN_RATIO), int(
        input_height / cfg.MODEL.BACKBONE.DOWN_RATIO)
    output_shapes = [(1, cfg.MODEL.SMOKE_HEAD.REGRESSION_HEADS, output_height, output_width),
                     (1, len(cfg.DATASETS.DETECT_CLASSES), output_height, output_width)]
    post_processor = make_smoke_post_processor(cfg)
    context = engine.create_execution_context()
    # allocate the buffer of the host device
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    val_list_path = os.path.join(validation_dir, 'ImageSets/val.txt')
    images_dir = os.path.join(validation_dir, 'image_2')
    calibs_dir = os.path.join(validation_dir, 'calib')
    if "waymo720" in dataset_type:
        images_dir = os.path.join(validation_dir, 'image_2', camera_type)
        calibs_dir = os.path.join(validation_dir, 'calib')
        val_list_path = os.path.join(validation_dir, 'ImageSets', 'val_{}.txt'.format(camera_type))

    list_file = open(val_list_path, 'r')
    for idx, image_name in enumerate(tqdm(list_file.readlines())):
        image_name = image_name.strip()
        image_path = os.path.join(images_dir, image_name + '.jpg') if os.path.exists(
            os.path.join(images_dir, image_name + '.jpg')) else os.path.join(images_dir, image_name + '.png')
        calib_path = os.path.join(calibs_dir, image_name + '.txt')
        img_cv = cv2.imread(image_path)
        image = Image.fromarray(img_cv)
        K, P = load_intrinsic_matrix(calib_path, camera_type)
        K_src = K.copy()

        if cfg.INPUT.TEST_AFFINE_TRANSFORM:
            center = np.array([i / 2 for i in image.size], dtype=np.float32)
            size = np.array([i for i in image.size], dtype=np.float32)
            center_size = [center, size]
            trans_affine = get_transfrom_matrix(center_size, [input_width, input_height])
            trans_affine_inv = np.linalg.inv(trans_affine)
            image = image.transform(
                (input_width, input_height),
                method=Image.AFFINE,
                data=trans_affine_inv.flatten()[:6],
                resample=Image.BILINEAR)
        else:
            # Resize the image and change the instric params
            src_width, src_height = image.size
            image = image.resize((input_width, input_height), Image.BICUBIC)
            K[0] = K[0] * input_width / src_width
            K[1] = K[1] * input_height / src_height
            center = np.array([i / 2 for i in image.size], dtype=np.float32)
            size = np.array([i for i in image.size], dtype=np.float32)
            center_size = [center, size]

        trans_mat = get_transfrom_matrix(center_size, [output_width, output_height])
        target = ParamsList(image_size=[src_width, src_height], is_train=False)
        target.add_field('K_src', K_src)
        target.add_field('trans_mat', trans_mat)
        target.add_field('K', K)
        target = [target.to(cfg.MODEL.DEVICE)]
        # transform
        img = F.to_tensor(image)
        img = img[[2, 1, 0]]
        img = img * 255.0

        img = np.array(img.numpy(), dtype=np.float32, order='C')
        inputs[0].host = img
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        ''' 2 postprocess the output of the TensorRT engine'''
        # reshape the label prediction and bbox prediction
        trt_outputs = [torch.from_numpy(output.reshape(shape)) for output, shape in zip(trt_outputs, output_shapes)]
        trt_outputs.reverse()
        trt_outputs = [output.to(cfg.MODEL.DEVICE) for output in trt_outputs]
        prediction = post_processor.forward(trt_outputs, target)

        image = draw_3d_box_on_image(image, prediction.to('cpu'), P)
        cv2.imwrite(os.path.join(output_image_dir, image_name + '.jpg'), image)
        generate_kitti_3d_detection(prediction.to('cpu'), os.path.join(output_pred_dir, image_name + '.txt'))


if __name__ == '__main__':
    '''Create a TensorRT engine for ONNX-based and run inference.'''
    engine = create_trt_engine(onnx_path, engine_path, data_type=trt_data_type)
    run_demo(engine, args.output_dir)
