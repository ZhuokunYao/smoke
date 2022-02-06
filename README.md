# Monocular 3D Object Detection
## SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation
# **合入代码前请确保训练模型的Performance** 
## **configs/smoke_kitti_resnet18_batch8.yaml**
<img align="center" src="figures/animation.gif" width="750">

[Video](https://www.youtube.com/watch?v=pvM_bASOQmo)

This repository is the official implementation of paper [SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation](https://arxiv.org/pdf/2002.10111.pdf).
Github repository: **https://github.com/lzccccc/SMOKE**
The pretrained weights can be downloaded [here](https://drive.google.com/open?id=11VK8_HfR7t0wm-6dCNP5KS3Vh-Qm686-)

## Introduction
SMOKE is a real-time monocular 3D object detector for autonomous driving. 

The following performance on KITTI 3D detection (3D/BEV), which is trained with config file **configs/smoke_kitti_resnet18_batch8.yaml**:

|             |     Easy      |    Moderate    |     Hard     |
|-------------|:-------------:|:--------------:|:------------:|
| Car         | 39.58 / 44.03 | 29.85 / 35.65  | 28.56 / 31.03 | 
| Pedestrian  | 10.89 / 11.57 | 10.22 / 10.99  | 10.11 / 10.33  | 
| Cyclist     | 0.47  / 1.82  | 0.38  / 1.52   | 0.38  / 1.52  |

## Major features
We have finished the following improvements:
*   Support to train different jdx dataset
*   Add script tool to cleanup waymo dataset
*   Support to train waymo dataset with different image resolution
*   Support to training with a specified checkpoint
*   Support to prune and finetune training with pruned Resnet model (ResnetX.py)
*   Support to sparse training based on BN layers
*   Support to transfer pth model to Onnx and TensorRT
*   Support to validate with different bathsize in training process
*   Support to train in Pytorch 1.1 and Pytorch 1.3

## Requirements
All codes are tested under the following environment:
*   Ubuntu 16.04/18.04
*   Python 3.7
*   Pytorch 1.3
*   CUDA 10.0

## Dataset
We train and test our model on official [KITTI 3D Object Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). 
Please first download the dataset and organize it as following structure:
```
kitti
│──training
│    ├──calib 
│    ├──label_2 
│    ├──image_2
│    └──ImageSets
└──testing
     ├──calib 
     ├──image_2
     └──ImageSets
```  

## Setup
1. Setup Conda training environment:
```
conda create -n SMOKE python=3.7
conda install cudatoolkit==10.0.130
conda install pytorch=1.3.1 torchvision -c pytorch
or
pip install torch==1.3.1+cu100 torchvision==0.4.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install scikit-build opencv-python yacs scikit-image tqdm numba fire pybind11
sudo apt-get install build-essential
sudo apt-get install libboost-python-dev
```
If you want to use "WeightedDisL1", please install following packages:
(1) install mmcv
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple/ # package mmcv-full will be installed after this step

(2) install mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
mmdetection/mmdet/version.py中的__version__ =中的版本呢改为：__version__==2.11.0
pip install -r requirements/build.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install -v -e . -i https://pypi.tuna.tsinghua.edu.cn/simple/

(3) install mmsegmentation
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple/

(4) install mmdetection3d
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -v -e . -i https://pypi.tuna.tsinghua.edu.cn/simple/

Q: ImportError: libGL.so.1: cannot open shared object file
A: sudo apt install libgl1-mesa-glx
Q: ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject
A: pip install pycocotools==2.0.0

IF you meet the following errors:
(1) "... fatal error: cuda_runtime_api.h: No such file or directory ...", you would reinstall cuda, like "./YourPath/cuda_10.0.130_410.48_linux.run"
(2) "... /usr/bin/ld: cannot find -lcublas or /usr/bin/ld: cannot find -lcudart ...", you would build the soft chain like "ln -s /usr/local/cuda/lib64/libcublas.so.10.0 /usr/local/cuda/lib64/libcublas.so"

2. Clone this repo:
```
git clone git@git.jd.com:xinyu.xu/dnn_jdx.git
git clone https://git.jd.com/anyaozu/monocular3d
```

3. Build codes:
```
python setup.py build develop
```

4. Link to dataset directory:
MCP的训练数据集说明参照CF：https://cf.jd.com/pages/viewpage.action?pageId=421775547
```
mkdir datasets
ln -s /path_to_kitti_dataset datasets/kitti
ln -s /path_to_waymo720_front_dataset/ datasets/wayno720_front
ln -s /path_to_waymo720_dataset/ datasets/wayno720
ln -s /path_to_jdx2020_tracker_dataset datasets/jdx2020_tracker
ln -s /path_to_jdx2021_tracker_dataset datasets/jdx2021_tracker
ln -s /path_to_jdx2021_fusion_dataset datasets/jdx2021_fusion
ln -s /path_to_jdx_test_day_dataset datasets/jdx_test_day
ln -s /path_to_jdx_test_night_dataset datasets/jdx_test_night
```

## Getting started
First check the config file under `configs/`. 

We train the model on 4 GPUs with 32 batch size:
```
python tools/train_smoke.py --num-gpus 4 --config-file configs/smoke_kitti_benchmark.yaml
```

For single GPU training, simply run:
```
python tools/train_smoke.py --config-file configs/smoke_kitti_benchmark.yaml
```

For GPU testing:
```
python tools/evaluate.py --config-file configs/smoke_kitti_benchmark.yaml --model_path checkpoint/model_best.pth
```

## Model Convertion
### Setup ONNX and TensorRT5.0 conversion environment
```
Install Conda environment for Onnx conversion
    conda create -n pytorch1.1 python=3.7
    conda install cudatoolkit==10.0.130
    conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
    conda install numba
    pip install scikit-build opencv-python yacs scikit-image pybind11 onnxruntime pycuda
    sudo apt-get install build-essential
    sudo apt-get install libboost-python-dev

Install Conda environment for TensorRT5.0 conversion
    conda create -n trt5 python=3.5
    conda install cudatoolkit==10.0.130
    conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
    pip install scikit-build scikit-image opencv-python tqdm pycuda
    pip install python/tensorrt-5.0.2.6-py2.py3-none-any.whl
    pip install graphsurgeon-0.3.2-py2.py3-none-any.whl
    # 拷贝lib下库文件到conda环境的对应目录
    cp lib/* /your_conda_env_path/lib/python3.5/site-packages/tensorrt
```
### Convert to ONNX
```
python tools/tensorrt/convert_onnx.py --config-file configs/smoke_kitti_benchmark.yaml --model_path checkpoint/model_best.pth --onnx_path checkpoint/model_best.onnx
```
### Convert to TensorRT
```
python tools/tensorrt/convert_tensorrt.py
```

## Model Evaluation
```
python tools/pykitti_eval/kitti_eval_tools.py --gt_label_path datasets/kitti/training/label_2 --pred_label_path pred/kitti_val
```
## Model visualization
```
python tools/demo.py
```
## Model prunning base BN layers
### Reserve the output channels of each stage
```
python tools/prunning/prune_smoke_resnet_v1.py
```

### Prune the skip layers and change output channels of each stage
```
python tools/prunning/prune_smoke_resnet.py
```
## Calculate the pram numbers and FLOPS
```
python tools/compute_flops.py
```

## tools of dealing with waymo dataset, generate list, calculate the number of classes, cleanup the source waymo
```
python tools/dealwith_dataset.py
```

## visualize the GT of kitti, jdx, waymo dataset
```
python tools/check_visualization.py
```

## visualize val result by tensorboard during trainning
sh tools/tensorboard.sh