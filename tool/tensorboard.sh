### example
MODEL_DIR="./checkpoint/waymo720front_balancedclass_resnet18_704x384_batch40_sampleratio1_lr/logs"
tensorboard --logdir=${MODEL_DIR} --port=6009
