import os
import csv
import logging
from tqdm import tqdm
from smoke.utils.imports import import_file
import torch

from smoke.utils.miscellaneous import mkdir
#from smoke.utils import comm
from smoke.utils.timer import Timer, get_time_str
from smoke.data.datasets.evaluation import evaluate

def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch["images"], batch["targets"], batch["img_ids"]
        images = images.tensors.to(device)
        targets = [target.to(device) for target in targets]
        #print(images.shape[0])     # 1
        trans_mat = torch.stack([t.get_field("trans_mat") for t in targets]).to(device)
        K = torch.stack([t.get_field("K") for t in targets]).to(device)
        K_src = torch.stack([t.get_field("K_src") for t in targets]).to(device)
        size = torch.stack([torch.tensor(t.size) for t in targets]).to(device)
        
        target_dict = {'trans_mat':trans_mat,  'K':K,  'K_src':K_src, 'size':size}
        with torch.no_grad():
            if timer:
                timer.tic()
            output = model(images, target_dict)
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = output.to(cpu_device)
            #print(output)
        results_dict.update(
            {img_id: output for img_id in image_ids}
        )
    return results_dict

def write_kitti_3d_detection(predictions, output_folder, class_to_name):
    for image_id, prediction in predictions.items():
        predict_txt = image_id + '.txt'
        predict_txt = os.path.join(output_folder, predict_txt)
        generate_kitti_3d_detection(prediction, predict_txt, class_to_name)


def generate_kitti_3d_detection(prediction, predict_txt, class_to_name):
    with open(predict_txt, 'w', newline='') as f:
        w = csv.writer(f, delimiter=' ', lineterminator='\n')
        if len(prediction) == 0:
            w.writerow([])
        else:
            for p in prediction:
                p = p.numpy()
                p = p.round(4)
                type = class_to_name[int(p[0])]
                row = [type, 0, 0] + p[1:].tolist()
                w.writerow(row)

    check_last_line_break(predict_txt)


def check_last_line_break(predict_txt):
    f = open(predict_txt, 'rb+')
    try:
        f.seek(-1, os.SEEK_END)
    except:
        pass
    else:
        if f.__next__() == b'\n':
            f.seek(-1, os.SEEK_END)
            f.truncate()
    f.close()


def inference(
        cfg,
        model,
        data_loader,
        dataset_name,
        eval_types=("detections",),
        device="cuda",
        output_folder=None,

):
    class_to_name = {}
    for idx, cls in enumerate(cfg.DATASETS.DETECT_CLASSES):
        class_to_name[idx] = cls
    device = torch.device(device)
    #num_devices = comm.get_world_size()
    logger = logging.getLogger(__name__)
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))

    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    # results_dict:  {img_id: output}
    # output:  torch.cat([clses, pred_alphas, box2d, pred_dimensions, 
    #                     pred_locations, pred_rotys, scores], dim=1)
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    # write to output txt
    write_kitti_3d_detection(predictions, output_folder, class_to_name)
    #comm.synchronize()

    #total_time = total_timer.toc()
    #total_time_str = get_time_str(total_time)
    #logger.info(
    #    "Total run time: {} ({} s / img per device, on {} devices)".format(
    #        total_time_str, total_time * num_devices / len(dataset), num_devices
    #    )
    #)
    #total_infer_time = get_time_str(inference_timer.total_time)
    #logger.info(
    #    "Model inference time: {} ({} s / img per device, on {} devices)".format(
    #        total_infer_time,
    #        inference_timer.total_time * num_devices / len(dataset),
    #        num_devices,
    #    )
    #)
    #if not comm.is_main_process():
    #    return None

    return evaluate(cfg=cfg,
                    eval_type=eval_types,
                    dataset=dataset,
                    dataset_name = dataset_name)
