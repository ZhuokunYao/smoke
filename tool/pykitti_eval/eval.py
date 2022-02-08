import io as sysio
import numba
import numpy as np

from tool.pykitti_eval.non_max_suppression.nms_gpu import rotate_iou_gpu_eval
import warnings

warnings.filterwarnings("ignore")


# scores:               [tp1+tp2+... for all imgs] (scores)
# total_num_valid_gt:   total number of valid gts of all images
@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    # print(len(thresholds), len(scores), num_gt)
    return thresholds

# a function for single image
#[all in list] name,truncated,occluded,alpha,bbox(4),dimensions(l,h,w),location(3)
#[all in list] rotation_y,score,index(one of 0,1,2,...numobject-1),group_ids(same with index)
#[all in list] image_idx
def clean_data(gt_anno, dt_anno, current_class_name, depth_range=[0, 80]):
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        if gt_anno["name"][i] == current_class_name:
            valid_class = 1
        else:
            valid_class = 0

        ignore = False
        depth = gt_anno["location"][i][2]
        if depth < depth_range[0] or depth >= depth_range[1]:
            ignore = True

        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        else:
            ignored_gt.append(1)

    for i in range(num_dt):
        if dt_anno["name"][i] == current_class_name:
            valid_class = 1
        else:
            valid_class = 0

        ignore = False
        depth = dt_anno["location"][i][2]
        if depth <= depth_range[0] or depth > depth_range[1]:
            ignore = True

        if valid_class == 1 and not ignore:
            ignored_dt.append(0)
        else:
            ignored_dt.append(1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


@numba.jit(nopython=True)
# [[N1,4],[N2,4],...,[NX,4]] -----> [N,4]
#                     gt     pred
def image_box_overlap(boxes, query_boxes, criterion=-1):
    # gt box nums
    N = boxes.shape[0]
    # pred box nums
    K = query_boxes.shape[0]
    # N * K
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        # (xmax-xmin)*(ymax-ymin)  of one pred
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                                (boxes[n, 2] - boxes[n, 0]) *
                                (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    # N * K   iou matrix
    # N = gt box nums
    # K = pred box nums
    return overlaps

# [N, 5] (x,z,lx,lz,roty)
def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True, parallel=True)
# boxes    [N,7]  (x,y,z,lx,ly,lz,roty)
# qboxes   [K,7]
# rinc     [N,K]   xz plane rotate inter_area
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lider.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                # y is top car!!
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = 1.0
                    rinc[i, j] = inc / ua if ua > 1e-5 else 0
                else:
                    rinc[i, j] = 0.0

#   [N,7]  (x,y,z,lx,ly,lz,roty)
#   [K,7]
def d3_box_overlap(boxes, qboxes, criterion=-1):
    # [N,K]  area_inter of x,z,roty
    # the normal(criterion=-1) means "area_inter / (area1 + area2 - area_inter)"
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,           # iou matrix [num_pred_box , num_gt_box]
                           gt_datas,           # [gt_nums, 5]   2d bbox & alpha 
                           dt_datas,           # [dt_nums, 6]   2d bbox & alpha & score
                           ignored_gt,         # [0/1] repeat gt_nums times
                           ignored_det,        # [0/1] repeat dt_nums times 
                           dc_bboxes,          # np.zeros((0, 4))
                           metric,             # 0/1/2
                           min_overlap,        # 0.7/0.25 for class
                           thresh=0,           # 0.0
                           compute_fp=False,   
                           compute_aos=False):
    det_size = dt_datas.shape[0]    # dt_nums
    gt_size = gt_datas.shape[0]     # gt_nums
    dt_scores = dt_datas[:, -1]     # [dt_nums]      score
    dt_alphas = dt_datas[:, 4]      # [dt_nums]      alpha
    gt_alphas = gt_datas[:, 4]      # [gt_nums]      alpha
    dt_bboxes = dt_datas[:, :4]     # [dt_nums, 4]   2d bbox
    # gt_bboxes = gt_datas[:, :4]
    
    # [dt_nums]  true or false
    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    
    # if compute_fp filter dt by score & thresh  --->  ignored_threshold
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
                
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    
    # [gt_nums]
    thresholds = np.zeros((gt_size,))
    thresh_idx = 0
    # [gt_nums]
    delta = np.zeros((gt_size,))
    delta_idx = 0
    
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            # if compute_fp filter dt by score & thresh
            if (ignored_threshold[j]):
                continue
            # overlaps: iou matrix [num_pred_box , num_gt_box]
            # ????
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            # search the most match det
            
            ##### not compute_fp:
            #have ious > thresh  ---->  assigned to the max score (valid_detection=max_score)
            #not                 ---->  det_idx = -1 (valid_detection=NO_DETECTION)
            
            ##### compute_fp:
            #in class-matched dets, there are ious  > thresh ----> assigned to the max iou (valid_detection=1)
            #else
            #in class-dismatched dets, there are ious  > thresh ----> assigned to one of them (valid_detection=1)
            #else
            #---->  det_idx = -1 (valid_detection=NO_DETECTION)
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True
        
        # GT match the class
        #           assigned to a pred:
        #                       assigned to class-matched pred:    tp+1, set threshold(for gt), set assigned_detection(for pred)
        #                       assigned to class-mismatched pred: set assigned_detection(for pred)
        #           not assigned to pred:                          fn+1
        
        # GT missmatch the class
        #           assigned to a pred:                            set assigned_detection(for pred)
        #           not assigned to pred:                          dont handle
        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            # false negative
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            # only a tp add a threshold.
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                # delta.append(gt_alphas[i] - dt_alphas[det_idx])
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            # for a class-matched pred
            # if not assigned to a gt  &  score>thresh
            # fp + 1
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if (assigned_detection[j]):
                        continue
                    if (ignored_det[j] == -1 or ignored_det[j] == 1):
                        continue
                    if (ignored_threshold[j]):
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        fp -= nstuff
        if compute_aos:
            # [fp+tp]
            tmp = np.zeros((fp + delta_idx,))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
            # similarity:  tp's alpha residual lower, similarity bigger
    return tp, fp, fn, similarity, thresholds[:thresh_idx]

#num = num_samples(number of imgs)    num_part=50
def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    elif same_part == 0:
        return [remain_num]
    else:
        return [same_part] * num_part + [remain_num]
    #[num // num_part, num // num_part, ... repeat num_part times, num % num_part]

@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,           # list of iou [K, N] (per a part)
                             pr,                 # np.zeros([len(thresholds), 4])
                             gt_nums,            # [num_of_gt, num_of_gt, ..., num_of_gt]
                             dt_nums,            # [num_of_det, num_of_det, ..., num_of_det]
                             dc_nums,            # [0, 0, ..., 0]
                             gt_datas,           # [gt_nums * num_part, 5]   2d bbox & alpha
                             dt_datas,           # [dt_nums * num_part, 6]   2d bbox & alpha & score
                             dontcares,
                             ignored_gts,        # "[0/1] repeat gt_nums*numpart times"
                             ignored_dets,       # "[0/1] repeat dt_nums*numpart times"
                             metric,             # 0/1/2
                             min_overlap,        # class specified
                             thresholds,         # get some top scores
                             compute_aos=False): # true or false
    gt_num = 0
    dt_num = 0
    dc_num = 0
    #                 num_imgs
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            # iou matrix [Kdt, Ngt] for a image
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:
                                                           gt_num + gt_nums[i]]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,   #  declare  thresh  there!!!!!!!!!
                compute_fp=True,
                compute_aos=compute_aos)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]
    #  pr: np.zeros([len(thresholds), 4])
    #  tp, fp, fn, similarity

        #annos:
        #list of (per image)
        # for each object
        #[all in list] name,truncated,occluded,alpha,bbox(4),dimensions(l,h,w),location(3)
        #[all in list] rotation_y,score,index(one of 0,1,2,...numobject-1),group_ids(same with index)
        #[all in list] image_idx
def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """fast iou algorithm. this function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
    """
    assert len(gt_annos) == len(dt_annos)
    # [num_of_det, num_of_det, ..., num_of_det] for each image
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    # [num_of_gt, num_of_gt, ..., num_of_gt] for each image
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    # image numbers
    num_examples = len(gt_annos)
    #[num // num_part, num // num_part, ... repeat num_part times, num % num_part]
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0
    
    # num_part : image numbers to do eval
    for num_part in split_parts:
        # slice
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 0:
            # [[N1,4],[N2,4],...,[NX,4]] -----> [N,4]
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            # [[K1,4],[K2,4],...,[KX,4]] -----> [K,4]
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            # N * K   iou matrix
            # N = gt box nums
            # K = pred box nums
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate(
                # [[N1,2],[N2,2],...,[NX,2]] -----> [N,2]   (x,z)
                [a["location"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate(
                # [[N1,2],[N2,2],...,[NX,2]] -----> [N,2]  (lx,lz)
                [a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
            # [N] (roty)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            # [N, 5] (x,z,lx,lz,roty)
            
            # N * K   iou matrix
            # N = gt box nums
            # K = pred box nums
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            #   [N,7]  (x,y,z,lx,ly,lz,roty)
            
            # N * K   iou matrix
            # N = gt box nums
            # K = pred box nums
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
            
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            # per image
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part
    #overlaps:  list of iou [num_gt_box , num_pred_box] (per image)
    #parted_overlaps:  list of iou [N, K] (per part)
    #total_gt_num:  [num_of_gt, num_of_gt, ..., num_of_gt] for each image
    #total_dt_num:  [num_of_det, num_of_det, ..., num_of_det] for each image
    return overlaps, parted_overlaps, total_gt_num, total_dt_num

#annos:
#list of (per image)
# for each object
#[all in list] name,truncated,occluded,alpha,bbox(4),dimensions(l,h,w),location(3)
#[all in list] rotation_y,score,index(one of 0,1,2,...numobject-1),group_ids(same with index)
#[all in list] image_idx

#                                         Car 
def _prepare_data(gt_annos, dt_annos, current_class, depth_range):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    #                img_num
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, depth_range)
        # filter by class & depth_range
        #             [o/1]       [0/1]        []
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        # np.zeros((0, 4))
        
        # 0
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        # [gt_nums, 5]   2d bbox & alpha
        gt_datas = np.concatenate(
            [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1)
        # [dt_nums, 6]   2d bbox & alpha & score
        dt_datas = np.concatenate([
            dt_annos[i]["bbox"], dt_annos[i]["alpha"][..., np.newaxis],
            dt_annos[i]["score"][..., np.newaxis]
        ], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    # [0,0,...] repeat img_num times
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)
    # gt_datas_list:  list of "[gt_nums, 5]   2d bbox & alpha"          per image
    # dt_datas_list:  list of "[dt_nums, 5]   2d bbox & alpha & score"  per image
    
    # filter by class & depth range
    # ignored_gts:    list of "[0/1]" repeat gt_nums times              per image
    # ignored_dets:   list of "[0/1]" repeat st_nums times              per image
    # total_num_valid_gt:   total number of valid gts of all images
    # dontcares:      list of "np.zeros((0, 4))"                        per image
    # total_dc_num:   [0, 0, ...]     repeat  img_num times

def eval_class(gt_annos,
               dt_annos,
               class_id_name,
               metric,     # 0/1/2
               min_overlaps,
               compute_aos=False,
               num_parts=50,
               depth_range=[0, 80]):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_class: int, 0: car, 1: pedestrian, 2: cyclist
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlap: [0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.5]
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    # [num // num_part, num // num_part, ... repeat num_part times, num % num_part]
    split_parts = get_split_parts(num_examples, num_parts)
    
    #overlaps:  list of iou [num_pred_box , num_gt_box] (per image)
    #parted_overlaps:  list of iou [K, N] (per part)
    #total_gt_num:  [num_of_gt, num_of_gt, ..., num_of_gt] for each image
    #total_dt_num:  [num_of_det, num_of_det, ..., num_of_det] for each image
    #
    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    # ????
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    N_SAMPLE_PTS = 41
    num_class = len(class_id_name)
    # 6, 41
    precision = np.zeros([num_class, N_SAMPLE_PTS])
    recall = np.zeros([num_class, N_SAMPLE_PTS])
    aos = np.zeros([num_class, N_SAMPLE_PTS])
    for id, name in class_id_name.items():
        # gt_datas_list:  list of "[gt_nums, 5]   2d bbox & alpha"          per image
        # dt_datas_list:  list of "[dt_nums, 6]   2d bbox & alpha & score"  per image
    
        # filter by class & depth range
        # ignored_gts:    list of "[0/1] repeat gt_nums times"              per image
        # ignored_dets:   list of "[0/1] repeat dt_nums times"              per image
        # total_num_valid_gt:   total number of valid gts of all images
        # dontcares:      list of "np.zeros((0, 4))"                        per image
        # total_dc_num:   [0, 0, ...]     repeat  img_num times
        rets = _prepare_data(gt_annos, dt_annos, name, depth_range)
        (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
         dontcares, total_dc_num, total_num_valid_gt) = rets
        thresholdss = []
        for i in range(len(gt_annos)):
            # for one image
            # return tp, fp, fn, similarity, thresholds[:thresh_idx]
            # similarity:  tp's alpha residual lower, similarity bigger
            # thresholds: [tp] (scores)
            rets = compute_statistics_jit(
                overlaps[i],        # iou matrix [num_pred_box , num_gt_box]
                gt_datas_list[i],   # [gt_nums, 5]   2d bbox & alpha
                dt_datas_list[i],   # [dt_nums, 6]   2d bbox & alpha & score
                ignored_gts[i],     # [0/1] repeat gt_nums times
                ignored_dets[i],    # [0/1] repeat dt_nums times
                dontcares[i],       # np.zeros((0, 4))
                metric,             # 0/1/2
                min_overlap=min_overlaps[id],  #0.7/0.25 for class
                thresh=0.0,
                compute_fp=False)
            tp, fp, fn, similarity, thresholds = rets
            thresholdss += thresholds.tolist()
        #[tp1+tp2+... for all imgs] (scores)
        thresholdss = np.array(thresholdss)
        # total_num_valid_gt:   total number of valid gts of all images
        thresholds = get_thresholds(thresholdss, total_num_valid_gt)
        # get some top scores
        thresholds = np.array(thresholds)
        pr = np.zeros([len(thresholds), 4])
        idx = 0
        # [num // num_part, num // num_part, ... repeat num_part times, num % num_part]
        for j, num_part in enumerate(split_parts):
            # [gt_nums * num_part, 5]   2d bbox & alpha
            gt_datas_part = np.concatenate(
                gt_datas_list[idx:idx + num_part], 0)
            # [dt_nums * num_part, 6]   2d bbox & alpha & score
            dt_datas_part = np.concatenate(
                dt_datas_list[idx:idx + num_part], 0)
            dc_datas_part = np.concatenate(
                dontcares[idx:idx + num_part], 0)
            # "[0/1] repeat dt_nums*numpart times"
            ignored_dets_part = np.concatenate(
                ignored_dets[idx:idx + num_part], 0)
            # "[0/1] repeat gt_nums*numpart times"
            ignored_gts_part = np.concatenate(
                ignored_gts[idx:idx + num_part], 0)
            fused_compute_statistics(
                parted_overlaps[j],               # list of iou [K, N]
                pr,                               # np.zeros([len(thresholds), 4])
                total_gt_num[idx:idx + num_part], # [num_of_gt, num_of_gt, ..., num_of_gt]
                total_dt_num[idx:idx + num_part], # [num_of_det, num_of_det, ..., num_of_det]
                total_dc_num[idx:idx + num_part], # [0, 0, ..., 0]
                gt_datas_part,                    # [gt_nums * num_part, 5]   2d bbox & alpha
                dt_datas_part,                    # [dt_nums * num_part, 6]   2d bbox & alpha & score
                dc_datas_part,
                ignored_gts_part,                 # "[0/1] repeat gt_nums*numpart times"
                ignored_dets_part,                # "[0/1] repeat dt_nums*numpart times"
                metric,                           # 0/1/2
                min_overlap=min_overlaps[id],     # class specified
                thresholds=thresholds,            # get some top scores
                compute_aos=compute_aos)          # true or false
            idx += num_part
        #  pr: np.zeros([len(thresholds), 4])
        #  tp, fp, fn, similarity
        for i in range(len(thresholds)):
            recall[id, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
            precision[id, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
            if compute_aos:
                aos[id, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
        #precision = np.zeros([num_class, N_SAMPLE_PTS])
        #recall = np.zeros([num_class, N_SAMPLE_PTS])
        #aos = np.zeros([num_class, N_SAMPLE_PTS])
        for i in range(len(thresholds)):
            precision[id, i] = np.max(precision[id, i:], axis=-1)
            recall[id, i] = np.max(recall[id, i:], axis=-1)
            if compute_aos:
                aos[id, i] = np.max(aos[id, i:], axis=-1)
    ret_dict = {
        "recall": recall,
        "precision": precision,
        "orientation": aos,
    }
    return ret_dict


def check_aos(annos):
    compute_aos = False
    for anno in annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    return compute_aos

# prec: num_class, N_SAMPLE_PTS
def get_average(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


def do_eval(gt_annos,
            dt_annos,
            class_id_name, #{ 0:"car"  ,   ... }
            min_overlaps,  # [0.7, 0.25, 0.25, 0.7, 0.25, 0.7]
            compute_aos=False, # True or False
            depth_range=[0, 80]):
    # {"recall": recall, "precision": precision, "orientation": aos}
    # np.zeros([num_class, N_SAMPLE_PTS])
    ret = eval_class(gt_annos, dt_annos, class_id_name, 0,
                     min_overlaps, compute_aos, num_parts=50, depth_range=depth_range)
    APbbox = get_average(ret["precision"])
    ARbbox = get_average(ret["recall"])
    APaos = get_average(ret["orientation"]) if compute_aos else None

    ret = eval_class(gt_annos, dt_annos, class_id_name, 1,
                     min_overlaps, num_parts=50, depth_range=depth_range)
    APbev = get_average(ret["precision"])
    ARbev = get_average(ret['recall'])

    ret = eval_class(gt_annos, dt_annos, class_id_name, 2,
                     min_overlaps, num_parts=50, depth_range=depth_range)
    AP3d = get_average(ret["precision"])
    AR3d = get_average(ret["recall"])

    return APbbox, APbev, AP3d, APaos, ARbbox, ARbev, AR3d

        #annos:
        #list of (per image)
        # for each object
        #[all in list] name,truncated,occluded,alpha,bbox(4),dimensions(l,h,w),location(3)
        #[all in list] rotation_y,score,index(one of 0,1,2,...numobject-1),group_ids(same with index)
        #[all in list] image_idx
def evaluate_mAP(gt_annos, dt_annos, class_name, depth_ranges=[[0, 30], [0, 60]],
                 iou_thresholds=[0.7, 0.25, 0.25, 0.7, 0.25, 0.7]):
    min_overlaps = np.stack(iou_thresholds)
    # min_overlaps = np.stack([0.7, 0.25, 0.25, 0.7, 0.25, 0.7])
    # min_overlaps = np.stack([0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.5])
    # min_overlaps = np.stack([0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.25])
    if not isinstance(class_name, (list, tuple)):
        class_name = [class_name]
    class_id_name = {idx: cls for idx, cls in enumerate(class_name)}
    compute_aos = check_aos(dt_annos)

    result_dict = {}
    AP_AR_print = ""
    bev_mAP_print = ""
    for i, depth_range in enumerate(depth_ranges):
        # all in np.zeros([num_class])
        APbbox, APbev, AP3d, APaos, ARbbox, ARbev, AR3d = do_eval(
            gt_annos, dt_annos, class_id_name, min_overlaps, compute_aos, depth_range)

        range_str = "range[{},{}]".format(depth_range[0], depth_range[1])

        APprint_temp = format_print(class_id_name, APbbox, APbev, AP3d, APaos, True)
        ARprint_temp = format_print(class_id_name, ARbbox, ARbev, AR3d, None, False)
        AP_AR_print += "{}:\n{}{}\n".format(range_str, APprint_temp, ARprint_temp)
        
        # defination of map
        map_1 = APbev[0]
        map_4 = np.average(APbev[0:4])
        map_5 = (APbev[0] + APbev[1] + APbev[2] + APbev[3] + APbev[5]) / 5.0
        map_all = np.average(APbev)
        bev_mAP_print += "{}:bev:mAP_1: {}    mAP_4: {}   mAP_5: {}    mAP_all: {}\n".format(range_str,
                                                                                             round(map_1, 2),
                                                                                             round(map_4, 2),
                                                                                             round(map_5, 2),
                                                                                             round(map_all, 2))

        result_dict[range_str] = {}
        result_dict[range_str]["APbbox"] = APbbox if APbbox is not None else []
        result_dict[range_str]["APbev"] = APbev if APbev is not None else []
        result_dict[range_str]["AP3d"] = AP3d if AP3d is not None else []
        result_dict[range_str]["APaos"] = APaos if APaos is not None else []
        result_dict[range_str]["ARbbox"] = ARbbox if ARbbox is not None else []
        result_dict[range_str]["ARbev"] = ARbev if ARbev is not None else []
        result_dict[range_str]["AR3d"] = AR3d if AR3d is not None else []
        result_dict[range_str]["bev_mAP"] = [map_1, map_4, map_5, map_all]

    all_print = "{}\n{}".format(bev_mAP_print, AP_AR_print)
    result_dict["print"] = all_print if all_print is not None else ""
    result_dict["class_id_name"] = class_id_name

    return result_dict


def format_print(class_id_name, bbox, bev, ddd, aos=None, is_precision=True):
    rst_cls = "\n{:<12s}".format('Precision') if is_precision else "\n{:<12s}".format('Recall')
    rst_bbox = "{:<12s}".format('BBox')
    rst_bev = "{:<12s}".format('BEV')
    rst_3d = "{:<12s}".format('3D')
    rst_aos = "{:<12s}".format('AOS')
    for id, name in class_id_name.items():
        rst_cls += "{:<12s}".format(name)
        rst_bbox += "{:<12.2f}".format(bbox[id])
        rst_bev += "{:<12.2f}".format(bev[id])
        rst_3d += "{:<12.2f}".format(ddd[id])
        if aos is not None:
            rst_aos += "{:<12.2f}".format(aos[id])
    if aos is not None:
        return "{}\n{}\n{}\n{}\n{}\n".format(rst_cls, rst_bbox, rst_bev, rst_3d, rst_aos)
    else:
        return "{}\n{}\n{}\n{}\n".format(rst_cls, rst_bbox, rst_bev, rst_3d)
