import numpy as np
import cv2
import os
import csv

TYPE_ID_CONVERSION = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
    'Truck': 3,
    'Tricycle': 4,
    'Bus': 5,
    'Motobike': 6
}
CAMERA_TO_ID = {
    'front': 0,
    'front_left': 1,
    'front_right': 2,
    'side_left': 3,
    'side_right': 4,
}
classes = ['Car', 'Cyclist', 'Pedestrian', 'Truck', 'Tricycle', 'Bus']
label_dir = '../datasets/waymo720_front/train/label_2/front'
image_dir = '../datasets/waymo720_front/train/image_2/front'
calib_dir = '../datasets/waymo720_front/train/calib'
imageset_txt = '../datasets/waymo720_front/train/ImageSets/train_front.txt'
distance_threshold = 80
camera = 'front'

def encode_label(K, ry, dims, locs):
    l, h, w = dims[0], dims[1], dims[2]
    x, y, z = locs[0], locs[1], locs[2]

    x_corners = [0, l, l, l, l, 0, 0, 0]
    y_corners = [0, 0, h, h, 0, 0, h, h]
    z_corners = [0, 0, 0, w, w, w, w, 0]

    x_corners += - np.float32(l) / 2
    y_corners += - np.float32(h)
    z_corners += - np.float32(w) / 2

    corners_3d = np.array([x_corners, y_corners, z_corners])
    rot_mat = np.array([[np.cos(ry), 0, np.sin(ry)],
                        [0, 1, 0],
                        [-np.sin(ry), 0, np.cos(ry)]])
    corners_3d = np.matmul(rot_mat, corners_3d)
    corners_3d += np.array([x, y, z]).reshape([3, 1])

    loc_center = np.array([x, y - h / 2, z])
    proj_point = np.matmul(K, loc_center)
    proj_point = proj_point[:2] / proj_point[2]

    corners_2d = np.matmul(K, corners_3d)
    corners_2d = corners_2d[:2] / corners_2d[2]
    box2d = np.array([min(corners_2d[0]), min(corners_2d[1]),
                      max(corners_2d[0]), max(corners_2d[1])])
    
    # proj_point: 3D center projected into 2D
    # box2d: xmin, ymin, xmax, ymax
    # corners_3d: 3*8 
    return proj_point, box2d, corners_3d

def dist(p1, p2):
    return np.sqrt ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)

def load_annotations(file_name):
        annotations = []
        fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                      'dl', 'lx', 'ly', 'lz', 'ry']

        with open(os.path.join(label_dir, file_name), 'r') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
            for line, row in enumerate(reader):
                if row["type"] in classes and float(row['lz']) < distance_threshold:
                    annotations.append({
                        "class": row["type"],
                        "label": TYPE_ID_CONVERSION[row["type"]],
                        "truncation": float(row["truncated"]),
                        "occlusion": float(row["occluded"]),
                        "alpha": float(row["alpha"]),
                        "dimensions": [float(row['dl']), float(row['dh']), float(row['dw'])],
                        "locations": [float(row['lx']), float(row['ly']), float(row['lz'])],
                        "rot_y": float(row["ry"]),
                        "xmin": row["xmin"],
                        "ymin": row["ymin"],
                        "xmax": row["xmax"],
                        "ymax": row["ymax"]
                    })

        # get camera intrinsic matrix K
        proj_type = 'P{}:'.format(CAMERA_TO_ID[camera])
        with open(os.path.join(calib_dir, file_name), 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=' ')
            for line, row in enumerate(reader):
                ### lyb:add
                if not row:
                    continue
                if row[0] == proj_type:
                    P = row[1:]
                    P = [float(i) for i in P]
                    P = np.array(P, dtype=np.float32).reshape(3, 4)
                    K = P[:3, :3]
        return annotations, K

image_files = []
label_files = []
anno_dicts = []
K_dicts = []

num_control = 0
for line in open(imageset_txt, "r"):
    image_name = line.strip() + ".png" if os.path.exists(
            os.path.join(image_dir, line.strip() + ".png")) else line.strip() + ".jpg"
    label_file = line.strip() + ".txt"
    anns, K = load_annotations(label_file)
    if anns:
        image_files.append(image_name)
        label_files.append(label_file)
        anno_dicts.append(anns)
        K_dicts.append(K)
    num_control += 1
    if(num_control>=100):
        break
#print(len(image_files),len(label_files),len(anno_dicts),len(K_dicts))

for image_name,anno,K in zip(image_files,anno_dicts,K_dicts):
    print(f'image: {os.path.basename(image_name)}: ')
    for ann in anno:
        alpha = float(ann["alpha"])

        dimensions = np.array(ann["dimensions"])
        lx = dimensions[0]
        ly = dimensions[1]
        lz = dimensions[2]

        locations = np.array(ann["locations"])
        x = locations[0]
        y = locations[1]
        z = locations[2]

        rot_y = np.array(ann["rot_y"])

        xmin = float(ann["xmin"])
        ymin = float(ann["ymin"])
        xmax = float(ann["xmax"])
        ymax = float(ann["ymax"])

        # proj_point: 3D center projected into 2D  (float)
        # box2d: xmin, ymin, xmax, ymax
        # corners_3d: 3*8
        point, box2d, box3d = encode_label(K, rot_y, dimensions, locations)
        #print(f'{xmin:.0f},{ymin:.0f},{xmax:.0f},{ymax:.0f}     {box2d[0]:.0f},{box2d[1]:.0f},{box2d[2]:.0f},{box2d[3]:.0f}')
        p1 = np.array([box3d[0,0], box3d[1,0], box3d[2,0]])
        p2 = np.array([box3d[0,1], box3d[1,1], box3d[2,1]])
        p3 = np.array([box3d[0,2], box3d[1,2], box3d[2,2]])
        p4 = np.array([box3d[0,3], box3d[1,3], box3d[2,3]])
        p5 = np.array([box3d[0,4], box3d[1,4], box3d[2,4]])
        p6 = np.array([box3d[0,5], box3d[1,5], box3d[2,5]])
        p7 = np.array([box3d[0,6], box3d[1,6], box3d[2,6]])
        p8 = np.array([box3d[0,7], box3d[1,7], box3d[2,7]])

        #print(p1[2]-p8[2])
        #print(f'{dist(p1,p6):.2f},{dist(p7,p8):.2f},{dist(p3,p4):.2f},{dist(p2,p5):.2f} {lz:.2f}')
        #print(f'{dist(p1,p2):.2f},{dist(p3,p8):.2f},{dist(p4,p7):.2f},{dist(p5,p6):.2f} {lx:.2f}')

        box_conner_in_2d = np.matmul(K, box3d)
        # 2*8
        box_conner_in_2d = box_conner_in_2d[:2] / box_conner_in_2d[2]

        p1_2d = np.array([box_conner_in_2d[0,0], box_conner_in_2d[1,0]])
        p2_2d = np.array([box_conner_in_2d[0,1], box_conner_in_2d[1,1]])
        p3_2d = np.array([box_conner_in_2d[0,2], box_conner_in_2d[1,2]])
        p4_2d = np.array([box_conner_in_2d[0,3], box_conner_in_2d[1,3]])
        p5_2d = np.array([box_conner_in_2d[0,4], box_conner_in_2d[1,4]])
        p6_2d = np.array([box_conner_in_2d[0,5], box_conner_in_2d[1,5]])
        p7_2d = np.array([box_conner_in_2d[0,6], box_conner_in_2d[1,6]])
        p8_2d = np.array([box_conner_in_2d[0,7], box_conner_in_2d[1,7]])

        dy = np.array([p8_2d[1]-p1_2d[1], p3_2d[1]-p2_2d[1], p4_2d[1]-p5_2d[1], p7_2d[1]-p6_2d[1]])
        f  = np.array([dy[0]*p8[2]/lz, dy[1]*p3[2]/lz, dy[2]*p4[2]/lz, dy[3]*p7[2]/lz])
        
        focal_ref = 720.0
        dz_fenzhiyi = f[0]/focal_ref
        dy_fenzhiyi = K[1,1]/focal_ref
        dx_fenzhiyi = K[0,0]/focal_ref
        print(dz_fenzhiyi/dx_fenzhiyi)
        
        
        #print(f'{dy[0]:.2f}, {dy[1]:.2f}, {dy[2]:.2f}, {dy[3]:.2f}')
        #print(f'{f[0]:.2f}, {f[1]:.2f}, {f[2]:.2f}, {f[3]:.2f}, {K[1,1]:.2f}, {K[0,0]:.2f}')