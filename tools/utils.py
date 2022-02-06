from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

def compute_box_3d(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  l, w, h = dim[2], dim[1], dim[0]
  x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
  y_corners = [0,0,0,0,-h,-h,-h,-h]
  z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

  corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
  corners_3d = np.dot(R, corners) 
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
  return corners_3d.transpose(1, 0)

def project_to_image(pts_3d, P):
  # pts_3d: n x 3
  # P: 3 x 4
  # return: n x 2
  pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
  pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
  pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
  # import pdb; pdb.set_trace()
  return pts_2d

def compute_orientation_3d(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 2 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  orientation_3d = np.array([[0, dim[2]], [0, 0], [0, 0]], dtype=np.float32)
  orientation_3d = np.dot(R, orientation_3d)
  orientation_3d = orientation_3d + \
                   np.array(location, dtype=np.float32).reshape(3, 1)
  return orientation_3d.transpose(1, 0)

def draw_box_3d(image, corners, c=(0, 255, 0)):
  face_idx = [[0,1,5,4],
              [1,2,6, 5],
              [2,3,7,6],
              [3,0,4,7]]
  for ind_f in range(3, -1, -1):
    f = face_idx[ind_f]
    for j in range(4):
      cv2.line(image, (int(corners[f[j], 0]), int(corners[f[j], 1])),
               (int(corners[f[(j+1)%4], 0]), int(corners[f[(j+1)%4], 1])), c, 2, lineType=cv2.LINE_AA)
    if ind_f == 0:
      cv2.line(image, (int(corners[f[0], 0]), int(corners[f[0], 1])),
               (int(corners[f[2], 0]), int(corners[f[2], 1])), c, 1, lineType=cv2.LINE_AA)
      cv2.line(image, (int(corners[f[1], 0]), int(corners[f[1], 1])),
               (int(corners[f[3], 0]), int(corners[f[3], 1])), c, 1, lineType=cv2.LINE_AA)
  return image

def unproject_2d_to_3d(pt_2d, depth, P):
  # pts_2d: 2
  # depth: 1
  # P: 3 x 4
  # return: 3
  z = depth - P[2, 3]
  x = (pt_2d[0] * depth - P[0, 3] - P[0, 2] * z) / P[0, 0]
  y = (pt_2d[1] * depth - P[1, 3] - P[1, 2] * z) / P[1, 1]
  pt_3d = np.array([x, y, z], dtype=np.float32)
  return pt_3d

def unproject_2d_to_3d_geometry(center, wh, alpha, dimensions, rotation_y, P):

  bbox = [center[0] - wh[0] / 2, center[1] - wh[1] / 2, 
          center[0] + wh[0] / 2, center[1] + wh[1] / 2]
  
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  l, w, h = dimensions[2], dimensions[1], dimensions[0]

  A = np.zeros((4, 3))
  b = np.zeros((4, 1))
  I = np.identity(3)

  rot_local = get_new_alpha(alpha)

  xmin_candi, xmax_candi, ymin_candi, ymax_candi = box3d_candidate(h, w, l, rot_local, soft_range=8)

  X  = np.bmat([xmin_candi, xmax_candi,
                ymin_candi, ymax_candi])

  X = X.reshape(4,3).T

  for i in range(4):
      matrice = np.bmat([[I, np.matmul(R, X[:,i])], [np.zeros((1,3)), np.ones((1,1))]])
      M = np.matmul(P, matrice)

      if i % 2 == 0:
          A[i, :] = M[0, 0:3] - bbox[i] * M[2, 0:3]
          b[i, :] = M[2, 3] * bbox[i] - M[0, 3]

      else:
          A[i, :] = M[1, 0:3] - bbox[i] * M[2, 0:3]
          b[i, :] = M[2, 3] * bbox[i] - M[1, 3]

  Tran = np.matmul(np.linalg.pinv(A), b)
  pt_3d = np.array([float(np.around(tran, 2)) for tran in Tran], dtype=np.float32)
  return pt_3d

def box3d_candidate(h, w, l, rot_local, soft_range):
  x_corners = [l/2,  l/2,  l/2, l/2, -l/2, -l/2, -l/2, -l/2]
  y_corners = [0,    -h,   0,   -h,  0,    -h,   0,    -h]
  z_corners = [-w/2, -w/2, w/2, w/2, w/2,  w/2,  -w/2, -w/2]

  corners_3d = np.transpose(np.array([x_corners, y_corners, z_corners]))
  point1 = corners_3d[0, :]
  point2 = corners_3d[1, :]
  point3 = corners_3d[2, :]
  point4 = corners_3d[3, :]
  point5 = corners_3d[6, :]
  point6 = corners_3d[7, :]
  point7 = corners_3d[4, :]
  point8 = corners_3d[5, :]

  xmin_candi = xmax_candi = ymin_candi = ymax_candi = 0

  if 0 < rot_local < np.pi / 2:
      xmin_candi = point8
      xmax_candi = point2
      ymin_candi = point2
      ymax_candi = point5

  if np.pi / 2 <= rot_local <= np.pi:
      xmin_candi = point6
      xmax_candi = point4
      ymin_candi = point4
      ymax_candi = point1

  if np.pi < rot_local <= 3 / 2 * np.pi:
      xmin_candi = point2
      xmax_candi = point8
      ymin_candi = point8
      ymax_candi = point1

  if 3 * np.pi / 2 <= rot_local <= 2 * np.pi:
      xmin_candi = point4
      xmax_candi = point6
      ymin_candi = point6
      ymax_candi = point5

  div = soft_range * np.pi / 180
  if 0 < rot_local < div or 2*np.pi-div < rot_local < 2*np.pi:
      xmin_candi = point8
      xmax_candi = point6
      ymin_candi = point6
      ymax_candi = point5

  if np.pi - div < rot_local < np.pi + div:
      xmin_candi = point2
      xmax_candi = point4
      ymin_candi = point8
      ymax_candi = point1

  return xmin_candi, xmax_candi, ymin_candi, ymax_candi

def get_new_alpha(alpha):
    """
    change the range of orientation from [-pi, pi] to [0, 2pi]
    :param alpha: original orientation in KITTI
    :return: new alpha
    """
    new_alpha = float(alpha) + np.pi / 2.
    if new_alpha < 0:
        new_alpha = new_alpha + 2. * np.pi
        # make sure angle lies in [0, 2pi]
    new_alpha = new_alpha - int(new_alpha / (2. * np.pi)) * (2. * np.pi)

    return new_alpha

def alpha2rot_y(alpha, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = alpha + np.arctan2(x - cx, fx)
    if rot_y > np.pi:
      rot_y -= 2 * np.pi
    if rot_y < -np.pi:
      rot_y += 2 * np.pi
    return rot_y

def rot_y2alpha(rot_y, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    alpha = rot_y - np.arctan2(x - cx, fx)
    if alpha > np.pi:
      alpha -= 2 * np.pi
    if alpha < -np.pi:
      alpha += 2 * np.pi
    return alpha


def ddd2locrot(center, alpha, dim, depth, calib):
  # single image
  locations = unproject_2d_to_3d(center, depth, calib)
  locations[1] += dim[0] / 2
  rotation_y = alpha2rot_y(alpha, center[0], calib[0, 2], calib[0, 0])
  return locations, rotation_y

def ddd2locrot_geometry(center, wh, alpha, dim, calib):
  # single image
  rotation_y = alpha2rot_y(alpha, center[0], calib[0, 2], calib[0, 0])
  locations = unproject_2d_to_3d_geometry(center, wh, alpha, dim, rotation_y, calib)
  return locations, rotation_y

def project_3d_bbox(location, dim, rotation_y, calib):
  box_3d = compute_box_3d(dim, location, rotation_y)
  box_2d = project_to_image(box_3d, calib)
  return box_2d


if __name__ == '__main__':
  calib = np.array(
    [[7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02, 4.575831000000e+01],
     [0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02, -3.454157000000e-01],
     [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 4.981016000000e-03]],
    dtype=np.float32)
  alpha = -0.20
  tl = np.array([712.40, 143.00], dtype=np.float32)
  br = np.array([810.73, 307.92], dtype=np.float32)
  ct = (tl + br) / 2
  rotation_y = 0.01
  print('alpha2rot_y', alpha2rot_y(alpha, ct[0], calib[0, 2], calib[0, 0]))
  print('rotation_y', rotation_y)
