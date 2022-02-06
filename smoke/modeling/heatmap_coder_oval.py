import numpy as np
from skimage import transform as trans


def get_transfrom_matrix(center_scale, output_size):
    center, scale = center_scale[0], center_scale[1]
    # todo: further add rot and shift here.
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    src_dir = np.array([0, src_w * -0.5])
    dst_dir = np.array([0, dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = np.array([dst_w * 0.5, dst_h * 0.5])
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = get_3rd_point(dst[0, :], dst[1, :])

    get_matrix = trans.estimate_transform("affine", src, dst)
    matrix = get_matrix.params

    return matrix.astype(np.float32)


def affine_transform(point, matrix):
    point_exd = np.array([point[0], point[1], 1.])
    new_point = np.matmul(matrix, point_exd)

    return new_point[:2]


def get_3rd_point(point_a, point_b):
    d = point_a - point_b
    point_c = point_b + np.array([-d[1], d[0]])
    return point_c


def gaussian_radius_oval(h, w, thresh_min=0.7):
    a1 = 1
    b1 = h + w
    c1 = h * w * (1 - thresh_min) / (1 + thresh_min)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (h + w)
    c2 = (1 - thresh_min) * w * h
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * thresh_min
    b3 = -2 * thresh_min * (h + w)
    c3 = (thresh_min - 1) * w * h
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)

    # return min(r1, r2, r3)
    r = min(r1, r2, r3)
    return (np.sqrt(w/h) * r, np.sqrt(h/w) * r)



def gaussian2D_oval(shape, sigma=[1,1]):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    # h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    # h[h < np.finfo(h.dtype).eps * h.max()] = 0

    h = np.exp(-(x * x / (2 * sigma[1] * sigma[1])  + y * y / (2 * sigma[0] * sigma[0])))
    h[h < np.exp(-(n*n)/(2 * sigma[1] * sigma[1]))] = 0
    return h


def draw_umich_gaussian_oval(heatmap, center, radius, k=1):
    # diameter = 2 * radius + 1
    # gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    diameter = [2 * radius[1] + 1, 2 * radius[0] + 1]
    sigma = [d / 6 for d in diameter]
    gaussian = gaussian2D_oval(diameter, sigma=sigma)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius[0]), min(width - x, radius[0] + 1)
    top, bottom = min(y, radius[1]), min(height - y, radius[1] + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius[1] - top:radius[1] + bottom, radius[0] - left:radius[0] + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap

def draw_umich_gaussian_oval_all(heatmap, center, radius, k=1):
    # diameter = 2 * radius + 1
    # gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    diameter = [2 * radius[1] + 1, 2 * radius[0] + 1]
    sigma = [d / 6 for d in diameter]
    gaussian = gaussian2D_oval(diameter, sigma=sigma)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    # left, right = min(x, radius[0]), min(width - x, radius[0] + 1)
    # top, bottom = min(y, radius[1]), min(height - y, radius[1] + 1)

    x0 = max(0, x - radius[0])
    x1 = min(width, x + radius[0] + 1)
    y0 = max(0, y - radius[1])
    y1 = min(height, y + radius[1] + 1)
    left, right = x - x0, x1 - x
    top, bottom = y - y0, y1 - y

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius[1] - top:radius[1] + bottom, radius[0] - left:radius[0] + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap
