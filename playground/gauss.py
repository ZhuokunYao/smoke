import numpy as np
import cv2
def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    # 2r * 2r   center=1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    # y is to the bottom!!!!  so that z can to the front!!!
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    #eps min val of >=0
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

radius = 3
diameter = 2 * radius + 1
gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
cv2.imwrite("gauss.png", gaussian*255)
print(gaussian.shape) #11,11
print(np.max(gaussian),np.min(gaussian))

point_int = [20, 8]
all_point = []
scores = []
gauss_square = gaussian2D((2*radius+1,2*radius+1), (2*radius+1)/6)
for gauss_h in range(2*radius+1):
    for gauss_w in range(2*radius+1):
        offset_h = gauss_h - radius
        offset_w = gauss_w - radius
        gauss_score = gauss_square[gauss_h, gauss_w]
        if(gauss_score > 0.1):
            all_point.append((point_int[0]+offset_w, point_int[1]+offset_h))
            scores.append(gauss_score)
for p,s in zip(all_point,scores):
    print(f"point: {p[0]},{p[1]}   score:{s}")