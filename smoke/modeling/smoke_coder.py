import numpy as np

import torch

PI = 3.14159


def encode_label_iou_box(dims, locs):
    l, h, w = dims[0], dims[1], dims[2]
    x, y, z = locs[0], locs[1], locs[2]

    x_corners = [0, l, l, l, l, 0, 0, 0]
    y_corners = [0, 0, h, h, 0, 0, h, h]
    z_corners = [0, 0, 0, w, w, w, w, 0]

    x_corners += - np.float32(l) / 2
    y_corners += - np.float32(h)
    z_corners += - np.float32(w) / 2

    corners_3d = np.array([x_corners, y_corners, z_corners])
    corners_3d += np.array([x, y, z]).reshape([3, 1])
    # corners_3d: 3*8 
    return corners_3d

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
    # 3, 8
    corners_3d += np.array([x, y, z]).reshape([3, 1])

    loc_center = np.array([x, y - h / 2, z])
    proj_point = np.matmul(K, loc_center)
    proj_point = proj_point[:2] / proj_point[2]

    corners_2d = np.matmul(K, corners_3d)
    # 2, 8
    corners_2d = corners_2d[:2] / corners_2d[2]
    box2d = np.array([min(corners_2d[0]), min(corners_2d[1]),
                      max(corners_2d[0]), max(corners_2d[1])])
    
    # proj_point: 3D center projected into 2D
    # box2d: xmin, ymin, xmax, ymax
    # corners_3d: 3*8 
    return proj_point, box2d, corners_3d, corners_2d


class SMOKECoder():
    def __init__(self, depth_ref, normalized_focal_ref, dim_ref, device="cuda"):
        # (28.01, 16.32)
        self.depth_ref = torch.as_tensor(depth_ref).to(device=device)
        # 720.0
        self.y_focal_ref = torch.as_tensor(normalized_focal_ref).to(device=device)
        # ((4.392, 1.658, 1.910),
        # (1.773, 1.525, 0.740),
        # (0.505, 1.644, 0.582),
        # (7.085, 2.652, 2.523),
        # (2.790, 1.651, 1.201),
        # (8.208, 2.869, 2.645))
        self.dim_ref = torch.as_tensor(dim_ref).to(device=device)
        # 720.0
        self.normalized_focal_ref = normalized_focal_ref

    # K is the origin instrict matric!
    def encode_box2d(self, K, rotys, dims, locs, img_size):
        device = rotys.device
        K = K.to(device=device)

        # H,W
        img_size = img_size.flatten()

        # [N,3,8]
        box3d = self.encode_box3d(rotys, dims, locs)
        # [N,3,8]
        box3d_image = torch.matmul(K, box3d)
        # [N,2,8]
        box3d_image = box3d_image[:, :2, :] / box3d_image[:, 2, :].view(
            box3d.shape[0], 1, box3d.shape[2]
        )

        xmins, _ = box3d_image[:, 0, :].min(dim=1)
        xmaxs, _ = box3d_image[:, 0, :].max(dim=1)
        ymins, _ = box3d_image[:, 1, :].min(dim=1)
        ymaxs, _ = box3d_image[:, 1, :].max(dim=1)

        xmins = xmins.clamp(0, img_size[0])
        xmaxs = xmaxs.clamp(0, img_size[0])
        ymins = ymins.clamp(0, img_size[1])
        ymaxs = ymaxs.clamp(0, img_size[1])

        bboxfrom3d = torch.cat((xmins.unsqueeze(1), ymins.unsqueeze(1),
                                xmaxs.unsqueeze(1), ymaxs.unsqueeze(1)), dim=1)
        # [N,4]    (xmin, ymin, xmax, ymax)*N
        return bboxfrom3d

    @staticmethod
    # rotys: [rotys, rotys, rotys, ...., rotys]
    def rad_to_matrix(rotys, N):
        device = rotys.device

        cos, sin = rotys.cos(), rotys.sin()

        i_temp = torch.tensor([[1, 0, 1],
                               [0, 1, 0],
                               [-1, 0, 1]]).to(dtype=torch.float32,
                                               device=device)
        ry = i_temp.repeat(N, 1).view(N, -1, 3)
        
        # rotation with axis y
        # [cos,  0,   sin]
        # [0,    1,   0  ]
        # [-sin, 0,   cos]

        ry[:, 0, 0] *= cos
        ry[:, 0, 2] *= sin
        ry[:, 2, 0] *= sin
        ry[:, 2, 2] *= cos

        return ry

    def encode_box3d(self, rotys, dims, locs):
        '''
        construct 3d bounding box for each object.
        Args:
            rotys: rotation in shape N
            dims: dimensions of objects
            locs: locations of objects     y is the top of the car!!!!!

        Returns:

        '''
        #rotys is an angle
        # [rotys, rotys, rotys, ...., rotys]
        if len(rotys.shape) == 2:
            rotys = rotys.flatten()

        # [[lx,ly,lz], [lx,ly,lz], [lx,ly,lz] ...]
        if len(dims.shape) == 3:
            dims = dims.view(-1, 3)

        # [[x,y,z], [x,y,z], [x,y,z] ...]
        if len(locs.shape) == 3:
            locs = locs.view(-1, 3)

        device = rotys.device
        # num of objects
        N = rotys.shape[0]

        # N,3,3
        ry = self.rad_to_matrix(rotys, N)
        # [ [3*N], [3*N], [3*N], [3*N], [3*N], [3*N], [3*N], [3*N] ]
        dims = dims.view(-1, 1).repeat(1, 8)
        # x,z of the first 4 points   *  0.5
        dims[::3, :4], dims[2::3, :4] = 0.5 * dims[::3, :4], 0.5 * dims[2::3, :4]
        # x,z of the last  4 points   *  -0.5
        dims[::3, 4:], dims[2::3, 4:] = -0.5 * dims[::3, 4:], -0.5 * dims[2::3, 4:]
        # y of the 8 points
        dims[1::3, :4], dims[1::3, 4:] = 0., -dims[1::3, 4:]
        index = torch.tensor([[4, 0, 1, 2, 3, 5, 6, 7],
                              [4, 5, 0, 1, 6, 7, 2, 3],
                              [4, 5, 6, 0, 1, 2, 3, 7]]).repeat(N, 1).to(device=device)
        #  -0.5lx  0.5lx  0.5lx  0.5lx  0.5lx  -0.5lx  -0.5lx  -0.5lx
        #  -ly     -ly    0      0      -ly    -ly     0       0
        #  -0.5lz  -0.5lz -0.5lz 0.5lz  0.5lz  0.5lz   0.5lz   -0.5lz
        #  repeat for N objects
        #  N*3 , 8
        box_3d_object = torch.gather(dims, 1, index)

        # rotated   [N,3,8]
        box_3d = torch.matmul(ry, box_3d_object.view(N, 3, -1))

        # shifted   [N,3,8]
        box_3d += locs.unsqueeze(-1).repeat(1, 1, 8)

        return box_3d
    def decode_target_depth(self, target_depth, K):
        '''
        Transform depth offset to depth
        '''
        if self.normalized_focal_ref < 0:
            pass
        else:
            # batch_size * max_points
            N = target_depth.shape[0]
            # batch size
            N_batch = K.shape[0]

            device = target_depth.device
            # [batch_size, 9]
            K = K.to(device=device).view(N_batch, -1)
            # self.y_focal_ref : 720.0
            # K:  [f/dx  0    u]
            #     [0    f/dy  v]
            #     [0     0    1]
            # dy: the real-word length of each pixel
            y_focal_ratio = K[:, 4] / self.y_focal_ref   #[batch_size] (1/dy)
            # [batch * max_points]
            # z length in pixel
            y_focal_ratio = y_focal_ratio.unsqueeze(1).repeat(1, N // N_batch).view(-1)

            #self.depth_ref : (28.01, 16.32)
            depth = target_depth * y_focal_ratio
        # [batch_size * max_points]
        return depth
    
    # depths_offset: [batch_size * max_points]
    # K: [batch_size, 3, 3]
    def decode_depth(self, depths_offset, K):
        '''
        Transform depth offset to depth
        '''

        # ### orginal
        # 720.0
        if self.normalized_focal_ref < 0:
            depth = depths_offset * self.depth_ref[1] + self.depth_ref[0]
        else:
            # batch_size * max_points
            N = depths_offset.shape[0]
            # batch size
            N_batch = K.shape[0]

            device = depths_offset.device
            # [batch_size, 9]
            K = K.to(device=device).view(N_batch, -1)
            # self.y_focal_ref : 720.0
            # K:  [f/dx  0    u]
            #     [0    f/dy  v]
            #     [0     0    1]
            # dy: the real-word length of each pixel
            y_focal_ratio = K[:, 4] / self.y_focal_ref   #[batch_size] (1/dy)
            # [batch * max_points]
            # z length in pixel
            y_focal_ratio = y_focal_ratio.unsqueeze(1).repeat(1, N // N_batch).view(-1)

            #self.depth_ref : (28.01, 16.32)
            depth = (depths_offset * self.depth_ref[1] + self.depth_ref[0]) * y_focal_ratio
        
        # [batch_size * max_points]
        return depth

    # directly infer the depth
    def decode_depth_std(self, depths_offset, K):
        '''
        Transform depth offset to depth
        '''

        # ### orginal
        if self.normalized_focal_ref < 0:
            depth = depths_offset * self.depth_ref[1] + self.depth_ref[0]
        else:
            # number of points
            N = depths_offset.shape[0]
            # batch size
            N_batch = K.shape[0]

            device = depths_offset.device
            K = K.to(device=device).view(N_batch, -1)
            y_focal_ratio = K[:, 4] / self.y_focal_ref

            y_focal_ratio = y_focal_ratio.unsqueeze(1).repeat(1, N // N_batch).view(-1)
            depth = depths_offset * y_focal_ratio

        return depth

    def decode_location(self,
                        points,
                        points_offset,
                        depths,
                        Ks,
                        trans_mats):
        '''
        retrieve objects location in camera coordinate based on projected points
        Args:
            points: projected points on feature map in (x, y)
            points_offset: project points offset in (delata_x, delta_y)
            depths: object depth z
            Ks: camera intrinsic matrix, shape = [N, 3, 3]
            trans_mats: transformation matrix from image to feature map, shape = [N, 3, 3]

        Returns:
            locations: objects location, shape = [N, 3]
        '''

        # points:            batzh_size * max_points, 2
        #                            select from the prediced heatmap(test)
        #                            select from the annotation(test)
        # points_offset:     batzh_size * max_points, 2
        #                            from the regression
        # depths:            batzh_size * max_points
        #                            from func decode_depth, z in the paper
        #                            located at "points"
        # Ks:                batzh_size, 3, 3
        # trans_mats:        batzh_size, 3, 3
        device = points.device

        Ks = Ks.to(device=device)
        trans_mats = trans_mats.to(device=device)

        # number of points
        N = points_offset.shape[0]
        # batch size
        N_batch = Ks.shape[0]
        batch_id = torch.arange(N_batch).unsqueeze(1)
        # [0,0,..,0,   1,...1,   batch_size-1,...,batch_size-1] (repeat max_points times)
        obj_id = batch_id.repeat(1, N // N_batch).flatten()
        

        # [batzh_size * max_points, 3, 3]
        trans_mats_inv = trans_mats.inverse()[obj_id]
        # [batzh_size * max_points, 3, 3]
        Ks_inv = Ks.inverse()[obj_id]

        # batzh_size * max_points, 2
        points = points.view(-1, 2)
        assert points.shape[0] == N
        # xc + oxc,   yc + oyc in the paper
        # batzh_size * max_points, 2
        proj_points = points + points_offset

        # transform project points in homogeneous form.
        # batzh_size * max_points, 3
        proj_points_extend = torch.cat(
            (proj_points, torch.ones(N, 1).to(device=device)), dim=1)

        # expand project points as [N, 3, 1]
        proj_points_extend = proj_points_extend.unsqueeze(-1)

        # transform project points back on image
        # batzh_size * max_points, 3, 1
        proj_points_img = torch.matmul(trans_mats_inv, proj_points_extend)
        # with depth
        proj_points_img = proj_points_img * depths.view(N, -1, 1)

        # transform image coordinates back to object locations
        # x,y,z in the paper
        locations = torch.matmul(Ks_inv, proj_points_img)
        
        # batzh_size * max_points, 3 (xyz in the real world)
        return locations.squeeze(2)

    def decode_dimension(self, cls_id, dims_offset):
        '''
        retrieve object dimensions
        Args:
            cls_id: each object id
            dims_offset: dimension offsets, shape = (N, 3)

        Returns:

        '''
        # cls_id:          batzh_size * max_points
        #                             select from the prediced heatmap (test)
        #                             select from the annotation (train)
        # dims_offset:     batzh_size * max_points, 3
        #                             from the regression
        #print("***********")
        #print(cls_id.device,dims_offset.device)
        cls_id = cls_id.flatten().long()
        #print(cls_id.device)
        #self.dim_ref:
        # ((4.392, 1.658, 1.910),
        # (1.773, 1.525, 0.740),
        # (0.505, 1.644, 0.582),
        # (7.085, 2.652, 2.523),
        # (2.790, 1.651, 1.201),
        # (8.208, 2.869, 2.645))
        
        self.dim_ref = self.dim_ref.to(cls_id.device)
        dims_select = self.dim_ref[cls_id, :]     # [batzh_size * max_points, 3]
        #print(dims_select.device)
        dimensions = dims_offset.exp() * dims_select
        #print(dimensions.device)
        
        # [batzh_size * max_points, 3]
        return dimensions

    def decode_orientation(self, vector_ori, locations, flip_mask=None):
        '''
        retrieve object orientation
        Args:
            vector_ori: local orientation in [sin, cos] format
            locations: object location

        Returns: for training we only need roty
                 for testing we need both alpha and roty

        '''
        # vector_ori: batzh_size * max_points, 2
        #                  from the regression
        # locations:  batzh_size * max_points, 3
        #                  func(decode_location)&func(decode_dimension) ---> topcar 3D locations (test)
        #                  from the annotations (train)      

        ### lyb: angle definition and direction(inverse is negative) by paper#####
        locations = locations.view(-1, 3)
        alphas_x = torch.atan2(vector_ori[:, 0], vector_ori[:, 1] + 1e-7)
        ray_thetas = torch.atan2(locations[:, 0], locations[:, 2] + 1e-7)

        if flip_mask is not None:
            fm = flip_mask.flatten()
            alphas_x = (1 - 2 * fm.float()) * alphas_x
        ##########################################################################

        #### alphas_z = kitti's alphas
        alphas = (alphas_x - PI / 2)
        rotys = alphas + ray_thetas

        # in training time, it does not matter if angle lies in [-PI, PI]
        # it matters at inference time? todo: does it really matter if it exceeds.

        # rotys:  batzh_size * max_points, theta in the paper
        # alphas: batzh_size * max_points, alphaz in the paper
        return rotys, alphas

if __name__ == '__main__':
    sc = SMOKECoder(depth_ref=(28.01, 16.32),
                    dim_ref=((3.88, 1.63, 1.53),
                             (1.78, 1.70, 0.58),
                             (0.88, 1.73, 0.67)))
    depth_offset = torch.tensor([-1.3977,
                                 -0.9933,
                                 0.0000,
                                 -0.7053,
                                 -0.3652,
                                 -0.2678,
                                 0.0650,
                                 0.0319,
                                 0.9093])
    depth = sc.decode_depth(depth_offset)
    print(depth)

    points = torch.tensor([[4, 75],
                           [200, 59],
                           [0, 0],
                           [97, 54],
                           [105, 51],
                           [165, 52],
                           [158, 50],
                           [111, 50],
                           [143, 48]], )
    points_offset = torch.tensor([[0.5722, 0.1508],
                                  [0.6010, 0.1145],
                                  [0.0000, 0.0000],
                                  [0.0365, 0.1977],
                                  [0.0267, 0.7722],
                                  [0.9360, 0.0118],
                                  [0.8549, 0.5895],
                                  [0.6011, 0.6448],
                                  [0.4246, 0.4782]], )
    K = torch.tensor([[721.54, 0., 631.44],
                      [0., 721.54, 172.85],
                      [0, 0, 1]]).unsqueeze(0)

    trans_mat = torch.tensor([[2.5765e-01, -0.0000e+00, 2.5765e-01],
                              [-2.2884e-17, 2.5765e-01, -3.0918e-01],
                              [0, 0, 1]], ).unsqueeze(0)
    locations = sc.decode_location(points, points_offset, depth, K, trans_mat)

    cls_ids = torch.tensor([[0],
                            [0],
                            [0],
                            [0],
                            [0],
                            [0],
                            [0],
                            [0],
                            [0]])

    dim_offsets = torch.tensor([[-0.0375, 0.0755, -0.1469],
                                [-0.1309, 0.1054, 0.0179],
                                [0.0000, 0.0000, 0.0000],
                                [-0.0765, 0.0447, -0.1803],
                                [-0.1170, 0.1286, 0.0552],
                                [-0.0568, 0.0935, -0.0235],
                                [-0.0898, -0.0066, -0.1469],
                                [-0.0633, 0.0755, 0.1189],
                                [0.0061, -0.0537, -0.1088]]).roll(1, 1)
    dimensions = sc.decode_dimension(cls_ids, dim_offsets)

    locations[:, 1] += dimensions[:, 1] / 2
    print(locations)
    print(dimensions)

    vector_ori = torch.tensor([[0.4962, 0.8682],
                               [0.3702, -0.9290],
                               [0.0000, 0.0000],
                               [0.2077, 0.9782],
                               [0.1189, 0.9929],
                               [0.2272, -0.9738],
                               [0.1979, -0.9802],
                               [0.0990, 0.9951],
                               [0.3421, -0.9396]])
    flip_mask = torch.tensor([1, 1, 0, 1, 1, 1, 1, 1, 1])
    rotys = sc.decode_orientation(vector_ori, locations, flip_mask)
    print(rotys)
    rotys = torch.tensor([[1.4200],
                          [-1.7600],
                          [0.0000],
                          [1.4400],
                          [1.3900],
                          [-1.7800],
                          [-1.7900],
                          [1.4000],
                          [-2.0200]])
    box3d = sc.encode_box3d(rotys, dimensions, locations)
    print(box3d)
