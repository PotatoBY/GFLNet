import numpy as np
import torch
import torch.nn as nn
import trimesh
import scipy.sparse as sparse
from sklearn.preprocessing import normalize
from sksparse.cholmod import cholesky_AAt
import os.path as osp

class LapMeshDeform():
    def __init__(self, verts, faces):
        self.verts = verts  # numpy array with shape of [n_v, 3], n_v is the number of  vertices
        self.faces = faces  # numpy array with shape of [n_f, 3], n_f is the number of vertices

    def _compute_edge(self):
        '''
        given faces with numpy array of [n_f, 3], return its edges[num_edge, 2], in addition edges[:, 0] < edges[:, 1].
        we do this sorting operation in edges' array for removing duplicated elements.
        For example self.faces[[0, 1, 2],
                               [0, 3, 1],
                               [0, 2, 4],
                              ]
        The returned edges will be: [[0, 1],
                                     [0, 2],
                                     [0, 3],
                                     [0, 4],
                                     [1, 2],
                                     [1, 3],
                                     [2, 4],
                                    ]

        Args:
            self.faces: numpy array with size of [n_f, 3], n_f is the number of faces.

        Return:
            uni_all_edge: numpy array with size of [n_e, 2], n_e is the number of edges.

        '''
        # get the edges of triangles. numpy array with shape [n_f, 2]
        edge_v0v1 = self.faces[:, [0, 1]]  # edge of v0---v1
        edge_v0v2 = self.faces[:, [0, 2]]  # edge of v0---v2
        edge_v1v2 = self.faces[:, [1, 2]]  # edge of v1---v2

        # sorting the vertex index in edges for the purpose of removing duplicated elements.
        # for example if edge_v0v1[i, :] = [4, 1]. we will change edge_v0v1[i, :] to be [1, 4]
        edge_v0v1.sort()
        edge_v0v2.sort()
        edge_v1v2.sort()

        all_edge = np.vstack((edge_v0v1, edge_v0v2, edge_v1v2))  # numpy array with shape [n_f*3, 2]

        # remove duplicated edges
        uni_all_edge = np.unique(all_edge, axis=0)

        return uni_all_edge

    def uniform_laplacian(self):
        '''
        computing the uniform laplacian matrix L, L is an n by n matrix sparse matrix.
        See the reference of <<Differential Representations for Mesh Processing>>
                   --  =   1 ,       if i=j
        L[i, j] --|    =  -wij,      wij = 1/|N(i)|if (i, j) belong to an edge of a face
                   --  =   0       , others.

        Args:
            self.faces: numpy array with shape of [n_f, 3], mesh's faces.
            self.verts: numpy array with shape of [n_v, 3], mesh's vertices.
                        we only used its n_v to create [n_v, n_v] sparse matrix.

        Return:
            lap_matrix: the returned Laplacian matrix, it is a spare matrix.

        '''
        # initial the laplacian matrix(i.e. L) of self.faces. with the shape of [n_v, n_v], n_v is the number of vertices
        lap_matrix = sparse.lil_matrix((self.verts.shape[0], self.verts.shape[0]), dtype=np.float32)

        # get the edges with sorted index. the edges_sorted is with the shape of [n_e, 2]
        edges_sorted = self._compute_edge()
        lap_matrix[edges_sorted[:, 0], edges_sorted[:, 1]] = 1  # L[i, j] = 1, for edge i----j
        lap_matrix[edges_sorted[:, 1], edges_sorted[:, 0]] = 1  # L[j, i] = 1, for edge i----j
        lap_matrix = normalize(lap_matrix, norm='l1',
                               axis=1) * -1  # normaliz the L[i, j], for edge i----j . to -1/|N(i)|

        unit_diagonal = sparse.identity(self.verts.shape[0], dtype=np.float32)  # L[i,j] = 1, if i=j
        lap_matrix += unit_diagonal

        return lap_matrix

    def cot_laplacian(self, area_normalize=True):
        '''
        computing the uniform laplacian matrix L, L is an n by n matrix sparse matrix.
        See the reference of <<Differential Representations for Mesh Processing>> for definition of the cot weight.
                   --  =   1 ,       if i=j
        L[i, j] --|    =  -wij,    wij = (cotαij + cotβij)/4A(i). 4A(i) = 0.5*sum_(k in N(i)) (cotαkj + cotβkj)|vi-vk|^2
                   --  =   0       , others.

        This operation of cot_laplacian use the area normalize. It's sum of weight not equal to 1.
        To compute the size of Voronoi regions : A(i)
        I follow the reference https://stackoverflow.com/questions/13882225/compute-the-size-of-voronoi-regions-from-delaunay-triangulation
        The cotαik*|vi-vk| = H, H is the length of perpendicular line from vertices of αik to edge vi---vk.
        so 0.5*cotαik*|vi-vk|^2 is the area of triangle which contain edge of vi---vk, and angle of αkj.

        Args:
            self.faces: numpy array with shape of [n_f, 3], mesh's faces.
            self.verts: numpy array with shape of [n_v, 3], mesh's vertices.
                        we only used its n_v to create [n_v, n_v] sparse matrix.
            area_normalize: True for wij = (cotαij + cotβij)/4A(i) depicted above.
                            False for wij = (cotαij + cotβij)/sum_(k in N(i)) (cotαkj + cotβkj)

        Return:
            lap_matrix: the returned Laplacian matrix, it is a spare matrix.
        '''
        # initial the laplacian matrix(i.e. L) of self.faces. with the shape of [n_v, n_v], n_v is the number of vertices
        lap_matrix = sparse.lil_matrix((self.verts.shape[0], self.verts.shape[0]), dtype=np.float32)
        sum_area_vert = np.zeros([self.verts.shape[0], 1])

        # get the vertex index of edges of triangles. numpy array with shape [n_f, 2].
        edge_v0v1 = self.faces[:, [0, 1]]  # edge of v0---v1
        edge_v0v2 = self.faces[:, [0, 2]]  # edge of v0---v2
        edge_v1v2 = self.faces[:, [1, 2]]  # edge of v1---v2

        # compute length of edges, numpy array, shape is (n_f, )
        length_edge_v0v1 = np.linalg.norm(self.verts[edge_v0v1[:, 0], :] - self.verts[edge_v0v1[:, 1], :], axis=1)
        length_edge_v0v2 = np.linalg.norm(self.verts[edge_v0v2[:, 0], :] - self.verts[edge_v0v2[:, 1], :], axis=1)
        length_edge_v1v2 = np.linalg.norm(self.verts[edge_v1v2[:, 0], :] - self.verts[edge_v1v2[:, 1], :], axis=1)

        # compute area of each triangle see the reference https://pythonguides.com/find-area-of-a-triangle-in-python/
        # faces_area is numpy array with shape: (n_f, )
        average_edge_len = (length_edge_v0v1 + length_edge_v0v2 + length_edge_v1v2) / 2.0
        faces_area = (average_edge_len * (average_edge_len - length_edge_v0v1) * (
                    average_edge_len - length_edge_v0v2) * (average_edge_len - length_edge_v1v2)) ** 0.5

        # compute the cot value of angle, the angle is face towards to edges.
        # cot value is numpy array with shape of (n_f, )
        cot_value_angle_face_v0v1 = (length_edge_v1v2 ** 2 + length_edge_v0v2 ** 2 - length_edge_v0v1 ** 2) / (
                    4 * faces_area)
        cot_value_angle_face_v0v2 = (length_edge_v1v2 ** 2 + length_edge_v0v1 ** 2 - length_edge_v0v2 ** 2) / (
                    4 * faces_area)
        cot_value_angle_face_v1v2 = (length_edge_v0v2 ** 2 + length_edge_v0v1 ** 2 - length_edge_v1v2 ** 2) / (
                    4 * faces_area)

        # sum the triangles' area of vertices belong to.
        # the sum_area_vert is numpy array, shape is (n_v, 1)
        for i in range(faces_area.shape[0]):
            sum_area_vert[self.faces[i, 0]] += faces_area[i]
            sum_area_vert[self.faces[i, 1]] += faces_area[i]
            sum_area_vert[self.faces[i, 2]] += faces_area[i]

        # cot laplacian matrix
        for j in range(edge_v0v1.shape[0]):
            lap_matrix[edge_v0v1[j, 0], edge_v0v1[j, 1]] += cot_value_angle_face_v0v1[j]
            lap_matrix[edge_v0v1[j, 1], edge_v0v1[j, 0]] += cot_value_angle_face_v0v1[j]
            lap_matrix[edge_v0v2[j, 0], edge_v0v2[j, 1]] += cot_value_angle_face_v0v2[j]
            lap_matrix[edge_v0v2[j, 1], edge_v0v2[j, 0]] += cot_value_angle_face_v0v2[j]
            lap_matrix[edge_v1v2[j, 0], edge_v1v2[j, 1]] += cot_value_angle_face_v1v2[j]
            lap_matrix[edge_v1v2[j, 1], edge_v1v2[j, 0]] += cot_value_angle_face_v1v2[j]

        lap_matrix_nonormalize = lap_matrix.copy()
        if area_normalize:
            # normalize wij with the size of Voronoi regions: 4A(i) = 0.5*sum_(k in N(i)) (cotαkj + cotβkj)|vi-vk|^2
            for k in range(self.verts.shape[0]):
                if sum_area_vert[k, :] != 0:
                    lap_matrix[k, :] = lap_matrix[k, :] / (sum_area_vert[k, :])
            lap_matrix = lap_matrix * -1
        else:
            # normalize wij with uniform value: sum_(k in N(i)) (cotαkj + cotβkj)
            lap_matrix = normalize(lap_matrix, norm='l1', axis=1)
            lap_matrix = lap_matrix * -1

        unit_diagonal = identity(self.verts.shape[0], dtype=np.float32)  # L[i,j] = 1, if i=j
        lap_matrix += unit_diagonal

        return lap_matrix


class GARMENT(nn.Module):
    def __init__(self, opt):
        super(GARMENT, self).__init__()
        cuda = torch.device('cuda') if len(opt.gpu_ids) > 1 else torch.device('cuda:%d' % opt.gpu_id)
        self.cur_device = cuda

    def setParams(self,name):
        self.collar_np = torch.Tensor().to(self.cur_device)
        name = "/mnt/data/Project/GFLNet-master/gfl-main/lib/"+name
        with open(name + "/collar.txt", "r") as f:
            for line in f.readlines():
                line = line.split(',')
                line = list(filter(None, line))
                for i in line:
                    self.collar_np = torch.cat([self.collar_np, self.up_faces[int(i)].unsqueeze(0)], dim=0).int()
        self.sleeve_l_np = torch.Tensor().to(self.cur_device)
        with open(name + "/sleeve_l.txt", "r") as f:
            for line in f.readlines():
                line = line.split(',')
                line = list(filter(None, line))
                for i in line:
                    self.sleeve_l_np = torch.cat([self.sleeve_l_np, self.up_faces[int(i)].unsqueeze(0)], dim=0).int()
        self.shoulder_l_np = torch.Tensor().to(self.cur_device)
        with open(name + "/shoulder_l.txt", "r") as f:
            for line in f.readlines():
                line = line.split(',')
                line = list(filter(None, line))
                for i in line:
                    self.shoulder_l_np = torch.cat([self.shoulder_l_np, self.up_faces[int(i)].unsqueeze(0)], dim=0).int()
        self.sleeve_r_np = torch.Tensor().to(self.cur_device)
        with open(name + "/sleeve_r.txt", "r") as f:
            for line in f.readlines():
                line = line.split(',')
                line = list(filter(None, line))
                for i in line:
                    self.sleeve_r_np = torch.cat([self.sleeve_r_np, self.up_faces[int(i)].unsqueeze(0)], dim=0).int()

        self.shoulder_r_np = torch.Tensor().to(self.cur_device)
        with open(name + "/shoulder_r.txt", "r") as f:
            for line in f.readlines():
                line = line.split(',')
                line = list(filter(None, line))
                for i in line:
                    self.shoulder_r_np = torch.cat([self.shoulder_r_np, self.up_faces[int(i)].unsqueeze(0)],
                                                   dim=0).int()

        self.up_waist_np = torch.Tensor().to(self.cur_device)
        with open(name + "/waist.txt", "r") as f:
            for line in f.readlines():
                line = line.split(',')
                line = list(filter(None, line))
                for i in line:
                    self.up_waist_np = torch.cat([self.up_waist_np, self.up_faces[int(i)].unsqueeze(0)], dim=0).int()

        collar_x = torch.Tensor().to(self.cur_device)
        collar_y = torch.Tensor().to(self.cur_device)
        collar_z = torch.Tensor().to(self.cur_device)
        for f in self.collar_np:
            for i in range(3):
                collar_x = torch.cat([collar_x, self.up_vertices_deform[f[i]][0].unsqueeze(0)], dim=0)
                collar_y = torch.cat([collar_y, self.up_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)
                collar_z = torch.cat([collar_z, self.up_vertices_deform[f[i]][2].unsqueeze(0)], dim=0)

        self.collar_x_min = torch.min(collar_x)  # 当前数据最大值
        self.collar_x_max = torch.max(collar_x)  # 当前数据最小值
        self.collar_y_min = torch.min(collar_y)  # 当前数据最大值
        self.collar_y_max = torch.max(collar_y)  # 当前数据最小值
        self.collar_z_min = torch.min(collar_z)  # 当前数据最大值
        self.collar_z_max = torch.max(collar_z)  # 当前数据最小值
        y_max = torch.max(self.up_vertices[:, 1])

        sleeve_l_y = torch.Tensor().to(self.cur_device)
        for f in self.sleeve_l_np:
            for i in range(3):
                sleeve_l_y = torch.cat([sleeve_l_y, self.up_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)

        sleeve_l_y_min = torch.min(sleeve_l_y)  # 最左边点的x
        sleeve_l_y_max = torch.max(sleeve_l_y)  # 当前数据最小值

        waist_x = torch.Tensor().to(self.cur_device)
        for f in self.up_waist_np:
            for i in range(3):
                waist_x = torch.cat([waist_x, self.up_vertices_deform[f[i]][0].unsqueeze(0)], dim=0)

        waist_x_min = torch.min(waist_x)  # 当前数据最大值
        waist_x_max = torch.max(waist_x)  # 当前数据最小值

        scale_y = self.lines[0] / (sleeve_l_y_max - sleeve_l_y_min)
        # scale_x = self.lines[1] / (waist_x_max - waist_x_min)
        for i in range(self.up_vertices.shape[0]):
            # tmp = y_max - self.up_vertices[i][1]
            # y = tmp * scale_y
            # i[0] = 0.2 + x_r
            # self.up_vertices[i][0] *= scale_x
            self.up_vertices[i][1] *= scale_y
            # self.up_vertices[i][2] *= scale_x * 1.2
        self.up_vertices_deform = self.up_vertices.clone()

    def set_sleeve_l(self, name):
        face_np = self.sleeve_l_np
        sl = self.lines[0]
        set_x = torch.Tensor().to(self.cur_device)
        set_y = torch.Tensor().to(self.cur_device)
        set_z = torch.Tensor().to(self.cur_device)
        is_change = np.zeros(self.up_vertices_deform.shape[0])
        vert_anchor = torch.Tensor().to(self.cur_device)
        idx_anchor_in_mesh = torch.Tensor().to(self.cur_device)

        for f in face_np:
            for i in range(3):
                set_x = torch.cat([set_x,self.up_vertices_deform[f[i]][0].unsqueeze(0)],dim=0)
                set_y = torch.cat([set_y,self.up_vertices_deform[f[i]][1].unsqueeze(0)],dim=0)
                set_z = torch.cat([set_z,self.up_vertices_deform[f[i]][2].unsqueeze(0)],dim=0)

        x_min = torch.min(set_x)  # 当前数据最大值
        x_max = torch.max(set_x)  # 当前数据最小值
        y_min = torch.min(set_y)  # 当前数据最大值
        y_max = torch.max(set_y)  # 当前数据最小值
        z_min = torch.min(set_z)  # 当前数据最大值
        z_max = torch.max(set_z)  # 当前数据最小值
        sl_ = torch.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
        sl_x = (sl / sl_) * (x_max - x_min)
        sl_y = (sl / sl_) * (y_max - y_min)
        sl_z = (z_max - z_min) / (x_max - x_min) * sl_x
        MIN_X = (x_max - x_min) / 2 + x_min - sl_x / 2
        MAX_X = (x_max - x_min) / 2 + x_min + sl_x / 2
        MIN_Y = y_max - sl_y
        MAX_Y = y_max
        # if y_max < self.collar_y_max:
        #     MIN_Y = y_max - sl_y
        #     MAX_Y = y_max
        # else:
        #     MIN_Y = (y_max - y_min) / 2 + y_min - sl_y / 2
        #     MAX_Y = (y_max - y_min) / 2 + y_min + sl_y / 2
        MIN_Z = (z_max - z_min) / 2 + z_min - sl_z / 2
        MAX_Z = (z_max - z_min) / 2 + z_min + sl_z / 2

        for f in face_np:
            for i in range(3):
                if is_change[f[i]] == 0:
                    is_change[f[i]] = 1
                    tmp = self.up_vertices_deform[f[i]]
                    tmp[0] = MIN_X + (MAX_X - MIN_X) / (x_max - x_min) * (
                            tmp[0] - x_min)
                    tmp[1] = MIN_Y + (MAX_Y - MIN_Y) / (y_max - y_min) * (
                            tmp[1] - y_min)
                    tmp[2] = MIN_Z + (MAX_Z - MIN_Z) / (z_max - z_min) * (
                            tmp[2] - z_min)
                    vert_anchor = torch.cat([vert_anchor,tmp.unsqueeze(0)],dim=0)
                    idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh,f[i].unsqueeze(0)],dim=0)
        return idx_anchor_in_mesh, vert_anchor

        # mesh = trimesh.Trimesh(vertices=self.up_vertices.cpu().detach().numpy(), faces=torch.Tensor(face_np).cpu().detach().numpy(), process=False)
        # mesh.show()
    def set_sleeve_r(self, name):
        face_np = self.sleeve_r_np
        sl = self.lines[0]
        # print(sl)
        set_x = torch.Tensor().to(self.cur_device)
        set_y = torch.Tensor().to(self.cur_device)
        set_z = torch.Tensor().to(self.cur_device)
        is_change = np.zeros(self.up_vertices_deform.shape[0])
        vert_anchor = torch.Tensor().to(self.cur_device)
        idx_anchor_in_mesh = torch.Tensor().to(self.cur_device)
        for f in face_np:
            for i in range(3):
                set_x = torch.cat([set_x, self.up_vertices_deform[f[i]][0].unsqueeze(0)], dim=0)
                set_y = torch.cat([set_y, self.up_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)
                set_z = torch.cat([set_z, self.up_vertices_deform[f[i]][2].unsqueeze(0)], dim=0)

        x_min = torch.min(set_x)  # 当前数据最大值
        x_max = torch.max(set_x)  # 当前数据最小值
        y_min = torch.min(set_y)  # 当前数据最大值
        y_max = torch.max(set_y)  # 当前数据最小值
        z_min = torch.min(set_z)  # 当前数据最大值
        z_max = torch.max(set_z)  # 当前数据最小值
        sl_ = torch.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
        sl_x = (sl / sl_) * (x_max - x_min)
        sl_y = (sl / sl_) * (y_max - y_min)
        sl_z = (z_max - z_min) / (x_max - x_min) * sl_x
        MIN_X = (x_max - x_min) / 2 + x_min - sl_x / 2
        MAX_X = (x_max - x_min) / 2 + x_min + sl_x / 2
        MIN_Y = y_max - sl_y
        MAX_Y = y_max
        # if y_max < self.collar_y_max:
        #     MIN_Y = y_max - sl_y
        #     MAX_Y = y_max
        # else:
        #     MIN_Y = (y_max - y_min) / 2 + y_min - sl_y / 2
        #     MAX_Y = (y_max - y_min) / 2 + y_min + sl_y / 2
        MIN_Z = (z_max - z_min) / 2 + z_min - sl_z / 2
        MAX_Z = (z_max - z_min) / 2 + z_min + sl_z / 2
        for f in face_np:
            for i in range(3):
                if is_change[f[i]] == 0:
                    is_change[f[i]] = 1
                    tmp = self.up_vertices_deform[f[i]]
                    tmp[0] = MIN_X + (MAX_X - MIN_X) / (x_max - x_min) * (
                            tmp[0] - x_min)
                    tmp[1] = MIN_Y + (MAX_Y - MIN_Y) / (y_max - y_min) * (
                            tmp[1] - y_min)
                    tmp[2] = MIN_Z + (MAX_Z - MIN_Z) / (z_max - z_min) * (
                            tmp[2] - z_min)
                    vert_anchor = torch.cat([vert_anchor,tmp.unsqueeze(0)],dim=0)
                    idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh,f[i].unsqueeze(0)],dim=0)
        # mesh = trimesh.Trimesh(vertices=self.up_vertices.cpu().detach().numpy(), faces=face_np.cpu().detach().numpy(), process=False)
        # mesh.show()
        return idx_anchor_in_mesh, vert_anchor
    def set_up_waist(self, name):

        face_np = self.up_waist_np

        sl = self.lines[1]-0.02
        set_x = torch.Tensor().to(self.cur_device)
        set_y = torch.Tensor().to(self.cur_device)
        set_z = torch.Tensor().to(self.cur_device)
        is_change = np.zeros(self.up_vertices_deform.shape[0])
        vert_anchor = torch.Tensor().to(self.cur_device)
        idx_anchor_in_mesh = torch.Tensor().to(self.cur_device)
        for f in face_np:
            for i in range(3):
                set_x = torch.cat([set_x, self.up_vertices_deform[f[i]][0].unsqueeze(0)], dim=0)
                set_y = torch.cat([set_y, self.up_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)
                set_z = torch.cat([set_z, self.up_vertices_deform[f[i]][2].unsqueeze(0)], dim=0)

        x_min = torch.min(set_x)  # 当前数据最大值
        x_max = torch.max(set_x)  # 当前数据最小值
        y_min = torch.min(set_y)  # 当前数据最大值
        y_max = torch.max(set_y)  # 当前数据最小值
        z_min = torch.min(set_z)  # 当前数据最大值
        z_max = torch.max(set_z)  # 当前数据最小值
        sl_ = torch.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
        sl_x = (sl / sl_) * (x_max - x_min)
        sl_y = (sl / sl_) * (y_max - y_min)
        sl_z = (z_max - z_min) / (x_max - x_min) * sl_x
        MIN_X = (x_max - x_min) / 2 + x_min - sl_x / 2
        MAX_X = (x_max - x_min) / 2 + x_min + sl_x / 2
        MIN_Y = (y_max - y_min) / 2 + y_min - sl_y / 2
        MAX_Y = (y_max - y_min) / 2 + y_min + sl_y / 2
        MIN_Z = (z_max - z_min) / 2 + z_min - sl_z / 2
        MAX_Z = (z_max - z_min) / 2 + z_min + sl_z / 2
        for f in face_np:
            for i in range(3):
                if is_change[f[i]] == 0:
                    is_change[f[i]] = 1
                    tmp = self.up_vertices_deform[f[i]]
                    tmp[0] = MIN_X + (MAX_X - MIN_X) / (x_max - x_min) * (
                            tmp[0] - x_min)
                    tmp[1] = MIN_Y + (MAX_Y - MIN_Y) / (y_max - y_min) * (
                            tmp[1] - y_min)
                    tmp[2] = MIN_Z + (MAX_Z - MIN_Z) / (z_max - z_min) * (
                            tmp[2] - z_min)
                    vert_anchor = torch.cat([vert_anchor, tmp.unsqueeze(0)], dim=0)
                    idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh, f[i].unsqueeze(0)], dim=0)
        # mesh = trimesh.Trimesh(vertices=self.up_vertices.cpu().detach().numpy(), faces=face_np, process=False)
        # mesh.show()
        return idx_anchor_in_mesh, vert_anchor
    def set_collar(self, name):

        sl = (self.lines[2]-0.1)/1.2
        is_change = np.zeros(self.up_vertices_deform.shape[0])
        vert_anchor = torch.Tensor().to(self.cur_device)
        idx_anchor_in_mesh = torch.Tensor().to(self.cur_device)

        # sl_ = torch.sqrt((self.collar_x_max - self.collar_x_min) ** 2 + (self.collar_y_max - self.collar_y_min) ** 2)
        # sl_x = (sl / sl_) * (self.collar_x_max - self.collar_x_min)
        # sl_y = (sl / sl_) * (self.collar_y_max - self.collar_y_min)
        # sl_z = (self.collar_z_max - self.collar_z_min) / (self.collar_x_max - self.collar_x_min) * sl_x
        MIN_X = (self.collar_x_max - self.collar_x_min) / 2 + self.collar_x_min - sl / 2
        MAX_X = (self.collar_x_max - self.collar_x_min) / 2 + self.collar_x_min + sl / 2
        MAX_Y = self.collar_y_max
        MIN_Y = self.collar_y_max - self.lines[3]/1.2 +0.02
        # MIN_Z = (self.collar_z_max - self.collar_z_min) / 2 + self.collar_z_min - sl_z / 2
        # MAX_Z = (self.collar_z_max - self.collar_z_min) / 2 + self.collar_z_min + sl_z / 2
        for f in self.collar_np:
            for i in range(3):
                if is_change[f[i]] == 0:
                    is_change[f[i]] = 1
                    tmp = self.up_vertices_deform[f[i]]
                    tmp[0] = MIN_X + (MAX_X - MIN_X) / (self.collar_x_max - self.collar_x_min) * (
                            tmp[0] - self.collar_x_min)
                    tmp[1] = MIN_Y + (MAX_Y - MIN_Y) / (self.collar_y_max - self.collar_y_min) * (
                            tmp[1] - self.collar_y_min)
                    # tmp[2] = MIN_Z + (MAX_Z - MIN_Z) / (self.collar_z_max - self.collar_z_min) * (
                    #         tmp[2] - self.collar_z_min)
                    vert_anchor = torch.cat([vert_anchor, tmp.unsqueeze(0)], dim=0)
                    idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh, f[i].unsqueeze(0)], dim=0)
        # mesh = trimesh.Trimesh(vertices=self.up_vertices.cpu().detach().numpy(), faces=face_np, process=False)
        # mesh.show()
        return idx_anchor_in_mesh, vert_anchor

        # mesh = trimesh.Trimesh(vertices=self.up_vertices.cpu().detach().numpy(), faces=torch.Tensor(face_np).cpu().detach().numpy(), process=False)
        # mesh.show()
    def set_ClothingLength(self, name):

        waist_np = self.up_waist_np

        sl = self.lines[4]
        is_change = np.zeros(self.up_vertices_deform.shape[0])
        vert_anchor = torch.Tensor().to(self.cur_device)
        idx_anchor_in_mesh = torch.Tensor().to(self.cur_device)

        waist_y = torch.Tensor().to(self.cur_device)
        for f in waist_np:
            for i in range(3):
                waist_y = torch.cat([waist_y, self.up_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)

        waist_y_max = torch.max(waist_y)  # 当前数据最小值

        Length = self.collar_y_min - waist_y_max


        if (Length> sl):
            Move = Length-sl
            for f in waist_np:
                for i in range(3):
                    if is_change[f[i]] == 0:
                        is_change[f[i]] = 1
                        tmp = self.up_vertices_deform[f[i]]
                        tmp[1] += Move
                        vert_anchor = torch.cat([vert_anchor, tmp.unsqueeze(0)], dim=0)
                        idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh, f[i].unsqueeze(0)], dim=0)
        else:
            Move = sl-Length
            for f in waist_np:
                for i in range(3):
                    if is_change[f[i]] == 0:
                        is_change[f[i]] = 1
                        tmp = self.up_vertices_deform[f[i]]
                        tmp[1] -= Move
                        vert_anchor = torch.cat([vert_anchor, tmp.unsqueeze(0)], dim=0)
                        idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh, f[i].unsqueeze(0)], dim=0)

        return idx_anchor_in_mesh, vert_anchor, sl, Length
        # mesh = trimesh.Trimesh(vertices=self.up_vertices.cpu().detach().numpy(), faces=torch.Tensor(waist_np+collar_np).cpu().detach().numpy(), process=False)
        # mesh.show()
    def set_SleeveLength(self, name):
        sleeve_l_np = self.sleeve_l_np
        shoulder_l_np = self.shoulder_l_np
        sleeve_r_np = self.sleeve_r_np
        shoulder_r_np = self.shoulder_r_np

        sl = self.lines[5]-0.03

        is_change = np.zeros(self.up_vertices_deform.shape[0])
        vert_anchor = torch.Tensor().to(self.cur_device)
        idx_anchor_in_mesh = torch.Tensor().to(self.cur_device)

        sleeve_l_x = torch.Tensor().to(self.cur_device)
        sleeve_l_y= torch.Tensor().to(self.cur_device)
        for f in sleeve_l_np:
            for i in range(3):
                sleeve_l_x = torch.cat([sleeve_l_x, self.up_vertices_deform[f[i]][0].unsqueeze(0)], dim=0)
                sleeve_l_y = torch.cat([sleeve_l_y, self.up_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)

        shoulder_l_x = torch.Tensor().to(self.cur_device)
        shoulder_l_y= torch.Tensor().to(self.cur_device)
        for f in shoulder_l_np:
            for i in range(3):
                shoulder_l_x = torch.cat([shoulder_l_x, self.up_vertices_deform[f[i]][0].unsqueeze(0)], dim=0)
                shoulder_l_y = torch.cat([shoulder_l_y, self.up_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)

        sleeve_r_x = torch.Tensor().to(self.cur_device)
        sleeve_r_y = torch.Tensor().to(self.cur_device)
        for f in sleeve_r_np:
            for i in range(3):
                sleeve_r_x = torch.cat([sleeve_r_x, self.up_vertices_deform[f[i]][0].unsqueeze(0)], dim=0)
                sleeve_r_y = torch.cat([sleeve_r_y, self.up_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)

        shoulder_r_x = torch.Tensor().to(self.cur_device)
        shoulder_r_y= torch.Tensor().to(self.cur_device)
        for f in shoulder_r_np:
            for i in range(3):
                shoulder_r_x = torch.cat([shoulder_r_x, self.up_vertices_deform[f[i]][0].unsqueeze(0)], dim=0)
                shoulder_r_y = torch.cat([shoulder_r_y, self.up_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)

        sleeve_l_y_top = torch.max(sleeve_l_y)  # 左边最顶点的y
        sleeve_l_y_top_index = torch.argmax(sleeve_l_y)
        sleeve_l_x_top = sleeve_l_x[sleeve_l_y_top_index]  # 左边最顶点的x

        shoulder_l_y_top = torch.max(shoulder_l_y)  # 左边最顶点的y
        shoulder_l_y_top_index = torch.argmax(shoulder_l_y)
        shoulder_l_x_top = shoulder_l_x[shoulder_l_y_top_index]  # 左边最顶点的x


        sleeve_r_y_top = torch.max(sleeve_r_y)  # 右边最顶点的y
        sleeve_r_y_top_index = torch.argmax(sleeve_r_y)
        sleeve_r_x_top = sleeve_r_x[sleeve_r_y_top_index]  # 右边最顶点的x

        shoulder_r_y_top = torch.max(shoulder_r_y)  # 左边最顶点的y
        shoulder_r_y_top_index = torch.argmax(shoulder_r_y)
        shoulder_r_x_top = shoulder_r_x[shoulder_r_y_top_index]  # 左边最顶点的x

        XL = shoulder_l_x_top - sleeve_l_x_top
        YL = shoulder_l_y_top - sleeve_l_y_top
        XR = sleeve_r_x_top - shoulder_r_x_top
        YR = shoulder_r_y_top - sleeve_r_y_top
        LengthL = torch.sqrt((XL) ** 2 + (YL) ** 2)
        LengthR = torch.sqrt((XR) ** 2 + (YR) ** 2)

        index_num = np.zeros(self.up_vertices_deform.shape[0])
        num = self.up_faces.shape[0]
        ff = 0
        for n in self.up_faces:
            if ff == num:
                break
            ff += 1
            for i in range(3):
                index_num[int(n[i])] += 1

        if (LengthL> sl):
            for f in sleeve_l_np:
                for i in range(3):
                    if is_change[f[i]] == 0 and index_num[f[i]] < 6:
                        is_change[f[i]] = 1
                        tmp = self.up_vertices_deform[f[i]]
                        tmp[0] = shoulder_l_x_top - (sl / LengthL * (shoulder_l_x_top - tmp[0]))
                        vert_anchor = torch.cat([vert_anchor, tmp.unsqueeze(0)], dim=0)
                        idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh, f[i].unsqueeze(0)], dim=0)
        else:
            for f in sleeve_l_np:
                for i in range(3):
                    if is_change[f[i]] == 0 and index_num[f[i]] < 6:
                        is_change[f[i]] = 1
                        tmp = self.up_vertices_deform[f[i]]
                        tmp[0] = shoulder_l_x_top - (sl / LengthL * (shoulder_l_x_top - tmp[0]))
                        vert_anchor = torch.cat([vert_anchor, tmp.unsqueeze(0)], dim=0)
                        idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh, f[i].unsqueeze(0)], dim=0)
        if (LengthR> sl):
            for f in sleeve_r_np:
                for i in range(3):
                    if is_change[f[i]] == 0 and index_num[f[i]] < 6:
                        is_change[f[i]] = 1
                        tmp = self.up_vertices_deform[f[i]]
                        tmp[0] = shoulder_r_x_top + (sl / LengthR * (tmp[0] - shoulder_r_x_top))
                        vert_anchor = torch.cat([vert_anchor, tmp.unsqueeze(0)], dim=0)
                        idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh, f[i].unsqueeze(0)], dim=0)
        else:
            for f in sleeve_r_np:
                for i in range(3):
                    if is_change[f[i]] == 0 and index_num[f[i]] < 6:
                        is_change[f[i]] = 1
                        tmp = self.up_vertices_deform[f[i]]
                        tmp[0] = shoulder_r_x_top + (sl / LengthR * (tmp[0] - shoulder_r_x_top))
                        vert_anchor = torch.cat([vert_anchor, tmp.unsqueeze(0)], dim=0)
                        idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh, f[i].unsqueeze(0)], dim=0)
        # mesh = trimesh.Trimesh(vertices=self.up_vertices.cpu().detach().numpy(), faces=(sleeve_r_np.cpu().detach().numpy()), process=False)
        # mesh.show()
        return idx_anchor_in_mesh, vert_anchor
    # def set_ChestWidth(self, name):
    #     # collar_np = torch.Tensor().to(self.cur_device)
    #     # with open(name + "/collar.txt", "r") as f1:
    #     #     for line in f1.readlines():
    #     #         line = line.split(',')
    #     #         line = list(filter(None, line))
    #     #         for i in line:
    #     #             collar_np = torch.cat([collar_np, self.up_faces[int(i)].unsqueeze(0)], dim=0).int()
    #     # face_np = torch.Tensor().to(self.cur_device)
    #     # with open(name + "/chest.txt", "r") as f:
    #     #     for line in f.readlines():
    #     #         line = line.split(',')
    #     #         line = list(filter(None, line))
    #     #         for i in line:
    #     #             face_np = torch.cat([face_np,self.up_faces[int(i)].unsqueeze(0)],dim=0).int()
    #     # # face_np = np.array(face_np)
    #     # sl = 0.3857937121391296
    #     # set_x = torch.Tensor().to(self.cur_device)
    #     # set_y = torch.Tensor().to(self.cur_device)
    #     # set_z = torch.Tensor().to(self.cur_device)
    #     # collar_y = torch.Tensor().to(self.cur_device)
    #     # is_change = np.zeros(self.up_vertices.shape[0])
    #     # vert_anchor = torch.Tensor().to(self.cur_device)
    #     # idx_anchor_in_mesh = torch.Tensor().to(self.cur_device)
    #     # for f in face_np:
    #     #     for i in range(3):
    #     #         set_x = torch.cat([set_x, self.up_vertices[f[i]][0].unsqueeze(0)], dim=0)
    #     #         set_y = torch.cat([set_y, self.up_vertices[f[i]][1].unsqueeze(0)], dim=0)
    #     #         set_z = torch.cat([set_z, self.up_vertices[f[i]][2].unsqueeze(0)], dim=0)
    #     # for f in collar_np:
    #     #     for i in range(3):
    #     #         collar_y = torch.cat([collar_y, self.up_vertices[f[i]][1].unsqueeze(0)], dim=0)
    #     # x_min = torch.min(set_x)  # 当前数据最大值
    #     # x_max = torch.max(set_x)  # 当前数据最小值
    #     # y_min = torch.min(set_y)  # 当前数据最大值
    #     # y_max = torch.max(set_y)  # 当前数据最小值
    #     # z_min = torch.min(set_z)  # 当前数据最大值
    #     # z_max = torch.max(set_z)  # 当前数据最小值
    #     # collar_y_min = torch.min(collar_y)  # 下摆最高值
    #     # # sl_x = sx_max - x_min
    #     # sl_x = sl
    #     # sl_z = (z_max - z_min) / (x_max - x_min) * sl_x
    #     # MIN_X = (x_max - x_min) / 2 + x_min - sl_x / 2
    #     # MAX_X = (x_max - x_min) / 2 + x_min + sl_x / 2
    #     # MIN_Z = (z_max - z_min) / 2 + z_min - sl_z / 2
    #     # MAX_Z = (z_max - z_min) / 2 + z_min + sl_z / 2
    #     # for f in face_np:
    #     #     for i in range(3):
    #     #         if is_change[f[i]] == 0:
    #     #             is_change[f[i]] = 1
    #     #             tmp = self.up_vertices[f[i]].clone()
    #     #             tmp[0] = MIN_X + (MAX_X - MIN_X) / (x_max - x_min) * (
    #     #                     tmp[0] - x_min)
    #     #             tmp[2] = MIN_Z + (MAX_Z - MIN_Z) / (z_max - z_min) * (
    #     #                     tmp[2] - z_min)
    #     #             if cl > CL:
    #     #                 l = collar_y_min - y_max
    #     #                 L = (l/cl)*CL
    #     #                 move = l - L
    #     #                 tmp[1] += move
    #     #             else:
    #     #                 l = collar_y_min - y_max
    #     #                 L = (l/cl)*CL
    #     #                 move = L - l
    #     #                 tmp[1] -= move
    #     #             vert_anchor = torch.cat([vert_anchor, tmp.unsqueeze(0)], dim=0)
    #     #             idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh, f[i].unsqueeze(0)], dim=0)
    #     sleeve_l_np = self.sleeve_l_np
    #     sleeve_r_np = self.sleeve_r_np
    #
    #     sleeve_l_x = torch.Tensor().to(self.cur_device)
    #     sleeve_l_y = torch.Tensor().to(self.cur_device)
    #     for f in sleeve_l_np:
    #         for i in range(3):
    #             sleeve_l_x = torch.cat([sleeve_l_x, self.up_vertices_deform[f[i]][0].unsqueeze(0)], dim=0)
    #             sleeve_l_y = torch.cat([sleeve_l_y, self.up_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)
    #
    #     sleeve_r_x = torch.Tensor().to(self.cur_device)
    #     for f in sleeve_r_np:
    #         for i in range(3):
    #             sleeve_r_x = torch.cat([sleeve_r_x, self.up_vertices_deform[f[i]][0].unsqueeze(0)], dim=0)
    #
    #     sleeve_l_x_max = torch.max(sleeve_l_x)  # 最左边点的x
    #     sleeve_l_y_min = torch.min(sleeve_l_y)  # 最左边点的x
    #     sleeve_r_x_min = torch.min(sleeve_r_x)  # 最右边点的x
    #
    #     waist_np = self.up_waist_np
    #     chest_np = self.chest_np
    #
    #     waist_y_deform = torch.Tensor().to(self.cur_device)
    #     waist_x_deform = torch.Tensor().to(self.cur_device)
    #     waist_y_orgin = torch.Tensor().to(self.cur_device)
    #     waist_x_orgin = torch.Tensor().to(self.cur_device)
    #     for f in waist_np:
    #         for i in range(3):
    #             waist_y_deform = torch.cat([waist_y_deform, self.up_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)
    #             waist_x_deform = torch.cat([waist_x_deform, self.up_vertices_deform[f[i]][0].unsqueeze(0)], dim=0)
    #             waist_y_orgin = torch.cat([waist_y_orgin, self.up_vertices[f[i]][1].unsqueeze(0)], dim=0)
    #             waist_x_orgin = torch.cat([waist_x_orgin, self.up_vertices[f[i]][0].unsqueeze(0)], dim=0)
    #     waist_x_deform_min = torch.min(waist_x_deform)
    #     waist_x_deform_max = torch.max(waist_x_deform)
    #     waist_x_orgin_min = torch.min(waist_x_orgin)
    #     waist_x_orgin_max = torch.max(waist_x_orgin)
    #     waist_y_deform_max = torch.max(waist_y_deform)
    #
    #     is_change = np.zeros(self.up_vertices.shape[0])
    #     vert_anchor = torch.Tensor().to(self.cur_device)
    #     idx_anchor_in_mesh = torch.Tensor().to(self.cur_device)
    #     chest_y = torch.Tensor().to(self.cur_device)
    #     chest_l_x = torch.Tensor().to(self.cur_device)
    #     chest_r_x = torch.Tensor().to(self.cur_device)
    #     chest_l_y = torch.Tensor().to(self.cur_device)
    #     chest_r_y = torch.Tensor().to(self.cur_device)
    #     for f in chest_np:
    #         for i in range(3):
    #             if self.up_vertices_deform[f[i]][0] <0:
    #                 chest_l_x = torch.cat([chest_l_x, self.up_vertices_deform[f[i]][0].unsqueeze(0)], dim=0)
    #                 chest_l_y = torch.cat([chest_l_y, self.up_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)
    #             else:
    #                 chest_r_x = torch.cat([chest_r_x, self.up_vertices_deform[f[i]][0].unsqueeze(0)], dim=0)
    #                 chest_r_y = torch.cat([chest_r_y, self.up_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)
    #     chest_l_x_max = torch.max(chest_l_x)
    #     chest_l_x_min = torch.min(chest_l_x)
    #     chest_r_x_max = torch.max(chest_r_x)
    #     chest_r_x_min = torch.min(chest_r_x)
    #     chest_l_y_max = torch.max(chest_l_y)
    #     chest_l_y_min = torch.min(chest_l_y)
    #
    #
    #     for f in chest_np:
    #         for i in range(3):
    #             if is_change[f[i]] == 0:
    #                 is_change[f[i]] = 1
    #                 tmp = self.up_vertices_deform[f[i]]
    #                 if tmp[0] < 0:
    #                     if tmp[0] < 0:
    #                         MIN_Y = sleeve_l_y_min
    #                     else:
    #                         MIN_Y = waist_y_deform_max
    #                     MAX_Y = chest_l_y_max
    #                     tmp[1] = MIN_Y + (MAX_Y - MIN_Y) / (chest_l_y_max - chest_l_y_min) * (tmp[1] - chest_l_y_min)
    #                     MIN_X = sleeve_l_x_max
    #                     MAX_X = chest_l_x_max - (waist_x_orgin_min - waist_x_deform_min)
    #                     tmp[0] = MIN_X + (MAX_X - MIN_X) / (chest_l_x_max - chest_l_x_min) * (tmp[0] - chest_l_x_min)
    #                 else:
    #                     MAX_X = sleeve_r_x_min
    #                     MIN_X = chest_r_x_min + (waist_x_deform_max - waist_x_orgin_max)
    #                     tmp[0] = MIN_X + (MAX_X - MIN_X) / (chest_r_x_max - chest_r_x_min) * (tmp[0] - chest_r_x_min)
    #                 vert_anchor = torch.cat([vert_anchor, tmp.unsqueeze(0)], dim=0)
    #                 idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh, f[i].unsqueeze(0)], dim=0)
    #     # mesh = trimesh.Trimesh(vertices=self.up_vertices_deform.cpu().detach().numpy(), faces=chest_np.cpu().detach().numpy(), process=False)
    #     # mesh.show()
    #     return idx_anchor_in_mesh, vert_anchor
    def set_bottom_waist(self, name):
        face_np = torch.Tensor().to(self.cur_device)
        name = "/mnt/data/Project/GFLNet-master/gfl-main/lib/" + name
        with open(name + "/waist.txt", "r") as f:
            for line in f.readlines():
                line = line.split(',')
                line = list(filter(None, line))
                for i in line:
                    face_np = torch.cat([face_np,self.bottom_faces[int(i)].unsqueeze(0)],dim=0).int()
        # face_np = np.array(face_np)
        sl = self.lines[6]#0.37253114581108093

        set_x = torch.Tensor().to(self.cur_device)
        set_y = torch.Tensor().to(self.cur_device)
        set_z = torch.Tensor().to(self.cur_device)
        is_change = np.zeros(self.bottom_vertices_deform.shape[0])
        vert_anchor = torch.Tensor().to(self.cur_device)
        idx_anchor_in_mesh = torch.Tensor().to(self.cur_device)
        for f in face_np:
            for i in range(3):
                set_x = torch.cat([set_x, self.bottom_vertices_deform[f[i]][0].unsqueeze(0)], dim=0)
                set_y = torch.cat([set_y, self.bottom_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)
                set_z = torch.cat([set_z, self.bottom_vertices_deform[f[i]][2].unsqueeze(0)], dim=0)

        x_min = torch.min(set_x)  # 当前数据最大值
        x_max = torch.max(set_x)  # 当前数据最小值
        y_min = torch.min(set_y)  # 当前数据最大值
        y_max = torch.max(set_y)  # 当前数据最小值
        z_min = torch.min(set_z)  # 当前数据最大值
        z_max = torch.max(set_z)  # 当前数据最小值
        sl_ = torch.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
        sl_x = (sl / sl_) * (x_max - x_min)
        sl_y = (sl / sl_) * (y_max - y_min)
        sl_z = (z_max - z_min) / (x_max - x_min) * sl_x
        MIN_X = (x_max - x_min) / 2 + x_min - sl_x / 2
        MAX_X = (x_max - x_min) / 2 + x_min + sl_x / 2
        MIN_Y = (y_max - y_min) / 2 + y_min - sl_y / 2
        MAX_Y = (y_max - y_min) / 2 + y_min + sl_y / 2
        MIN_Z = (z_max - z_min) / 2 + z_min - sl_z / 2
        MAX_Z = (z_max - z_min) / 2 + z_min + sl_z / 2

        scale_x = (MAX_X - MIN_X) / (x_max - x_min)
        for i in range(self.bottom_vertices.shape[0]):
            self.bottom_vertices[i][0] *= scale_x
            self.bottom_vertices[i][2] *= scale_x
        self.bottom_vertices_deform = self.bottom_vertices.clone()

        for f in face_np:
            for i in range(3):
                if is_change[f[i]] == 0:
                    is_change[f[i]] = 1
                    tmp = self.bottom_vertices_deform[f[i]]
                    # tmp[0] = MIN_X + (MAX_X - MIN_X) / (x_max - x_min) * (
                    #         tmp[0] - x_min)
                    # tmp[1] = MIN_Y + (MAX_Y - MIN_Y) / (y_max - y_min) * (
                    #         tmp[1] - y_min)
                    # tmp[2] = MIN_Z + (MAX_Z - MIN_Z) / (z_max - z_min) * (
                    #         tmp[2] - z_min)
                    vert_anchor = torch.cat([vert_anchor, tmp.unsqueeze(0)], dim=0)
                    idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh, f[i].unsqueeze(0)], dim=0)
        # mesh = trimesh.Trimesh(vertices=self.bottom_vertices.cpu().detach().numpy(), faces=face_np, process=False)
        # mesh.show()
        return idx_anchor_in_mesh, vert_anchor
        # mesh = trimesh.Trimesh(vertices=self.up_vertices.cpu().detach().numpy(), faces=torch.Tensor(face_np).cpu().detach().numpy(), process=False)
        # mesh.show()
    def set_bottom_trouser_l(self, name):
        name = "/mnt/data/Project/GFLNet-master/gfl-main/lib/" + name
        face_np = torch.Tensor().to(self.cur_device)
        with open(name + "/bottom_trouser_l.txt", "r") as f:
            for line in f.readlines():
                line = line.split(',')
                line = list(filter(None, line))
                for i in line:
                    face_np = torch.cat([face_np,self.bottom_faces[int(i)].unsqueeze(0)],dim=0).int()
        sl = self.lines[7]#0.23828348517417908
        set_x = torch.Tensor().to(self.cur_device)
        set_y = torch.Tensor().to(self.cur_device)
        set_z = torch.Tensor().to(self.cur_device)
        is_change = np.zeros(self.bottom_vertices_deform.shape[0])
        vert_anchor = torch.Tensor().to(self.cur_device)
        idx_anchor_in_mesh = torch.Tensor().to(self.cur_device)
        for f in face_np:
            for i in range(3):
                set_x = torch.cat([set_x, self.bottom_vertices_deform[f[i]][0].unsqueeze(0)], dim=0)
                set_y = torch.cat([set_y, self.bottom_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)
                set_z = torch.cat([set_z, self.bottom_vertices_deform[f[i]][2].unsqueeze(0)], dim=0)

        x_min = torch.min(set_x)  # 当前数据最大值
        x_max = torch.max(set_x)  # 当前数据最小值
        y_min = torch.min(set_y)  # 当前数据最大值
        y_max = torch.max(set_y)  # 当前数据最小值
        z_min = torch.min(set_z)  # 当前数据最大值
        z_max = torch.max(set_z)  # 当前数据最小值
        sl_ = torch.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
        sl_x = (sl / sl_) * (x_max - x_min)
        sl_y = (sl / sl_) * (y_max - y_min)
        sl_z = (z_max - z_min) / (x_max - x_min) * sl_x
        MIN_X = (x_max - x_min) / 2 + x_min - sl_x / 2
        MAX_X = (x_max - x_min) / 2 + x_min + sl_x / 2
        MIN_Y = (y_max - y_min) / 2 + y_min - sl_y / 2
        MAX_Y = (y_max - y_min) / 2 + y_min + sl_y / 2
        MIN_Z = (z_max - z_min) / 2 + z_min - sl_z / 2
        MAX_Z = (z_max - z_min) / 2 + z_min + sl_z / 2
        for f in face_np:
            for i in range(3):
                if is_change[f[i]] == 0:
                    is_change[f[i]] = 1
                    tmp = self.bottom_vertices_deform[f[i]]
                    tmp[0] = MIN_X + (MAX_X - MIN_X) / (x_max - x_min) * (
                            tmp[0] - x_min)
                    if tmp[0] > x_max:
                        tmp[0] = x_max
                    tmp[1] = MIN_Y + (MAX_Y - MIN_Y) / (y_max - y_min) * (
                            tmp[1] - y_min)
                    tmp[2] = MIN_Z + (MAX_Z - MIN_Z) / (z_max - z_min) * (
                            tmp[2] - z_min)
                    vert_anchor = torch.cat([vert_anchor, tmp.unsqueeze(0)], dim=0)
                    idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh, f[i].unsqueeze(0)], dim=0)
        # mesh = trimesh.Trimesh(vertices=self.bottom_vertices.cpu().detach().numpy(), faces=torch.Tensor(face_np).cpu().detach().numpy(), process=False)
        # mesh.show()
        return idx_anchor_in_mesh, vert_anchor
    def set_bottom_trouser_r(self, name):
        name = "/mnt/data/Project/GFLNet-master/gfl-main/lib/" + name
        face_np = torch.Tensor().to(self.cur_device)
        with open(name + "/bottom_trouser_r.txt", "r") as f:
            for line in f.readlines():
                line = line.split(',')
                line = list(filter(None, line))
                for i in line:
                    face_np = torch.cat([face_np,self.bottom_faces[int(i)].unsqueeze(0)],dim=0).int()
        # face_np = np.array(face_np)
        sl = self.lines[7]#0.2024664580821991
        set_x = torch.Tensor().to(self.cur_device)
        set_y = torch.Tensor().to(self.cur_device)
        set_z = torch.Tensor().to(self.cur_device)
        is_change = np.zeros(self.bottom_vertices_deform.shape[0])
        vert_anchor = torch.Tensor().to(self.cur_device)
        idx_anchor_in_mesh = torch.Tensor().to(self.cur_device)
        for f in face_np:
            for i in range(3):
                set_x = torch.cat([set_x, self.bottom_vertices_deform[f[i]][0].unsqueeze(0)], dim=0)
                set_y = torch.cat([set_y, self.bottom_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)
                set_z = torch.cat([set_z, self.bottom_vertices_deform[f[i]][2].unsqueeze(0)], dim=0)

        x_min = torch.min(set_x)  # 当前数据最大值
        x_max = torch.max(set_x)  # 当前数据最小值
        y_min = torch.min(set_y)  # 当前数据最大值
        y_max = torch.max(set_y)  # 当前数据最小值
        z_min = torch.min(set_z)  # 当前数据最大值
        z_max = torch.max(set_z)  # 当前数据最小值
        sl_ = torch.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
        sl_x = (sl / sl_) * (x_max - x_min)
        sl_y = (sl / sl_) * (y_max - y_min)
        sl_z = (z_max - z_min) / (x_max - x_min) * sl_x
        MIN_X = (x_max - x_min) / 2 + x_min - sl_x / 2
        MAX_X = (x_max - x_min) / 2 + x_min + sl_x / 2
        MIN_Y = (y_max - y_min) / 2 + y_min - sl_y / 2
        MAX_Y = (y_max - y_min) / 2 + y_min + sl_y / 2
        MIN_Z = (z_max - z_min) / 2 + z_min - sl_z / 2
        MAX_Z = (z_max - z_min) / 2 + z_min + sl_z / 2
        for f in face_np:
            for i in range(3):
                if is_change[f[i]] == 0:
                    is_change[f[i]] = 1
                    tmp = self.bottom_vertices_deform[f[i]]
                    tmp[0] = MIN_X + (MAX_X - MIN_X) / (x_max - x_min) * (
                            tmp[0] - x_min)
                    if tmp[0] < x_min:
                        tmp[0] = x_min
                    tmp[1] = MIN_Y + (MAX_Y - MIN_Y) / (y_max - y_min) * (
                            tmp[1] - y_min)
                    tmp[2] = MIN_Z + (MAX_Z - MIN_Z) / (z_max - z_min) * (
                            tmp[2] - z_min)
                    vert_anchor = torch.cat([vert_anchor, tmp.unsqueeze(0)], dim=0)
                    idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh, f[i].unsqueeze(0)], dim=0)
        # mesh = trimesh.Trimesh(vertices=self.bottom_vertices.cpu().detach().numpy(), faces=torch.Tensor(face_np).cpu().detach().numpy(), process=False)
        # mesh.show()
        return idx_anchor_in_mesh, vert_anchor
    def set_TrousersLength(self, name):
        name = "/mnt/data/Project/GFLNet-master/gfl-main/lib/" + name
        bottom_trouser_l_np = torch.Tensor().to(self.cur_device)
        with open(name + "/bottom_trouser_l.txt", "r") as f1:
            for line in f1.readlines():
                line = line.split(',')
                line = list(filter(None, line))
                for i in line:
                    bottom_trouser_l_np = torch.cat([bottom_trouser_l_np,self.bottom_faces[int(i)].unsqueeze(0)],dim=0).int()
        bottom_trouser_r_np = torch.Tensor().to(self.cur_device)
        with open(name + "/bottom_trouser_r.txt", "r") as f1:
            for line in f1.readlines():
                line = line.split(',')
                line = list(filter(None, line))
                for i in line:
                    bottom_trouser_r_np = torch.cat([bottom_trouser_r_np,self.bottom_faces[int(i)].unsqueeze(0)],dim=0).int()
        waist_np = torch.Tensor().to(self.cur_device)
        with open(name + "/waist.txt", "r") as f2:
            for line in f2.readlines():
                line = line.split(',')
                line = list(filter(None, line))
                for i in line:
                    waist_np = torch.cat([waist_np,self.bottom_faces[int(i)].unsqueeze(0)],dim=0).int()

        sl = self.lines[8]#0.4930139521383385
        waist_x = torch.Tensor().to(self.cur_device)
        waist_y = torch.Tensor().to(self.cur_device)
        is_change = np.zeros(self.bottom_vertices_deform.shape[0])
        vert_anchor = torch.Tensor().to(self.cur_device)
        idx_anchor_in_mesh = torch.Tensor().to(self.cur_device)
        for f in waist_np:
            for i in range(3):
                waist_x = torch.cat([waist_x, self.bottom_vertices_deform[f[i]][0].unsqueeze(0)], dim=0)
                waist_y = torch.cat([waist_y, self.bottom_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)

        bottom_trouser_l_x = torch.Tensor().to(self.cur_device)
        bottom_trouser_l_y = torch.Tensor().to(self.cur_device)
        for f in bottom_trouser_l_np:
            for i in range(3):
                bottom_trouser_l_x = torch.cat([bottom_trouser_l_x, self.bottom_vertices_deform[f[i]][0].unsqueeze(0)], dim=0)
                bottom_trouser_l_y = torch.cat([bottom_trouser_l_y, self.bottom_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)

        bottom_trouser_r_x = torch.Tensor().to(self.cur_device)
        bottom_trouser_r_y = torch.Tensor().to(self.cur_device)
        for f in bottom_trouser_r_np:
            for i in range(3):
                bottom_trouser_r_x = torch.cat([bottom_trouser_r_x, self.bottom_vertices_deform[f[i]][0].unsqueeze(0)], dim=0)
                bottom_trouser_r_y = torch.cat([bottom_trouser_r_y, self.bottom_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)

        waist_x_max = torch.max(waist_x)  # 最右边点的x
        waist_x_min = torch.min(waist_x)  # 最左边点的x
        waist_x_max_index = torch.argmax(waist_x)
        waist_x_min_index = torch.argmin(waist_x)
        waist_y_max = waist_y[waist_x_max_index]  # 最右边点的y
        waist_y_min = waist_y[waist_x_min_index]  # 最左边点的y

        bottom_trouser_l_x_min = torch.min(bottom_trouser_l_x)  # 最左边点的x
        bottom_trouser_l_x_min_index = torch.argmin(bottom_trouser_l_x)
        bottom_trouser_l_y_min = bottom_trouser_l_y[bottom_trouser_l_x_min_index]  # 最左边点的y

        bottom_trouser_r_x_max = torch.max(bottom_trouser_r_x) # 最右边点的x
        bottom_trouser_r_x_max_index = torch.argmax(bottom_trouser_r_x)
        bottom_trouser_r_y_max = bottom_trouser_r_y[bottom_trouser_r_x_max_index] # 最右边点的y

        XL = bottom_trouser_l_x_min - waist_x_min #负值
        YL = waist_y_min - bottom_trouser_l_y_min
        LengthL = torch.sqrt((XL) ** 2 + (YL) ** 2)

        XR = bottom_trouser_r_x_max - waist_x_max
        YR = waist_y_max - bottom_trouser_r_y_max
        LengthR = torch.sqrt((XR) ** 2 + (YR) ** 2)

        num = 0
        if (LengthL > sl):
            MoveL = LengthL - sl
            yl_move = (MoveL / LengthL) * YL
            xl_move = (MoveL / LengthL) * XL
            for f in bottom_trouser_l_np:
                for i in range(3):
                    if is_change[f[i]] == 0:
                        is_change[f[i]] = 1
                        tmp = self.bottom_vertices_deform[f[i]]
                        tmp[0] -= xl_move
                        tmp[1] += yl_move
                        vert_anchor = torch.cat([vert_anchor, tmp.unsqueeze(0)], dim=0)
                        idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh, f[i].unsqueeze(0)], dim=0)
                        num += 1
        else:
            MoveL = sl - LengthL
            yl_move = (MoveL / LengthL) * YL
            xl_move = (MoveL / LengthL) * XL
            for f in bottom_trouser_l_np:
                for i in range(3):
                    if is_change[f[i]] == 0:
                        is_change[f[i]] = 1
                        tmp = self.bottom_vertices_deform[f[i]]
                        tmp[0] += xl_move
                        tmp[1] -= yl_move
                        vert_anchor = torch.cat([vert_anchor, tmp.unsqueeze(0)], dim=0)
                        idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh, f[i].unsqueeze(0)], dim=0)
                        num += 1
        num = 0
        if (LengthR > sl):
            MoveR = LengthR - sl
            yr_move = (MoveR / LengthR) * YR
            xr_move = (MoveR / LengthR) * XR
            for f in bottom_trouser_r_np:
                for i in range(3):
                    if is_change[f[i]] == 0:
                        is_change[f[i]] = 1
                        tmp = self.bottom_vertices_deform[f[i]]
                        tmp[0] += xr_move
                        tmp[1] += yr_move
                        vert_anchor = torch.cat([vert_anchor, tmp.unsqueeze(0)], dim=0)
                        idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh, f[i].unsqueeze(0)], dim=0)
                        num += 1
        else:
            MoveR = sl - LengthR
            yr_move = (MoveR / LengthR) * YR
            xr_move = (MoveR / LengthR) * XR
            for f in bottom_trouser_r_np:
                for i in range(3):
                    if is_change[f[i]] == 0:
                        is_change[f[i]] = 1
                        tmp = self.bottom_vertices_deform[f[i]]
                        tmp[0] -= xr_move
                        tmp[1] -= yr_move
                        vert_anchor = torch.cat([vert_anchor, tmp.unsqueeze(0)], dim=0)
                        idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh, f[i].unsqueeze(0)], dim=0)
                        num += 1

        # mesh = trimesh.Trimesh(vertices=self.bottom_vertices.cpu().detach().numpy(), faces=(bottom_trouser_l_np),
        #                        process=False)
        # mesh.show()
        return idx_anchor_in_mesh, vert_anchor
    def set_CrotchLength(self, name):
        name = "/mnt/data/Project/GFLNet-master/gfl-main/lib/" + name
        crotch_np = torch.Tensor().to(self.cur_device)
        with open(name + "/crotch.txt", "r") as f1:
            for line in f1.readlines():
                line = line.split(',')
                line = list(filter(None, line))
                for i in line:
                    crotch_np = torch.cat([crotch_np,self.bottom_faces[int(i)].unsqueeze(0)],dim=0).int()
        waist_np = torch.Tensor().to(self.cur_device)
        with open(name + "/waist.txt", "r") as f2:
            for line in f2.readlines():
                line = line.split(',')
                line = list(filter(None, line))
                for i in line:
                    waist_np = torch.cat([waist_np,self.bottom_faces[int(i)].unsqueeze(0)],dim=0).int()

        sl = self.lines[9]#0.20416441559791565
        crotch_y = torch.Tensor().to(self.cur_device)
        is_change = np.zeros(self.bottom_vertices_deform.shape[0])
        vert_anchor = torch.Tensor().to(self.cur_device)
        idx_anchor_in_mesh = torch.Tensor().to(self.cur_device)
        for f in crotch_np:
            for i in range(3):
                crotch_y = torch.cat([crotch_y, self.bottom_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)

        waist_y = torch.Tensor().to(self.cur_device)
        for f in waist_np:
            for i in range(3):
                waist_y = torch.cat([waist_y, self.bottom_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)


        waist_y_min = torch.min(waist_y)  # 当前数据最小值
        crotch_y_max = torch.max(crotch_y)  # 当前数据最大值

        Length = waist_y_min - crotch_y_max

        if (Length> sl):
            Move = Length-sl
            for f in crotch_np:
                for i in range(3):
                    if is_change[f[i]] == 0:
                        is_change[f[i]] = 1
                        tmp = self.bottom_vertices_deform[f[i]]
                        tmp[1] += Move
                        vert_anchor = torch.cat([vert_anchor, tmp.unsqueeze(0)], dim=0)
                        idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh, f[i].unsqueeze(0)], dim=0)
        else:
            Move = sl-Length
            for f in crotch_np:
                for i in range(3):
                    if is_change[f[i]] == 0:
                        is_change[f[i]] = 1
                        tmp = self.bottom_vertices_deform[f[i]]
                        tmp[1] -= Move
                        vert_anchor = torch.cat([vert_anchor, tmp.unsqueeze(0)], dim=0)
                        idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh, f[i].unsqueeze(0)], dim=0)
        # mesh = trimesh.Trimesh(vertices=self.bottom_vertices.cpu().detach().numpy(), faces=self.bottom_faces.cpu().detach().numpy(),process=False)
        # mesh.show()

        return idx_anchor_in_mesh, vert_anchor
    def set_skirt(self, name):
        name = "/mnt/data/Project/GFLNet-master/gfl-main/lib/" + name
        face_np = torch.Tensor().to(self.cur_device)
        with open(name + "/skirt.txt", "r") as f:
            for line in f.readlines():
                line = line.split(',')
                line = list(filter(None, line))
                for i in line:
                    face_np = torch.cat([face_np,self.bottom_faces[int(i)].unsqueeze(0)],dim=0).int()
        sl = self.lines[7]
        set_x = torch.Tensor().to(self.cur_device)
        set_y = torch.Tensor().to(self.cur_device)
        set_z = torch.Tensor().to(self.cur_device)
        is_change = np.zeros(self.bottom_vertices_deform.shape[0])
        vert_anchor = torch.Tensor().to(self.cur_device)
        idx_anchor_in_mesh = torch.Tensor().to(self.cur_device)
        for f in face_np:
            for i in range(3):
                set_x = torch.cat([set_x, self.bottom_vertices_deform[f[i]][0].unsqueeze(0)], dim=0)
                set_y = torch.cat([set_y, self.bottom_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)
                set_z = torch.cat([set_z, self.bottom_vertices_deform[f[i]][2].unsqueeze(0)], dim=0)

        x_min = torch.min(set_x)  # 当前数据最大值
        x_max = torch.max(set_x)  # 当前数据最小值
        y_min = torch.min(set_y)  # 当前数据最大值
        y_max = torch.max(set_y)  # 当前数据最小值
        z_min = torch.min(set_z)  # 当前数据最大值
        z_max = torch.max(set_z)  # 当前数据最小值
        sl_ = torch.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
        sl_x = (sl / sl_) * (x_max - x_min)
        sl_y = (sl / sl_) * (y_max - y_min)
        sl_z = (z_max - z_min) / (x_max - x_min) * sl_x
        MIN_X = (x_max - x_min) / 2 + x_min - sl_x / 2
        MAX_X = (x_max - x_min) / 2 + x_min + sl_x / 2
        MIN_Y = (y_max - y_min) / 2 + y_min - sl_y / 2
        MAX_Y = (y_max - y_min) / 2 + y_min + sl_y / 2
        MIN_Z = (z_max - z_min) / 2 + z_min - sl_z / 2
        MAX_Z = (z_max - z_min) / 2 + z_min + sl_z / 2
        for f in face_np:
            for i in range(3):
                if is_change[f[i]] == 0:
                    is_change[f[i]] = 1
                    tmp = self.bottom_vertices_deform[f[i]]
                    tmp[0] = MIN_X + (MAX_X - MIN_X) / (x_max - x_min) * (
                            tmp[0] - x_min)
                    tmp[1] = MIN_Y + (MAX_Y - MIN_Y) / (y_max - y_min) * (
                            tmp[1] - y_min)
                    tmp[2] = MIN_Z + (MAX_Z - MIN_Z) / (z_max - z_min) * (
                            tmp[2] - z_min)
                    vert_anchor = torch.cat([vert_anchor, tmp.unsqueeze(0)], dim=0)
                    idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh, f[i].unsqueeze(0)],dim=0)
        # mesh = trimesh.Trimesh(vertices=self.bottom_vertices.cpu().detach().numpy(), faces=face_np, process=False)
        # mesh.show()
        return idx_anchor_in_mesh, vert_anchor
    def set_SkirtLength(self, name):
        name = "/mnt/data/Project/GFLNet-master/gfl-main/lib/" + name
        skirt_np = torch.Tensor().to(self.cur_device)
        with open(name + "/skirt.txt", "r") as f1:
            for line in f1.readlines():
                line = line.split(',')
                line = list(filter(None, line))
                for i in line:
                    skirt_np = torch.cat([skirt_np,self.bottom_faces[int(i)].unsqueeze(0)],dim=0).int()
        waist_np = torch.Tensor().to(self.cur_device)
        with open(name + "/waist.txt", "r") as f2:
            for line in f2.readlines():
                line = line.split(',')
                line = list(filter(None, line))
                for i in line:
                    waist_np = torch.cat([waist_np,self.bottom_faces[int(i)].unsqueeze(0)],dim=0).int()

        sl = self.lines[8]
        skirt_y = torch.Tensor().to(self.cur_device)
        is_change = np.zeros(self.bottom_vertices_deform.shape[0])
        vert_anchor = torch.Tensor().to(self.cur_device)
        idx_anchor_in_mesh = torch.Tensor().to(self.cur_device)
        for f in skirt_np:
            for i in range(3):
                skirt_y = torch.cat([skirt_y, self.bottom_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)

        waist_y = torch.Tensor().to(self.cur_device)
        for f in waist_np:
            for i in range(3):
                waist_y = torch.cat([waist_y, self.bottom_vertices_deform[f[i]][1].unsqueeze(0)], dim=0)


        waist_y_min = torch.min(waist_y)  # 当前数据最小值
        skirt_y_min = torch.min(skirt_y)  # 当前数据最小值

        Length = waist_y_min - skirt_y_min

        num = 0
        if (Length> sl):
            Move = Length-sl
            for f in skirt_np:
                for i in range(3):
                    if is_change[f[i]] == 0:
                        is_change[f[i]] = 1
                        tmp = self.bottom_vertices_deform[f[i]]
                        tmp[1] += Move
                        vert_anchor = torch.cat([vert_anchor, tmp.unsqueeze(0)], dim=0)
                        idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh, f[i].unsqueeze(0)], dim=0)
                        num += 1
        else:
            Move = sl-Length
            for f in skirt_np:
                for i in range(3):
                    if is_change[f[i]] == 0:
                        is_change[f[i]] = 1
                        tmp = self.bottom_vertices_deform[f[i]]
                        tmp[1] -= Move
                        vert_anchor = torch.cat([vert_anchor, tmp.unsqueeze(0)], dim=0)
                        idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh, f[i].unsqueeze(0)], dim=0)
                        num += 1
        # mesh = trimesh.Trimesh(vertices=self.bottom_vertices.cpu().detach().numpy(), faces=self.bottom_faces.cpu().detach().numpy(),process=False)
        # mesh.show()

        return idx_anchor_in_mesh, vert_anchor

    def set_Pose(self,v_cloth, indice, v_smpl, pose_params, W, T, index):
        garmentvnums = [4248, 4258, 5327, 3721, 5404, 2818]
        correspondances = []
        with open(str(garmentvnums[index]) + ".txt", "r") as f:
            for line in f.readlines():
                line = line.split(',')
                line = list(filter(None, line))
                line = [int(i) for i in line]
                correspondances.append(line)
        correspondances = torch.Tensor(sum(correspondances, [])).clone().type(torch.int64).to(self.cur_device)
        indice = torch.from_numpy(indice).to(self.cur_device)
        correspondance = torch.index_select(correspondances, 0, indice)

        v_cloth2 = torch.from_numpy(v_cloth).to(self.cur_device).unsqueeze(0)
        vertices = self.up_vertices.unsqueeze(0)
        num_batch = 1
        # print("11",v_cloth)
        # print("22",v_smpl)
        # Calculate closest SMPL vertex for each vertex of the cloth mesh
        # with torch.no_grad():
        #     dists = ((v_smpl.unsqueeze(1) - v_cloth.unsqueeze(2)) ** 2).sum(-1)
        #     correspondance = torch.argmin(dists, 2)
        # print(pose_params.shape)
        v_posed_vertices = pose_params[0, correspondances] + vertices
        v_posed_homo_vertices = torch.cat([v_posed_vertices, torch.ones(num_batch, v_posed_vertices.shape[1], 1, device=self.cur_device)], dim=2)
        v_homo_vertices = torch.matmul(T[0, correspondances], torch.unsqueeze(v_posed_homo_vertices, -1))
        verts_vertices = v_homo_vertices[:, :, :3, 0]

        v_posed_cloth = pose_params[0, correspondance] + v_cloth2
        v_posed_homo_cloth = torch.cat([v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device=self.cur_device)], dim=2)
        v_homo_cloth = torch.matmul(T[0, correspondance], torch.unsqueeze(v_posed_homo_cloth, -1))
        verts_cloth = v_homo_cloth[:, :, :3, 0]

        return verts_cloth[0].cpu().detach().numpy(),verts_vertices[0].cpu().detach().numpy()#verts_cloth[0].cpu().detach().numpy()

    def repose(self, ps, type):
        ws = torch.zeros([ps.shape[0],24]).to(self.cur_device)
        if type ==1 :
            ws[:156,0]=1
            ws[156:204, 19] = 1
            ws[204:252, 18] = 1
            ws[252:500, 9] = 1
            ws[500:603, 13:15] = 1
        # Js_transformed, A = batch_global_rigid_transformation(Rs, Js, self.smpl.parents, rotate_base = False)
        trans = ws.matmul(self.JS)
        T = ws.matmul(self.A.reshape(24, 16))
        T = T.reshape(-1, 4, 4)
        ps = torch.cat((ps, ps.new_ones(ps.shape[0], 1)), dim=-1).unsqueeze(-1)
        ps = torch.matmul(T, ps).squeeze(-1)
        return ps[:, 0:3]-trans

    def laplacian(self, vertices, faces, idx_anchor_in_mesh, vert_anchor):
        # verts是顶点 numpy 数组 ：[nv, 3], nv是顶点数量
        # faces是面片 numpy 数组： [nf, 3]，nf是面片数量
        # idx_verts ： verts顶点的索引 可以用 np.arrange(verts.shape[0])表示
        # vert_anchor: verts中锚点将要变换到新的位置（欧式坐标系）
        # idx_anchor_in_mesh： 锚点在idx_verts中的索引
        lap = LapMeshDeform(vertices, faces)
        L = lap.uniform_laplacian()
        # L  = lap.cot_laplacian(area_normalize=False)
        verts_sparse = sparse.lil_matrix(vertices)
        # delta矩阵
        # delta = L.dot(verts_sparse)
        delta = L.dot(verts_sparse)
        # delta = self.maritx_mutil(L, verts_sparse)
        # add anchor points
        # 锚点在整体网格索引idx_verts中的索引
        idx_verts = np.arange(vertices.shape[0])
        real_idx = idx_verts[idx_anchor_in_mesh]

        # 锚点约束项权重
        w_anchor = 1
        # 拉普拉斯矩阵的锚点
        L_anchor = sparse.lil_matrix((vert_anchor.shape[0], vertices.shape[0]), dtype=np.float32)
        for i in range(real_idx.shape[0]):
            L_anchor[i, real_idx[i]] = w_anchor
        L_anchor = L_anchor.tocsr()

        # δ矩阵的锚点
        delta_anchor = vert_anchor * w_anchor
        delta_anchor = sparse.csr_matrix(delta_anchor)
        # 构造矩阵A
        A = sparse.vstack((L, L_anchor))
        # 构造矩阵B
        B = sparse.vstack((delta, delta_anchor))
        # B = sparse.lil_matrix(B)

        # 解超定线性方程
        # A =  spsp.csr_matrix(A)
        factor = cholesky_AAt(A.T)
        x = factor(A.T * B)
        # new_verts就是最终形变结果
        new_verts = x.toarray()
        # mesh = trimesh.Trimesh(vertices=new_verts, faces=faces, process=False)
        # mesh.show()

        return new_verts
    def forward(self, lines=None, vertices=None, faces=None, gtypes=None, v_smpl=None, pose_params=None, W=None, T=None):
        # lines = [[0.9560/10, 5.7827/10, 2.3717/10, 3.1060/10, 4.2848/10, 2.8963/10, 4.8322/10, 3.1068/10]]
        garmentvnums = [4248, 4258, 5327, 3721, 5404, 2818]
        garmentfnums = [8348, 8352, 10516, 7284, 10660, 5468]
        # up_name = osp.join('/mnt/data/Project/BCNet-master/body_garment_dataset/tmps/shirts/garment_tmp.obj')
        # mesh1 = trimesh.load_mesh(up_name, process=False)
        # up_name = osp.join('/mnt/data/Project/BCNet-master/body_garment_dataset/tmps/short_skirts/garment_tmp.obj')
        # mesh2 = trimesh.load_mesh(up_name, process=False)
        batch_vertices = vertices#torch.cat([torch.Tensor(mesh1.vertices),torch.Tensor(mesh2.vertices)],dim=0).to(self.cur_device)
        batch_faces = faces#torch.cat([torch.Tensor(mesh1.faces),torch.Tensor(mesh2.faces)],dim=0).to(self.cur_device)
        new_vertices = torch.Tensor().to(self.cur_device)
        index = gtypes.nonzero(as_tuple=False)[:, 1]
        # index = [0,5]
        for ind in range(gtypes.shape[0]):
            # self.A = A[ind]
            line = lines[ind]
            if line[-1] == 100:
                line = line[:-1]
            self.lines = line
            self.up_vertices = batch_vertices[:garmentvnums[index[0+ind*2]]]
            self.up_vertices_deform = self.up_vertices.clone()
            self.bottom_vertices = batch_vertices[garmentvnums[index[0+ind*2]]:garmentvnums[index[0+ind*2]]+garmentvnums[index[1+ind*2]]]
            self.bottom_vertices_deform = self.bottom_vertices.clone()
            self.faces = batch_faces
            self.up_faces = batch_faces[:garmentfnums[index[0+ind*2]]]
            self.bottom_faces = batch_faces[garmentfnums[index[0+ind*2]]:garmentfnums[index[0+ind*2]]+garmentfnums[index[1+ind*2]]]
            up_verts = []
            bottom_verts = []
            if index[0+ind*2] == 0:
                name = "shirts"
                self.setParams(name)
                # idx_anchor_in_mesh1, vert_anchor1 = self.set_sleeve_l(name)
                # idx_anchor_in_mesh2, vert_anchor2 = self.set_sleeve_r(name)
                idx_anchor_in_mesh3, vert_anchor3 = self.set_up_waist(name)
                idx_anchor_in_mesh6, vert_anchor6 = self.set_SleeveLength(name)
                idx_anchor_in_mesh4, vert_anchor4 = self.set_collar(name)
                idx_anchor_in_mesh5, vert_anchor5, CL, cl = self.set_ClothingLength(name)

                # idx_anchor_in_mesh7, vert_anchor7 = self.set_ChestWidth(name)
                vert_anchor = torch.cat([vert_anchor6, vert_anchor5, vert_anchor4], dim=0)
                idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh6, idx_anchor_in_mesh5, idx_anchor_in_mesh4], dim=0)
                # vert_anchor = self.repose(vert_anchor_, 0)
                idx_anchor_in_mesh = idx_anchor_in_mesh.int().cpu().detach().numpy()
                vert_anchor = vert_anchor.cpu().detach().numpy()
                up_faces = self.up_faces.cpu().detach().numpy()
                up_vertices = self.up_vertices.cpu().detach().numpy()
                # vert_anchor,up_vertices = self.set_Pose(vert_anchor, idx_anchor_in_mesh, v_smpl, pose_params, W, T, 0)
                up_verts = self.laplacian(up_vertices, up_faces, idx_anchor_in_mesh, vert_anchor)
                # for v in up_verts:
                #     v[1]-=0.03
            if index[0+ind*2] == 1:
                name = "short_shirts"
                self.setParams(name)
                self.lines[4] += 0.05
                # idx_anchor_in_mesh1, vert_anchor1 = self.set_sleeve_l(name)
                # idx_anchor_in_mesh2, vert_anchor2 = self.set_sleeve_r(name)
                idx_anchor_in_mesh3, vert_anchor3 = self.set_up_waist(name)
                idx_anchor_in_mesh6, vert_anchor6 = self.set_SleeveLength(name)
                idx_anchor_in_mesh4, vert_anchor4 = self.set_collar(name)
                idx_anchor_in_mesh5, vert_anchor5, CL, cl = self.set_ClothingLength(name)
                # idx_anchor_in_mesh7, vert_anchor7 = self.set_ChestWidth(name)
                # idx_anchor_in_mesh = np.nonzero(np.array(idx_anchor_in_mesh) != 0)
                # vert_anchor1 = torch.cat([vert_anchor1, vert_anchor2, vert_anchor3_1, vert_anchor3_2, vert_anchor4], dim=0)
                # idx_anchor_in_mesh1 = torch.cat([idx_anchor_in_mesh1, idx_anchor_in_mesh2, idx_anchor_in_mesh3_1, idx_anchor_in_mesh3_2, idx_anchor_in_mesh4], dim=0)
                vert_anchor = torch.cat([vert_anchor6, vert_anchor5, vert_anchor4], dim=0)
                idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh6, idx_anchor_in_mesh5, idx_anchor_in_mesh4], dim=0)

                idx_anchor_in_mesh = idx_anchor_in_mesh.int().cpu().detach().numpy()
                vert_anchor = vert_anchor.cpu().detach().numpy()
                up_faces = self.up_faces.cpu().detach().numpy()
                up_vertices = self.up_vertices.cpu().detach().numpy()
                up_verts = self.laplacian(up_vertices, up_faces, idx_anchor_in_mesh, vert_anchor)
                # for v in up_verts:
                #     v[1]-=0.05
                # up_verts = self.laplacian(up_verts_, up_faces, idx_anchor_in_mesh2, vert_anchor2)

            if len(line) == 9 and (index[1+ind*2]==2 or index[1+ind*2]==3):
                bottom_verts = self.bottom_vertices.cpu().detach().numpy()
            elif len(line) == 10 and (index[1+ind*2]==4 or index[1+ind*2]==5):
                bottom_verts = self.bottom_vertices.cpu().detach().numpy()
            else:
                if index[1 + ind * 2] == 2:
                    name = "pants"
                    idx_anchor_in_mesh1, vert_anchor1 = self.set_bottom_waist(name)
                    idx_anchor_in_mesh2, vert_anchor2 = self.set_bottom_trouser_l(name)
                    idx_anchor_in_mesh3, vert_anchor3 = self.set_bottom_trouser_r(name)
                    idx_anchor_in_mesh4, vert_anchor4 = self.set_TrousersLength(name)
                    # idx_anchor_in_mesh5, vert_anchor5 = self.set_CrotchLength(name)
                    idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh1,idx_anchor_in_mesh4],dim=0)
                    vert_anchor = torch.cat([vert_anchor1,vert_anchor4], dim=0)
                    idx_anchor_in_mesh = idx_anchor_in_mesh.int().cpu().detach().numpy()
                    vert_anchor = vert_anchor.cpu().detach().numpy()
                    bottom_faces = self.bottom_faces.cpu().detach().numpy()
                    bottom_vertices = self.bottom_vertices.cpu().detach().numpy()
                    bottom_verts = self.laplacian(bottom_vertices, bottom_faces, idx_anchor_in_mesh, vert_anchor)
                if index[1 + ind * 2] == 3:
                    name = "short_pants"
                    idx_anchor_in_mesh1, vert_anchor1 = self.set_bottom_waist(name)
                    idx_anchor_in_mesh2, vert_anchor2 = self.set_bottom_trouser_l(name)
                    idx_anchor_in_mesh3, vert_anchor3 = self.set_bottom_trouser_r(name)
                    idx_anchor_in_mesh4, vert_anchor4 = self.set_TrousersLength(name)
                    # idx_anchor_in_mesh5, vert_anchor5 = self.set_CrotchLength(name)
                    idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh1,idx_anchor_in_mesh4],dim=0)
                    vert_anchor = torch.cat([vert_anchor1,vert_anchor4], dim=0)
                    idx_anchor_in_mesh = idx_anchor_in_mesh.int().cpu().detach().numpy()
                    vert_anchor = vert_anchor.cpu().detach().numpy()
                    bottom_faces = self.bottom_faces.cpu().detach().numpy()
                    bottom_vertices = self.bottom_vertices.cpu().detach().numpy()
                    bottom_verts = self.laplacian(bottom_vertices, bottom_faces, idx_anchor_in_mesh, vert_anchor)
                if index[1 + ind * 2] == 4:
                    name = "skirts"
                    idx_anchor_in_mesh1, vert_anchor1 = self.set_bottom_waist(name)
                    idx_anchor_in_mesh2, vert_anchor2 = self.set_skirt(name)
                    idx_anchor_in_mesh3, vert_anchor3 = self.set_SkirtLength(name)
                    idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh1, idx_anchor_in_mesh3], dim=0)
                    vert_anchor = torch.cat([vert_anchor1, vert_anchor3], dim=0)
                    idx_anchor_in_mesh = idx_anchor_in_mesh.int().cpu().detach().numpy()
                    vert_anchor = vert_anchor.cpu().detach().numpy()
                    bottom_faces = self.bottom_faces.cpu().detach().numpy()
                    bottom_vertices = self.bottom_vertices.cpu().detach().numpy()
                    bottom_verts = self.laplacian(bottom_vertices, bottom_faces, idx_anchor_in_mesh, vert_anchor)
                if index[1 + ind * 2] == 5:
                    name = "short_skirts"
                    idx_anchor_in_mesh1, vert_anchor1 = self.set_bottom_waist(name)
                    idx_anchor_in_mesh2, vert_anchor2 = self.set_skirt(name)
                    idx_anchor_in_mesh3, vert_anchor3 = self.set_SkirtLength(name)
                    idx_anchor_in_mesh = torch.cat([idx_anchor_in_mesh1, idx_anchor_in_mesh3], dim=0)
                    vert_anchor = torch.cat([vert_anchor1, vert_anchor3], dim=0)
                    idx_anchor_in_mesh = idx_anchor_in_mesh.int().cpu().detach().numpy()
                    vert_anchor = vert_anchor.cpu().detach().numpy()
                    bottom_faces = self.bottom_faces.cpu().detach().numpy()
                    bottom_vertices = self.bottom_vertices.cpu().detach().numpy()
                    bottom_verts = self.laplacian(bottom_vertices, bottom_faces, idx_anchor_in_mesh, vert_anchor)
            batch_vertices = batch_vertices[garmentvnums[index[0+ind*2]]+garmentvnums[index[1+ind*2]]:]
            batch_faces = batch_faces[garmentfnums[index[0+ind*2]] + garmentfnums[index[1+ind*2]]:]
            new_vertice = torch.cat([torch.Tensor(up_verts).to(self.cur_device), torch.Tensor(bottom_verts).to(self.cur_device)],dim=0)
            new_vertices = torch.cat([new_vertices, new_vertice],dim=0)
        return new_vertices
