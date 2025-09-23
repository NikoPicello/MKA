import sys 
import os
import cv2
import math
import smplx
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import numba
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange
import plotly.graph_objects as go
from PIL import Image
import io
import pytorch3d.renderer
import pytorch3d.transforms
import torch
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.transforms import RotateAxisAngle, axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.renderer import (
    BlendParams,
    look_at_view_transform,
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex,
    HardGouraudShader,
    SoftPhongShader,
    PointLights,
)

from pytorch3d.renderer.mesh.rasterizer import Fragments
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from .multiview1 import transform_camera
from .k4d_camera import cam_ext_setup, extrinsics, color_focal, depth_focal, color_camera_matrix, extrinsics1
from typing import Optional, Tuple
import time

# 射线-三角形相交检测（修正法向量方向）
@njit(numba.types.Tuple((numba.boolean, numba.float32))(numba.float32[:], 
                                                      numba.float32[:], 
                                                      numba.float32[:], 
                                                      numba.float32[:], 
                                                      numba.float32[:]))
def ray_triangle_intersection_numba(ray_origin, ray_dir, v0, v1, v2):
    epsilon = np.float32(1e-6)
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    h = np.cross(ray_dir, edge2)
    a = np.dot(edge1, h)
    
    if -epsilon < a < epsilon:
        return False, np.float32(0.0)
    
    f = np.float32(1.0) / a
    s = ray_origin - v0
    u = f * np.dot(s, h)
    
    if u < 0.0 or u > 1.0:
        return False, np.float32(0.0)
    
    q = np.cross(s, edge1)
    v_val = f * np.dot(ray_dir, q)
    
    if v_val < 0.0 or (u + v_val) > 1.0:
        return False, np.float32(0.0)
    
    t = f * np.dot(edge2, q)
    return t > epsilon, t

@njit(numba.boolean[:](numba.float32, 
                      numba.float32, 
                      numba.float32[:,:], 
                      numba.float32[:], 
                      numba.float32[:,:,:]))
def process_single_image_point(u, v, intrinsic_inv, cam_pos, triangles):
    homogeneous = np.array([u, v, np.float32(1.0)], dtype=np.float32)
    ray_dir = intrinsic_inv @ homogeneous
    ray_dir /= np.linalg.norm(ray_dir)
    ray_dir = ray_dir.astype(np.float32)
    
    num_triangles = triangles.shape[0]
    vis_flags = np.zeros(num_triangles, dtype=numba.types.boolean)
    min_depth = np.float32(np.inf)
    closest_idx = -1
    
    for i in prange(num_triangles):
        v0 = triangles[i, 0].astype(np.float32)
        v1 = triangles[i, 1].astype(np.float32)
        v2 = triangles[i, 2].astype(np.float32)
        
        intersect, t = ray_triangle_intersection_numba(cam_pos, ray_dir, v0, v1, v2)
        if not intersect:
            continue
        
        # 计算面片法向量（注意顶点顺序）
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        
        # 正确的前面判断：法向量应朝向相机
        if np.dot(normal, ray_dir) >= 0:
            continue  # 剔除背面
            
        # 深度测试
        if t < min_depth:
            min_depth = t
            closest_idx = i
    
    if closest_idx != -1:
        vis_flags[closest_idx] = True
    
    return vis_flags

def check_visible_faces_parallel(vertices, faces, intrinsic_matrix, image_points):
    intrinsic_matrix = intrinsic_matrix.astype(np.float32)
    intrinsic_inv = np.linalg.inv(intrinsic_matrix).astype(np.float32)
    intrinsic_inv = np.ascontiguousarray(intrinsic_inv)
    
    vertices = np.ascontiguousarray(vertices.astype(np.float32))
    cam_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    num_faces = faces.shape[0]
    triangles = np.empty((num_faces, 3, 3), dtype=np.float32)
    for i in range(num_faces):
        triangles[i] = vertices[faces[i]]  # 保持原始顶点顺序
    triangles = np.ascontiguousarray(triangles)
    
    image_points = [(np.float32(u), np.float32(v)) for (u, v) in image_points]
    
    results = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for u, v in image_points:
            future = executor.submit(
                process_single_image_point,
                u, v, intrinsic_inv, cam_pos, triangles
            )
            futures.append(future)
        
        for future in futures:
            results.append(future.result())
    
    combined_visible = np.any(np.stack(results), axis=0)
    return combined_visible

def ray_triangle_intersection_batch(ray_origins, ray_dirs, triangles, epsilon=1e-6):
    """
    批量射线-三角形相交检测 (PyTorch向量化实现)
    
    参数：
        ray_origins : (B, 3) 射线起点
        ray_dirs    : (B, 3) 射线方向（需归一化）
        triangles   : (T, 3, 3) 三角形顶点坐标
        epsilon     : 数值稳定项
        
    返回：
        hit_mask    : (B, T) 是否相交的布尔掩码
        hit_depth   : (B, T) 相交深度值（未击中的为inf）
    """
    device = ray_origins.device
    B, T = ray_origins.size(0), triangles.size(0)
    
    # 扩展维度用于广播计算
    ray_origins = ray_origins.view(B, 1, 3)  # (B,1,3)
    ray_dirs = ray_dirs.view(B, 1, 3)        # (B,1,3)
    triangles = triangles.view(1, T, 3, 3)  # (1,T,3,3)
    
    # 提取三角形顶点
    v0, v1, v2 = triangles.unbind(dim=2)  # 各形状为(1,T,3)
    
    # 计算边向量
    edge1 = v1 - v0  # (1,T,3)
    edge2 = v2 - v0
    
    # 计算行列式
    h = torch.cross(ray_dirs.expand(-1, T, -1), edge2, dim=-1)  # (B,T,3)
    det = torch.sum(edge1 * h, dim=-1)  # (B,T)
    
    # 过滤平行情况
    hit_mask = (det.abs() > epsilon)
    
    # 计算u参数
    f = 1.0 / (det + epsilon)
    s = ray_origins - v0  # (B,T,3)
    u = f * torch.sum(s * h, dim=-1)  # (B,T)
    hit_mask &= (u >= 0) & (u <= 1)
    
    # 计算q向量和v参数
    q = torch.cross(s, edge1.expand(B, -1, -1), dim=-1)  # (B,T,3)
    v = f * torch.sum(ray_dirs.expand(-1, T, -1) * q, dim=-1)  # (B,T)
    hit_mask &= (v >= 0) & (u + v <= 1)
    
    # 计算t参数
    t = f * torch.sum(edge2.expand(B, -1, -1) * q, dim=-1)  # (B,T)
    hit_mask &= (t > epsilon)
    
    # 合并结果
    hit_depth = torch.where(hit_mask, t, torch.tensor(math.inf, device=device))
    
    return hit_mask, hit_depth

def compute_visibility(vertices, faces, intrinsic, image_points):
    """
    GPU加速的可见性检测主函数（修正版）
    
    修改点：
    1. 添加有效命中过滤，避免无效索引污染结果
    2. 修正深度测试逻辑，严格筛选可见三角形
    3. 优化内存使用，减少显存占用
    """
    device = vertices.device
    F = faces.size(0)
    P = image_points.size(0)
    
    # 生成射线方向（增加数值稳定性）
    uv_homo = torch.cat([
        image_points, 
        torch.ones(P, 1, device=device, dtype=image_points.dtype)
    ], dim=1)
    intrinsic_inv = torch.linalg.inv(intrinsic.float())
    ray_dirs = (uv_homo @ intrinsic_inv.T)  # (P,3)
    ray_dirs = ray_dirs / (ray_dirs.norm(dim=1, keepdim=True) + 1e-8)  # 安全归一化
    
    # 准备三角形数据
    triangles = vertices[faces]  # (F,3,3)
    
    # 批量相交检测
    hit_mask, hit_depth = ray_triangle_intersection_batch(
        torch.zeros(P, 3, device=device, dtype=vertices.dtype),
        ray_dirs,
        triangles
    )  # hit_mask: (P,F), hit_depth: (P,F)
    
    # 法线方向过滤背面（优化广播计算）
    v0, v1, v2 = triangles.unbind(dim=1)  # (F,3)
    edge1 = v1 - v0
    edge2 = v2 - v0
    normals = torch.cross(edge1, edge2, dim=1)  # (F,3)
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-8)
    front_facing = torch.einsum('pd,fd->pf', ray_dirs, normals) < 0  # (P,F)
    hit_mask = hit_mask & front_facing
    
    # 深度测试（严格处理无效命中）
    hit_depth[~hit_mask] = float('inf')
    
    # 找到有效命中点（至少有一个三角形被击中）
    valid_rays = hit_depth.min(dim=1).values < float('inf')  # (P,)
    if not valid_rays.any():
        return torch.zeros(F, dtype=torch.bool, device=device)
    
    # 获取有效命中的最近三角形索引
    closest_indices = torch.argmin(hit_depth[valid_rays], dim=1)  # (K,) K=valid_rays.sum()
    
    # 生成可见性掩码（使用bincount避免unique的排序开销）
    visible_counts = torch.bincount(closest_indices, minlength=F)  # (F,)
    visible_mask = visible_counts > 0
    
    return visible_mask

def check_visible_faces_torch(vertices, faces, intrinsic_matrix, image_points, epsilon=1e-6):
    """
    利用 GPU 并行计算（PyTorch向量化实现），根据一批2D图像采样点判断 MANO mesh 中哪些面片可见，
    并检查深度遮挡（即每个采样点仅选择最近的交点）。
    
    参数：
      - vertices: torch.Tensor, shape=(num_vertices, 3)，类型 float32
      - faces: torch.Tensor, shape=(num_faces, 3)，每行存储一个面片的三个顶点索引（整型）
      - intrinsic_matrix: torch.Tensor, shape=(3, 3)，相机内参矩阵
      - image_points: torch.Tensor, shape=(N, 2)，采样的二维像素点（u,v）
      - epsilon: 浮点容差（默认1e-6）
      
    返回：
      - combined_visible: torch.BoolTensor, shape=(num_faces,)，若任一图像点检测到该面片可见，则标记为 True
    """
    device = vertices.device  # 假定所有数据都在同一设备上
    N = image_points.shape[0]  # 采样点数量
    ############# calculate rays
    intrinsic_inv = torch.inverse(intrinsic_matrix)
    # 对所有图像采样点计算射线方向
    # 将 image_points 转为齐次坐标: (N,2) -> (N,3)
    ones = torch.ones((N, 1), dtype=torch.float32, device=device)
    image_points_h = torch.cat([image_points, ones], dim=1)  # (N,3)
    # 每个采样点的射线方向: d = intrinsic_inv @ pixel_h  (注意批量矩阵乘法)
    rays = torch.matmul(image_points_h, intrinsic_inv.T)  # (N,3)
    rays = rays / torch.norm(rays, dim=1, keepdim=True)   # (N,3)
    # 相机位置（单目相机默认为原点）
    cam_pos = torch.zeros((1,3), dtype=torch.float32, device=device)  # shape (1,3)
    
    # 构造 triangles: 根据 faces 从 vertices 索引，得到 shape=(F, 3, 3)
    triangles = vertices[faces]  # faces shape: (F,3), triangles: (F,3,3)
    F = triangles.shape[0]  # 面片数量
    # 预计算每个三角形的边以及法向量
    v0 = triangles[:, 0, :]  # (F,3)
    v1 = triangles[:, 1, :]  # (F,3)
    v2 = triangles[:, 2, :]  # (F,3)
    edge1 = v1 - v0          # (F,3)
    edge2 = v2 - v0          # (F,3)
    # 计算三角形的法向量
    normals = torch.cross(edge1, edge2, dim=1)  # (F,3)
    norms = torch.norm(normals, dim=1, keepdim=True)
    normals = normals / (norms + epsilon)  # (F,3)
    
    # 扩展射线和三角形数据方便广播计算：
    # 将 rays 扩展到 (N, F, 3)
    rays_exp = rays.unsqueeze(1).expand(N, F, 3)
    # 将三角形的 v0, edge1, edge2 扩展到 (N, F, 3)
    v0_exp = v0.unsqueeze(0).expand(N, F, 3)
    edge1_exp = edge1.unsqueeze(0).expand(N, F, 3)
    edge2_exp = edge2.unsqueeze(0).expand(N, F, 3)
    
    # 计算 Möller–Trumbore 算法各中间变量：
    # 计算行列式 (光线方向与edge2的叉积)
    h = torch.cross(rays_exp, edge2_exp, dim=2)  # (N,F,3)
    det = torch.sum(edge1_exp * h, dim=2)  # (N,F)
    # 如果行列式接近0，则光线与三角形平行
    valid_det = (det.abs() > epsilon)
    # 计算逆行列式（对于不满足条件的，用 inf 代替）
    inv_det = torch.where(valid_det, 1.0 / det, torch.full_like(det, float('inf')))
    
    # s = ray_origin - v0, 由于相机位置为原点，则 s = -v0
    s = -v0_exp  # (N,F,3)
    # 计算重心坐标u
    u_val = inv_det * torch.sum(s * h, dim=2)  # (N,F)
    valid_u = (u_val >= 0) & (u_val <= 1)
    
    # 计算重心坐标v
    q = torch.cross(s, edge1_exp, dim=2)  # (N,F,3)
    v_val = inv_det * torch.sum(rays_exp * q, dim=2)  # (N,F)
    valid_v = (v_val >= 0) & ((u_val + v_val) <= 1)
    
    # 计算交点距离t
    t_val = inv_det * torch.sum(edge2_exp * q, dim=2)  # (N,F)
    valid_t = t_val > epsilon
    
    # 交点存在条件
    intersect = valid_det & valid_u & valid_v & valid_t  # (N,F) bool tensor
    
    # 背面剔除：对于每个射线和三角形，计算 dot(normal, ray_dir)
    normals_exp = normals.unsqueeze(0).expand(N, F, 3)  # (N,F,3)
    dot_normal = torch.sum(rays_exp * normals_exp, dim=2)  # (N,F)
    front_facing = dot_normal < 0  # (N,F)
    
    valid_intersect = intersect & front_facing  # (N,F)
    
    # 对于未满足条件的，将 t 值设为无穷大，便于后续选取最小深度
    t_masked = torch.where(valid_intersect, t_val, torch.full_like(t_val, float('inf')))
    
    # 对每个射线，找到最近交点的面片索引
    # shape: (N,), 每个射线选出对应面片索引（若全部为 inf，则返回 0）
    min_t, min_idx = torch.min(t_masked, dim=1)
    # 对于每个射线，若 min_t 不为 inf，则说明射线与某面片有效交点
    ray_has_intersect = min_t < float('inf')
    
    # 创建一个 (N, F) 的 bool 矩阵，标记每个射线选出的面片
    selected = torch.zeros((N, F), dtype=torch.bool, device=device)
    selected[torch.arange(N).cuda()[ray_has_intersect], min_idx[ray_has_intersect]] = True
    
    # 最终将所有射线的结果做逻辑或，得到每个面片是否被至少一条射线检测到
    combined_visible = torch.any(selected, dim=0)  # (F,)
    return combined_visible

def check_key_joints_visibility_torch(visible_faces, joint_face_indices, threshold=0.0):
    joint_visibility = {}
    for joint, face_indices in joint_face_indices.items():
        if not face_indices:  # 处理空列表情况
            joint_visibility[joint] = False
            continue
            
        # 将面片索引转为 GPU 张量
        indices_tensor = torch.tensor(face_indices, 
                                    dtype=torch.long,
                                    device=visible_faces.device)
        
        # 计算可见面片比例
        visible_ratio = visible_faces[indices_tensor].float().mean()
        joint_visibility[joint] = visible_ratio #(visible_ratio > threshold).item()
    
    return joint_visibility

def get_per_joint_segmentation(faces, skinning_weights):
    """
    根据 MANO 模型提供的顶点皮肤权重，将面片（faces）分割到各个关键关节上。

    参数：
      - vertices: np.ndarray, 形状为 (num_vertices, 3)，存储 3D 顶点坐标
      - faces: np.ndarray 或 list，形状为 (num_faces, 3)，每行存储一个面片由三个顶点的索引构成
      - skinning_weights: np.ndarray, 形状为 (num_vertices, num_joints)，每行对应每个顶点在各关节上的权重

    返回：
      - joint_face_indices: dict，键为关节索引（0 ~ num_joints-1），值为一个列表，
                            存储该关节对应的面片索引。
    """
    num_joints = skinning_weights.shape[1]
    # 1. 对每个顶点取最大权重对应的关节索引
    vertex_joint_labels = np.argmax(skinning_weights, axis=1)
    
    # 2. 初始化每个关节对应的面片索引字典
    joint_face_indices = {j: [] for j in range(num_joints)}
    
    # 3. 遍历每个面片，根据面片内顶点的归属标签确定该面片所属关节
    for face_idx, face in enumerate(faces):
        # face 为三个顶点的索引
        face_labels = vertex_joint_labels[face]
        unique_labels, counts = np.unique(face_labels, return_counts=True)
        # 如果三个顶点中至少两个的标签一致，则取多数标签作为面片的归属关节
        majority_label = unique_labels[np.argmax(counts)]
        joint_face_indices[majority_label].append(face_idx)
    
    return joint_face_indices

from collections import defaultdict

def get_per_joint_segmentation_vectorized(faces, skinning_weights):

    vertex_joint_labels = np.argmax(skinning_weights, axis=1)
    joint_face_indices = defaultdict(list)
    
    # 向量化获取所有面片的唯一关节标签
    face_labels = vertex_joint_labels[faces]  # 形状 (num_faces, 3)
    unique_labels = [np.unique(labels) for labels in face_labels]
    
    # 批量添加索引
    for face_idx, joints in enumerate(unique_labels):
        for j in joints:
            joint_face_indices[j].append(face_idx)
    
    return dict(joint_face_indices)

def get_per_joint_segmentation_vectorized_torch(faces, skinning_weights):
    # 获取每个顶点的最大权重关节 [num_vertices]
    vertex_joint_labels = torch.argmax(skinning_weights, dim=1)
    
    # 提取面片对应的关节标签 [num_faces, 3]
    face_labels = vertex_joint_labels[faces]
    
    # 向量化计算每个关节对应的面片
    num_joints = skinning_weights.size(1)
    joint_face_indices = {}
    
    for j in range(num_joints):
        # 找到包含该关节的面片 (any(axis=1) 检查三个顶点是否至少有一个属于该关节)
        mask = (face_labels == j).any(dim=1)
        indices = torch.where(mask)[0].cpu().tolist()  # 结果转回 CPU 避免 GPU 内存碎片
        joint_face_indices[j] = indices
    
    return joint_face_indices

def transform_camera(
    vertices: torch.Tensor,
    bboxs: torch.Tensor,
    input_body_shape: torch.Tensor,
    tgt_princpt: Optional[torch.Tensor] = None,
    tgt_focal: Optional[torch.Tensor] = None,
):
    transformed_mesh = vertices.clone()
    centroid = transformed_mesh[..., 2].mean(-1)
    k = input_body_shape[1] / bboxs[:, 2]
    tz = centroid * (k - 1)
    transformed_mesh[..., 2] += tz[:, None]
    if tgt_princpt is not None and tgt_focal is not None:
        Z = transformed_mesh[..., 2:3]
        K = Z / tgt_focal[:, None, :]
        src_princpt = 0.5 * bboxs[..., 2:4] + bboxs[..., 0:2]
        d_princpt = K * (src_princpt - tgt_princpt)[:, None, :]
        transformed_mesh[..., :2] += d_princpt
    return transformed_mesh

class Render(torch.nn.Module):
    def __init__(self, cam_param, img_size, device="cuda") -> None:
        super().__init__()
        self.device = torch.device(device)
        self.rot = RotateAxisAngle(180, axis="Z", device=device)
        self.cameras = PerspectiveCameras(
            focal_length=cam_param["focal"],
            principal_point=cam_param["princpt"],
            in_ndc=False,
            image_size=(img_size,),
            device=device,
        )
        raster_settings = RasterizationSettings(
            image_size=img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
        )
        self.rasterizer = MeshRasterizer(
            cameras=self.cameras, raster_settings=raster_settings
        )
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
        self.shader = SoftPhongShader(
            device=device, cameras=self.cameras, lights=lights
        )

    def forward_fragments(
        self, vertices: torch.Tensor, faces: torch.Tensor
    ) -> Fragments:
        vertices = self.rot.transform_points(vertices)
        texture = TexturesVertex(verts_features=torch.ones_like(vertices))
        mesh = Meshes(verts=vertices, faces=faces, textures=texture).to(
            device=vertices.device
        )
        return self.rasterizer(mesh), mesh

    def forward_depth(
        self, vertices: torch.Tensor, faces: torch.Tensor
    ) -> torch.Tensor:
        return self.forward_fragments(vertices, faces)[0].zbuf[..., 0].clip(0, None)

    def forward(self, vertices: torch.Tensor, faces: torch.Tensor):
        fragments, mesh = self.forward_fragments(vertices, faces)
        rendered = self.shader(fragments, mesh)
        return rendered, fragments

    def forward_img(
        self,
        img: np.ndarray,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        return_depth=False,
    ):
        rendered, fragments = self.forward(vertices, faces)
        depth = fragments.zbuf[..., 0]
        rgb = rendered[..., :3]
        valid_mask = (depth > 0)[..., None].cpu().numpy()
        img = rgb.mul(255).to(dtype=torch.uint8).cpu().numpy() * valid_mask + img * (
            1 - valid_mask
        )
        if return_depth:
            return img, depth
        else:
            return img

parents = [
    -1,  # 0: root
    0,   # 1: left_hip
    0,   # 2: right_hip
    0,   # 3: spine1
    1,   # 4: left_knee
    2,   # 5: right_knee
    3,   # 6: spine2
    4,   # 7: left_ankle
    5,   # 8: right_ankle
    6,   # 9: spine3
    7,   # 10: left_foot
    8,   # 11: right_foot
    9,   # 12: neck
    12,  # 13: left_clavicle
    12,  # 14: right_clavicle
    12,  # 15: head
    13,  # 16: left_upper_arm
    14,  # 17: right_upper_arm
    16,  # 18: left_forearm (前臂)
    17,  # 19: right_forearm
    18,  # 20: left_wrist
    19   # 21: right_wrist
]

def convert_mano_to_target_camera(mano_rot_source, R_target_to_source):
    """将MANO数据从源相机坐标系转换到目标相机坐标系"""
    # 转换旋转
    # if isinstance(mano_rot_source, np.ndarray):
    #     mano_rot_source = torch.tensor(mano_rot_source)
    R_mano_source = R.from_rotvec(mano_rot_source).as_matrix() if mano_rot_source.shape == (3,) else \
                   R.from_quat(mano_rot_source).as_matrix() if mano_rot_source.shape == (4,) else \
                   mano_rot_source
    
    # 关键修正：正确的旋转矩阵乘法顺序
    R_mano_target = mano_rot_source @ R_target_to_source.T
    
    # 转换位置
    # if isinstance(mano_pos_source, torch.Tensor):
    #     mano_pos_source = mano_pos_source.cpu().numpy()
    # P_mano_target = (R_source_to_target @ (mano_pos_source - T_source_to_target.reshape(3,1))).flatten()
    
    return R_mano_target

def replace_mano_to_smplx(smplx_root_pose, smplx_body_pose, mano_wrist_pose, is_left):
    """
    将 MANO 的全局手腕旋转替换到 SMPLX 的局部手腕旋转
    :param smplx_root_pose: SMPLX 根节点旋转 (1x3, 轴角)
    :param smplx_body_pose: SMPLX 身体姿势 (21x3, 轴角)
    :param mano_wrist_pose: MANO 手腕全局旋转 (1x3, 轴角)
    :param is_left: 是否替换左手腕（默认True，False表示右手腕）
    :return: 更新后的 SMPLX body_pose
    """
    # 合并 root_pose 和 body_pose 为完整姿势参数 (22x3)
    smplx_body_pose = smplx_body_pose.detach().cpu().numpy()
    smplx_root_pose = smplx_root_pose.detach().cpu().numpy()
    mano_wrist_pose = mano_wrist_pose.detach().cpu().numpy()
    full_pose = np.vstack([smplx_root_pose, smplx_body_pose])  # shape (22, 3)

    # 计算每个关节的全局旋转矩阵
    global_rotations = []
    for i in range(len(full_pose)):
        if i == 0:
            # 根节点的全局旋转即其自身旋转
            rot = R.from_rotvec(full_pose[i]).as_matrix()
            global_rotations.append(rot)
        else:
            # 其他关节：全局旋转 = 父节点全局旋转 × 自身局部旋转
            parent_idx = parents[i]
            parent_rot = global_rotations[parent_idx]
            local_rot = R.from_rotvec(full_pose[i]).as_matrix()
            global_rot = parent_rot @ local_rot
            global_rotations.append(global_rot)

    # 确定目标手腕和前臂的索引
    wrist_idx = 20 if is_left else 21
    forearm_idx = parents[wrist_idx]  # 前臂索引（左手腕的父节点是左前臂18）

    # 获取前臂的全局旋转矩阵
    R_forearm_global = global_rotations[forearm_idx]

    # 将 MANO 手腕的轴角转换为全局旋转矩阵
    R_mano_wrist_global = R.from_rotvec(mano_wrist_pose.flatten()).as_matrix()

    # 计算 SMPLX 手腕的局部旋转矩阵（相对于前臂）
    R_smplx_wrist_local = R_forearm_global.T @ R_mano_wrist_global

    # 转换为轴角并替换到 body_pose 中
    smplx_wrist_axis_angle = R.from_matrix(R_smplx_wrist_local).as_rotvec()
    wrist_body_idx = wrist_idx - 1  # body_pose 中手腕的索引（因 full_pose[0] 是 root）
    # smplx_body_pose[wrist_body_idx] = smplx_wrist_axis_angle

    return torch.tensor(smplx_wrist_axis_angle).cuda()

# def replace_mano_to_smplx(smplx_root_pose, smplx_body_pose, mano_wrist_pose, is_right):
#     """
#     将 MANO 的全局手腕旋转替换到 SMPLX 的局部手腕旋转（使用 PyTorch3D 封装函数）
    
#     参数:
#         smplx_root_pose (torch.Tensor): SMPLX 根节点旋转，形状 (1, 3)，采用轴角表示
#         smplx_body_pose (torch.Tensor): SMPLX 身体姿势，形状 (21, 3)，采用轴角表示
#         mano_wrist_pose (torch.Tensor): MANO 手腕全局旋转，形状 (1, 3)，采用轴角表示
#         is_left (bool): 是否替换左手腕（默认 True，False 表示右手腕）
    
#     返回:
#         torch.Tensor: 替换手腕旋转后的 SMPLX body_pose（形状 (21, 3)）
#     """
#     # 合并 root_pose 和 body_pose 为完整姿势参数 (22, 3)
#     full_pose = torch.cat([smplx_root_pose, smplx_body_pose], dim=0)  # shape: (22, 3)

#     # 计算每个关节的全局旋转矩阵（使用 PyTorch3D 的 axis_angle_to_matrix）
#     global_rotations = []
#     for i in range(full_pose.shape[0]):
#         if i == 0:
#             # 根节点的全局旋转即其自身旋转
#             rot = axis_angle_to_matrix(full_pose[i].unsqueeze(0))[0]  # 形状 (3, 3)
#             global_rotations.append(rot)
#         else:
#             # 其他关节：全局旋转 = 父节点全局旋转 × 自身局部旋转
#             parent_idx = parents[i]  # 注意：parents 为全局变量
#             parent_rot = global_rotations[parent_idx]
#             local_rot = axis_angle_to_matrix(full_pose[i].unsqueeze(0))[0]
#             global_rot = parent_rot @ local_rot
#             global_rotations.append(global_rot)

#     # 确定目标手腕和前臂的索引
#     wrist_idx = 21 if is_right else 20
#     forearm_idx = parents[wrist_idx]  # 前臂索引

#     # 获取前臂的全局旋转矩阵
#     R_forearm_global = global_rotations[forearm_idx]

#     # 将 MANO 手腕的轴角转换为全局旋转矩阵
#     R_mano_wrist_global = axis_angle_to_matrix(mano_wrist_pose.view(1, 3))[0]

#     # 计算 SMPLX 手腕的局部旋转矩阵（相对于前臂坐标系）
#     # 公式: R_local = (R_forearm_global)^T × R_mano_wrist_global
#     R_smplx_wrist_local = R_forearm_global.transpose(0, 1) @ R_mano_wrist_global

#     # 将局部旋转矩阵转换为轴角
#     smplx_wrist_axis_angle = matrix_to_axis_angle(R_smplx_wrist_local.unsqueeze(0))[0]

#     return smplx_wrist_axis_angle

def plotly_visualize_mano(
    vertices: np.ndarray,   # (778, 3) MANO顶点
    faces: np.ndarray,      # (1538, 3) MANO面片
    visible_faces: list,    # 可见面片索引列表
    view: int,
    output_path="mano_plotly.png",  # 输出路径
    highlight_color='rgb(255,0,0)',  # 高亮颜色
    base_color='rgb(200,200,200)',   # 基础颜色    resolution=(1600, 1200),         # 图片分辨率
    resolution=(1280,720),
    camera_angles=None,               # 自定义相机角度
):
    """
    MANO可见面片高亮可视化（Plotly方案）
    参数：
        vertices: 顶点坐标数组
        faces: 面片索引数组
        visible_faces: 需要高亮的可见面片索引
        output_path: 输出图片路径
        highlight_color: 高亮颜色（Plotly颜色格式）
        base_color: 基础颜色（Plotly颜色格式）
        resolution: 输出图片分辨率 (宽, 高)
        camera_angles: 自定义相机角度字典 (可选)
    """
    # 转换面片颜色数据
    face_colors = [base_color] * len(faces)
    for idx in visible_faces:
        if idx < len(faces):
            face_colors[idx] = highlight_color

    # 创建3D网格对象
    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        facecolor=face_colors,
        flatshading=True,    # 平面着色显示清晰面片
        lighting=dict(
            ambient=0.3,     # 环境光强度
            diffuse=0.8,     # 漫反射强度
            specular=0.04    # 高光强度
        ),
        lightposition=dict(
            x=1000,          # 光源位置
            y=1000,
            z=1000
        )
    )

    # 设置默认相机角度
    if not camera_angles:
        camera_angles = dict(
            eye=dict(x=1.5, y=-1.5, z=1.0),  # 相机位置
            up=dict(x=1, y=-1, z=2),          # 上方向
            center=dict(x=0, y=0, z=0)       # 视图中心
        )

    # 创建图表布局
    fig = go.Figure(data=[mesh])
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=camera_angles,
            aspectmode='data'  # 保持模型比例
        ),
        paper_bgcolor='rgb(255,255,255)',  # 白色背景
        margin=dict(l=0, r=0, t=0, b=0),   # 无边距
        width=resolution[0],
        height=resolution[1]
    )
    # fig.show()
    # 保存为图片
    fig.write_html(f"./3d_mesh_{view}.html", auto_open=False)
    # img_bytes = fig.to_image(format="png", engine="kaleido")
    # Image.open(io.BytesIO(img_bytes)).save(output_path)
    # print(f"可视化结果已保存至：{output_path}")

def load_hamer(smplx_params, params, is_right, src, device):
    '''
    smplx_params: dict of smplx parameters with key 'root_pose', 'body_pose'
    params: dict of hamer parameters with key 'view', 'root_pose', 'hand_pose', 'view' is the list of views(single view: [0], multview: [0, 1, 2, ...])
    is_right: 0 for left hand, 1 for right hand
    src: the source view to load hand pose, e.g. 0 for single view, 0-Num_views-1 for multi-view
    device: torch device to load the parameters
    return: wrist and hand pose in smplx format, if view not found, return None
    '''
    if params is None:
        return None, None
    idx = torch.where(params['view']==src)
    # idx = np.where(params['right']==is_right)
    if idx[0].shape[0]==0:
        return None, None
    idx = idx[0]
    wrist = params['root_pose'][:, idx*3:(idx+1)*3]
    hand = params['hand_pose'][:, idx*45:(idx+1)*45]
    wrist = wrist.to(device)[0]
    hand = hand.to(device)[0]

    if is_right == 0:
        wrist[1::3] *= -1
        wrist[2::3] *= -1
        hand[1::3] *= -1
        hand[2::3] *= -1

    wrist = replace_mano_to_smplx(smplx_params['root_pose'], smplx_params['body_pose'].reshape(-1,3), wrist, not is_right)
    hand = hand[None, ...].float()
    return wrist, hand

def smplx_forward(smplx_params, params, view, smplx_model, device):
    lwrist, lhand = load_hamer(smplx_params, params['left'], 0, view, device)
    rwrist, rhand = load_hamer(smplx_params, params['right'], 1, view, device)

    for k in smplx_params:
        smplx_params[k] = smplx_params[k].to(device)
    
    if lwrist is not None:
        smplx_params['body_pose'][:, 19 * 3 : (19 + 1) * 3] = lwrist
        smplx_params['left_hand_pose'] = lhand

    if rwrist is not None:
        smplx_params['body_pose'][:, 20 * 3 : (20 + 1) * 3] = rwrist
        smplx_params['right_hand_pose'] = rhand
    
    # create smplx class for inference
    out = smplx_model(
            betas=smplx_params['betas'],
            global_orient=smplx_params['root_pose'],
            body_pose=smplx_params['body_pose'],
            left_hand_pose=lhand,
            right_hand_pose=rhand,
            return_verts=True
        )
    # import pdb; pdb.set_trace()
    # smplx inference 
    # coordinate convertion from bbox to full image
    verts = out.vertices + smplx_params['cam_trans'][:,None,:]
    joints = out.joints + smplx_params['cam_trans'][:,None,:]
    res = {
        'vertices': verts,
        'faces': smplx_model.faces,
        'regressor': smplx_model.J_regressor,
        'joints': joints,
        'smplx_param': smplx_params,
        'has_left': lwrist is not None,
        'has_right': rwrist is not None
    }
    return res

    # if mano_params[base_view] is not None:

def cam2pixel_torch(cam_coord, f, c):
    x = cam_coord[..., 0] / cam_coord[..., 2] * f[0] + c[0]
    y = cam_coord[..., 1] / cam_coord[..., 2] * f[1] + c[1]
    z = cam_coord[..., 2]
    return torch.stack((x, y, z), 2)

def cal_surface_area(faces, visible_faces, verts, vert_ids):
    visible_faces_subset = faces[visible_faces]
    kept_set = set(vert_ids.tolist())
    mask = torch.tensor([
        any(v.item() in kept_set for v in face)
        for face in visible_faces_subset
    ], dtype=torch.bool)
    visible_faces_subset = visible_faces_subset[mask]
    v0 = verts[visible_faces_subset[:, 0]]
    v1 = verts[visible_faces_subset[:, 1]] 
    v2 = verts[visible_faces_subset[:, 2]] 
    
    a = v1 - v0 
    b = v2 - v0  
    cross = torch.cross(a, b, dim=1)
    areas = 0.5 * torch.norm(cross, p=2, dim=1)
    
    return torch.sum(areas)

def make_4x4(mat):
    assert mat.shape[-2:] == torch.Size((3, 4))
    add_row = torch.tensor([0, 0, 0, 1], dtype=mat.dtype, device=mat.device)
    add_row = add_row.expand(*mat.shape[:-2], 1, 4)
    return torch.cat([mat, add_row], dim=-2)

def make_novel_view_cam_trans_root_pose(
    root_pose: torch.Tensor,
    cam_trans: torch.Tensor,
    RTs: torch.Tensor,
    x_0: torch.Tensor,
    current_view_idx: torch.Tensor
):
    """
    Generate new smplx cam_trans and global orientation (root_pose) according to known camera extrinsics.
    Parameters:
        root_pose (torch.Tensor): Single view global orientation in axis-angle format (B, 1, 3)
        cam_trans (torch.Tensor): Single view camera translation in axis-angle format (B, 1, 3)
        RTs (torch.Tensor): Cameras' extrinsics (Rotation+translation matrix) (B, V, 3, 4) or (B, V, 4, 4)
        x_0 (torch.Tensor): Single view SMPLX model 3d root joint coord (B, 1, 3)
        current_view_idx (torch.Tensor): Indexing tensor showing which view the root_pose and cam_trans belongs to.
    Returns:
        Tuple[torch.Tensor,torch.Tensor]: novel view global orientation and camera translation params that can be directly used as SMPLX params.
    """
    B = root_pose.shape[0]
    B_idx = torch.arange(B)
    if RTs.shape[-2:] == torch.Size((3, 4)):
        RTs = make_4x4(RTs)
    root_pose_mat = axis_angle_to_matrix(root_pose)
    w2c_mat = torch.cat([root_pose_mat, (cam_trans+x_0)[..., None]], -1)
    w2c_mat = make_4x4(w2c_mat).squeeze(1)
    c2w_mat = RTs[B_idx, current_view_idx].inverse().float() # transform current view to world coord
    new_w2c_mat = torch.einsum('bvij,bjk,bkl->bvil', RTs, c2w_mat, w2c_mat)
    # new_w2c_mat = RTs @ c2w @ w2c_mat
    new_root_pose = matrix_to_axis_angle(new_w2c_mat[..., :3, :3])
    # new_cam_trans = new_w2c_mat[..., :3, 3] + (x_0 - cam_trans[..., None, :])
    new_cam_trans = new_w2c_mat[..., :3, 3] - x_0
    return new_root_pose, new_cam_trans

def rigid_transform_3D_batched(A, B):
    """
    Perform rigid 3D transformation for batched inputs.
    Inputs:
        A: numpy array of shape [B, N, D], source points
        B: numpy array of shape [B, N, D], target points
    Outputs:
        c: numpy array of shape [B], scale factors for each batch
        R: numpy array of shape [B, D, D], rotation matrices for each batch
        t: numpy array of shape [B, D], translation vectors for each batch
    """
    batch_size, n, dim = A.shape
    centroid_A = np.mean(A, 1, keepdims=True)  # Shape: [B, 1, D]
    centroid_B = np.mean(B, 1, keepdims=True)  # Shape: [B, 1, D]
    
    H = np.matmul((A - centroid_A).transpose(0, 2, 1), B - centroid_B) / n  # Shape: [B, D, D]
    U, s, V = np.linalg.svd(H)  # U: [B, D, D], s: [B, D], V: [B, D, D]
    R = np.matmul(V.transpose(0, 2, 1), U.transpose(0, 2, 1))  # Shape: [B, D, D]
    
    # Fix improper rotations
    det_R = np.linalg.det(R)  # Shape: [B]
    mask = det_R < 0
    s[mask, -1] *= -1
    V[mask, :, -1] *= -1
    R[mask] = np.matmul(V[mask].transpose(0, 2, 1), U[mask].transpose(0, 2, 1))
    
    varP = np.sum(np.var(A, axis=1), axis=1)  # Shape: [B]
    c = np.sum(s, axis=1) / varP  # Shape: [B]
    
    t = -np.matmul(c[:, None, None] * R, centroid_A.transpose(0, 2, 1)).squeeze(-1) + centroid_B.squeeze(1)  # Shape: [B, D]
    
    return c, R, t

def rigid_align_batched(A, B):
    """
    Align A to B using batched rigid transformation.
    Inputs:
        A: numpy array of shape [B, N, D], source points
        B: numpy array of shape [B, N, D], target points
    Output:
        A_aligned: numpy array of shape [B, N, D], transformed source points
    """
    c, R, t = rigid_transform_3D_batched(A, B)
    A_aligned = np.matmul(c[:, None, None] * R, A.transpose(0, 2, 1)).transpose(0, 2, 1) + t[:, None, :]  # Shape: [B, N, D]
    return A_aligned

def cal_joint_vis(smplx_params, mano_params, base_view, device, smplx_model, render, princpt, verbose=False):
    #     idx = np.where(mano_params[base_view]['right']==is_right)
    #     if idx[0].shape[0]!=0:
    #         return base_view
    # return 2 if is_right else 0

    best_view, max_jv = base_view, 0
    jvs = []
    rmax, lmax = 0, 0
    lview, rview = 3, 3
    R_i2j, T_i2j = cam_ext_setup(4, extrinsics)
    # focals = torch.tensor([
    #     [1145.0494384765625, 1143.7811279296875],
    #     [1149.6756591796875, 1147.5916748046875],
    #     [1149.1407470703125, 1148.7989501953125],
    #     [1145.5113525390625, 1144.77392578125]
    # ]).cuda()
    # princpts = torch.tensor([
    #     [512.54150390625, 515.4514770507812],
    #     [508.8486328125, 508.0649108886719],
    #     [519.8158569335938, 501.40264892578125],
    #     [514.9682006835938, 501.88201904296875]
    # ]).cuda()
    
    for view in range(4):
        # smplx_param = smplx_params[view]
        res = smplx_forward(smplx_params, mano_params, view, smplx_model, device)

        # if is_right and not res['has_right']:
        #     continue
        # if not is_right and not res['has_left']:
        #     continue
        
        K = torch.tensor([[609.281982421875, 0, 640],[0, 609.3950805664062, 360],[0, 0, 1]], device=device)
        # K = torch.tensor([[focals[view][0], 0, princpts[view][0]],[0, focals[view][1], princpts[view][1]],[0, 0, 1]], device=device)
        
        faces = (
            torch.from_numpy(res['faces'].astype(np.int64))
            .expand(4, -1, -1)
            .to(device)
        )
        verts = transform_camera(res['vertices'], smplx_params['bboxs'], (256, 192), princpt, depth_focal)
        # verts = transform_camera(res['vertices'], smplx_param['bboxs'], (256, 192), princpts[view][None, ...], focals[view][None, ...])
        verts = (torch.einsum("ni,bij->bnj", verts[0], R_i2j[base_view].permute(0, 2, 1))+ T_i2j[base_view, :, None])
        rot_verts = render.rot.transform_points(verts)
        verts_2d = render.cameras.transform_points_screen(rot_verts)
        # verts_2d = cam2pixel_torch(verts, focals[view], princpts[view])
        smplx_mano_vert_ids = pickle.load(open('/mnt/slurm_home/jyzhang/hamer/_DATA/MANO_SMPLX_vertex_ids.pkl','rb'))
        
        hand_mesh2d = verts_2d[view][np.concatenate([smplx_mano_vert_ids['right_hand'], smplx_mano_vert_ids['left_hand']])]
        hand_mesh2d = hand_mesh2d[:, :2]

        # rhand2d = verts_2d[view][smplx_mano_vert_ids['right_hand']][:, :2]
        # lhand2d = verts_2d[view][smplx_mano_vert_ids['left_hand']][:, :2]
        # rvisible = check_visible_faces_torch(verts[view], faces[view], K, rhand2d)
        # lvisible = check_visible_faces_torch(verts[view], faces[view], K, lhand2d)
        # rarea = cal_surface_area(faces[view], rvisible, verts[view], smplx_mano_vert_ids['right_hand'])
        # larea = cal_surface_area(faces[view], lvisible, verts[view], smplx_mano_vert_ids['left_hand'])
        # if rarea>rmax and view in mano_params['right']['view']:
        #     rmax = rarea
        #     rview = view
        # if larea>lmax and view in mano_params['left']['view']:
        #     lmax = larea
        #     lview = view

        visible1 = check_visible_faces_torch(verts[view], faces[view], K, hand_mesh2d)
        # print(np.where(visible1!=visible.detach().cpu().numpy()))
        if verbose:
            plotly_visualize_mano(verts[view].detach().cpu().numpy(),faces[0].detach().cpu().numpy(),visible_faces=torch.where(visible1==True)[0],view=view,output_path=f"mano_visualization{view}.png")
        J_regressor = res['regressor']
        start = time.time()
        joint_face_indices1 = get_per_joint_segmentation_vectorized_torch(faces[0], J_regressor.T)
        # continue
        jv = check_key_joints_visibility_torch(visible1, joint_face_indices1)
        hand_jv = np.array([jv[k].detach().cpu().numpy() for k in sorted(range(25, 55))])
        if not res['has_right']:
            hand_jv[15:] = 0
        if not res['has_left']:
            hand_jv[:15] = 0
        # import pdb; pdb.set_trace()
        jvs.append(hand_jv)

    return np.array(jvs)

def select_best_view(smplx_params, mano_params, device, smplx_model, verbose=False):
    rmax, lmax = 0, 0
    lview, rview = 3, 3
    R_i2j, T_i2j = cam_ext_setup(4, extrinsics1)
    focals = torch.tensor([
        [1145.0494384765625, 1143.7811279296875],
        [1149.6756591796875, 1147.5916748046875],
        [1149.1407470703125, 1148.7989501953125],
        [1145.5113525390625, 1144.77392578125]
    ]).cuda()
    princpts = torch.tensor([
        [512.54150390625, 515.4514770507812],
        [508.8486328125, 508.0649108886719],
        [519.8158569335938, 501.40264892578125],
        [514.9682006835938, 501.88201904296875]
    ]).cuda()
    
    smplx_mano_vert_ids = pickle.load(open('/mnt/petrelfs/luohuiwen/IMU/MV_MoCap/h3wb/human_models/smplx/MANO_SMPLX_vertex_ids.pkl','rb'))
        

    for view in range(4):
        smplx_param = smplx_params[view]
        res = smplx_forward(smplx_param, mano_params, view, smplx_model, device)
        K = torch.tensor([[focals[view][0], 0, princpts[view][0]],[0, focals[view][1], princpts[view][1]],[0, 0, 1]], device=device)
        faces = (
            torch.from_numpy(res['faces'].astype(np.int64))
            .expand(4, -1, -1)
            .to(device)
        )
        verts = transform_camera(res['vertices'], smplx_param['bboxs'], (256, 192), princpts[view][None, ...], focals[view][None, ...])
        verts = (torch.einsum("ni,bij->bnj", verts[0], R_i2j[view].permute(0, 2, 1))+ T_i2j[view, :, None])
        verts_2d = cam2pixel_torch(verts, focals[view], princpts[view])
        rhand2d = verts_2d[view][smplx_mano_vert_ids['right_hand']][:, :2]
        lhand2d = verts_2d[view][smplx_mano_vert_ids['left_hand']][:, :2]
        rvisible = check_visible_faces_torch(verts[view], faces[view], K, rhand2d)
        lvisible = check_visible_faces_torch(verts[view], faces[view], K, lhand2d)
        rarea = cal_surface_area(faces[view], rvisible, verts[view], smplx_mano_vert_ids['right_hand'])
        larea = cal_surface_area(faces[view], lvisible, verts[view], smplx_mano_vert_ids['left_hand'])
        if rarea>rmax and view in mano_params['right']['view']:
            rmax = rarea
            rview = view
        if larea>lmax and view in mano_params['left']['view']:
            lmax = larea
            lview = view

        if verbose:
            lvisible = check_visible_faces_torch(verts[view], faces[view], K, lhand2d)
            rvisible = check_visible_faces_torch(verts[view], faces[view], K, rhand2d)
            plotly_visualize_mano(verts[view].detach().cpu().numpy(),faces[0].detach().cpu().numpy(),visible_faces=torch.where(lvisible==True)[0],view=view,output_path=f"h3wb_vis_l_{view}.png")
            plotly_visualize_mano(verts[view].detach().cpu().numpy(),faces[0].detach().cpu().numpy(),visible_faces=torch.where(rvisible==True)[0],view=view,output_path=f"h3wb_vis_r_{view}.png")

    return lview, rview

