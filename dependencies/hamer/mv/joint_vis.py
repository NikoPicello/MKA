import sys 
import os
import cv2
import math
import smplx
import numpy as np
from PIL import Image
import io
import torch
import torch.nn as nn 
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
from typing import Optional, Tuple
import time

from ..utils.geometry import aa_to_rotmat, matrix_to_axis_angle, compute_twist_rotation

def get_rasterizer(img_h, img_w):
    cameras = look_at_view_transform(2.7, 10, 20)
    blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))
    raster_settings = RasterizationSettings(
        image_size=(img_h, img_w), 
        blur_radius=0.0, 
        faces_per_pixel=1, 
        bin_size = None,  # this setting controls whether naive or coarse-to-fine rasterization is used
        max_faces_per_bin = None  # this setting is for coarse rasterization
    )
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )
    return rasterizer

def render_mesh_pt3d(img, verts, faces, cam_param, rasterizer):
    device = verts.device
    img_h, img_w = img.shape[:2]
    image_size = torch.tensor([img_h, img_w]).unsqueeze(0).to(device)
    image_size_wh = image_size.flip(dims=(1, ))
    scale = image_size_wh.min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    focal, princpt = cam_param['focal'], cam_param['princpt']
    focal_length = torch.tensor([focal[0], focal[1]]).float().unsqueeze(0).to(device)
    principal_point = torch.tensor([princpt[0], princpt[1]]).float().unsqueeze(0).to(device)
    focal_pt = focal_length / scale
    p0_pt = -(principal_point - c0) / scale

    camera_pose = torch.eye(4).unsqueeze(0).to(device)
    R_pt = camera_pose[:, :3, :3].clone().permute(0, 2, 1)
    R_pt[:, :, :2] *= -1
    tvec_pt = camera_pose[:, :3, 3].clone()
    tvec_pt[:, :2] *= -1

    cameras = PerspectiveCameras(R=R_pt, T=tvec_pt, focal_length=focal_pt, principal_point=p0_pt, image_size=image_size, device=device)
    blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))
    lights = PointLights(device=device, location=((0.0, 2.0, -2.0),), specular_color=((0.0, 0.0, 0.0),))
    gray_renderer = MeshRenderer(
        rasterizer=rasterizer,        
        shader = HardGouraudShader(device=device, lights=lights, cameras=cameras, blend_params=blend_params)
    )

    verts_rgb = torch.ones_like(verts)
    tex = TexturesVertex(verts_features=verts_rgb).to(device)
    meshes = Meshes(verts=verts, faces=faces, textures=tex).to(device)
    rendered_imgs = gray_renderer(meshes_world=meshes, cameras=cameras)

    # rendered_imgs = rendered_imgs[:, :img_h, :img_w].clone() ### crop 
    render_img = rendered_imgs[0, ..., :3].detach().cpu().numpy() * 255.0
    render_mask = rendered_imgs[0, ..., 3:].detach().cpu().numpy() 
    output_image = img * (1.0 - render_mask) + render_img * render_mask
    output_image = output_image.astype(np.uint8)
    return output_image

def check_visibility_pt3d(rasterizer, img, verts, faces, cam_param):
    device = verts.device
    img_h, img_w = img.shape[:2]
    image_size = torch.tensor([img_h, img_w]).unsqueeze(0).to(device)
    image_size_wh = image_size.flip(dims=(1, ))
    scale = image_size_wh.min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    focal, princpt = cam_param['focal'], cam_param['princpt']
    focal_length = torch.tensor([focal[0], focal[1]]).float().unsqueeze(0).to(device)
    principal_point = torch.tensor([princpt[0], princpt[1]]).float().unsqueeze(0).to(device)
    focal_pt = focal_length / scale
    p0_pt = -(principal_point - c0) / scale

    camera_pose = torch.eye(4).unsqueeze(0).to(device)
    R_pt = camera_pose[:, :3, :3].clone().permute(0, 2, 1)
    R_pt[:, :, :2] *= -1
    tvec_pt = camera_pose[:, :3, 3].clone()
    tvec_pt[:, :2] *= -1
    cameras = PerspectiveCameras(R=R_pt, T=tvec_pt, focal_length=focal_pt, principal_point=p0_pt, image_size=image_size, device=device)
        
    mesh = Meshes(verts=verts, faces=faces).to(device)
    # with torch.no_grad():
    #     fragments = rasterizer(mesh, cameras=cameras)
    #     print("=== fragments", fragments.zbuf.size(), flush=True)
    #     depth_map = fragments.zbuf[0, ..., 0].cpu().numpy() 
    # cv2.imwrite("./check.jpg", depth_map*255)
    # exit(0)
    
    vertices = mesh.verts_packed().cpu().detach().numpy()
    homogeneous_vertices = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    cam_transform = cameras.get_world_to_view_transform()
    projected_vertices = cam_transform.transform_points(mesh.verts_packed())
    projected_vertices = projected_vertices.cpu().float().detach().numpy()
    screen_vertices = cameras.transform_points_screen(
        mesh.verts_packed(),
        image_size=(img_h, img_w)
    )[:, :2].cpu().detach().numpy().astype(int)

    verts_vis_arr = np.zeros(len(vertices), dtype=bool)
    min_depth_arr = np.ones((img_h, img_w), dtype=np.int32) * -1
    for i, (x, y) in enumerate(screen_vertices):
        if x < 0 or x >= img_w or y < 0 or y >= img_h:
            verts_vis_arr[i] = False  
            continue
        # compare vertex depth with depth map (need to consider nan)
        depth = projected_vertices[i, 2]
        if depth < 0: #  behind the camera
            continue 
        if min_depth_arr[y][x] < 0:
            min_depth_arr[y][x] = i 
        else:
            cur_midx = min_depth_arr[y][x]
            cur_mdepth = projected_vertices[cur_midx, 2]
            if depth < cur_mdepth:
                min_depth_arr[y][x] = i

    for i, (x, y) in enumerate(screen_vertices):
        if x < 0 or x >= img_w or y < 0 or y >= img_h:
            continue
        if min_depth_arr[y][x] < 0:
            continue             
        cur_idx = min_depth_arr[y][x]
        verts_vis_arr[cur_idx] = True 
    
    faces_np = faces[0].cpu().detach().numpy()
    faces_vis_arr = np.zeros(faces_np.shape[0], dtype=bool)
    for fi, face in enumerate(faces_np):
        vis_cnt = 0
        for vidx in face:
            if verts_vis_arr[vidx]:
                vis_cnt += 1
        if vis_cnt >= 2:
            faces_vis_arr[fi] = True

    # for i, v in enumerate(screen_vertices):
    #     # if not verts_vis_arr[i]:
    #     #     continue
    #     x = int(v[0])
    #     y = int(v[1])
    #     if verts_vis_arr[i]:
    #         cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
    #     # else:
    #     #     cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    # cv2.imwrite("./check.jpg", img)
    # print("=== visibility", np.count_nonzero(verts_vis_arr), flush=True)
    # print(np.amax(screen_vertices), np.amin(screen_vertices), flush=True)
    # exit(0)
    return np.where(verts_vis_arr)[0], np.where(faces_vis_arr)[0]

def batch_global_rotation(rot_mat):
    """
    rot_mat: [b, 22, 3, 3]
    """
    global_rotmat = []
    for item in rot_mat:
        parents = torch.tensor([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
            16, 17, 18, 19], dtype=torch.int64)
        transforms_mat = item.clone()
        transform_chain = [transforms_mat[0].detach()] # pelvis
        
        for i in range(1, parents.shape[0]):
            # Subtract the joint location at the rest pose
            # No need for rotation, since it's identity when at rest
            curr_res = torch.matmul(transform_chain[parents[i]],
                                    transforms_mat[i])
            transform_chain.append(curr_res)
        transforms = torch.stack(transform_chain, dim=0)
        global_rotmat.append(transforms)

    batch_global_rotmat = torch.stack(global_rotmat, dim=0)
    return batch_global_rotmat
    
def elbow_twist_adaptive_integration(tpose_joints, body_rotmat):
    lelbow_idx = 18
    relbow_idx = 19
    lwrist_idx = 20
    rwrist_idx = 21

    lelbow_twist_axis = nn.functional.normalize(tpose_joints[:, lwrist_idx] - tpose_joints[:, lelbow_idx], dim=1)
    relbow_twist_axis = nn.functional.normalize(tpose_joints[:, rwrist_idx] - tpose_joints[:, relbow_idx], dim=1)

    opt_lwrist = body_rotmat[:, lwrist_idx]
    opt_rwrist = body_rotmat[:, rwrist_idx]
    lelbow_twist, lelbow_twist_angle = compute_twist_rotation(opt_lwrist, lelbow_twist_axis)
    relbow_twist, relbow_twist_angle = compute_twist_rotation(opt_rwrist, relbow_twist_axis)

    min_angle = -0.4 * float(np.pi)
    max_angle = 0.4 * float(np.pi)

    lelbow_twist_angle[lelbow_twist_angle==torch.clamp(lelbow_twist_angle, min_angle, max_angle)]=0
    relbow_twist_angle[relbow_twist_angle==torch.clamp(relbow_twist_angle, min_angle, max_angle)]=0
    lelbow_twist_angle[lelbow_twist_angle > max_angle] -= max_angle
    lelbow_twist_angle[lelbow_twist_angle < min_angle] -= min_angle
    relbow_twist_angle[relbow_twist_angle > max_angle] -= max_angle
    relbow_twist_angle[relbow_twist_angle < min_angle] -= min_angle

    lelbow_twist = aa_to_rotmat(lelbow_twist_axis * lelbow_twist_angle)
    relbow_twist = aa_to_rotmat(relbow_twist_axis * relbow_twist_angle)

    opt_lwrist = torch.bmm(lelbow_twist.transpose(1, 2), opt_lwrist)
    opt_rwrist = torch.bmm(relbow_twist.transpose(1, 2), opt_rwrist)
    opt_lelbow = torch.bmm(body_rotmat[:, lelbow_idx], lelbow_twist)
    opt_relbow = torch.bmm(body_rotmat[:, relbow_idx], relbow_twist)

    new_rotmat = torch.cat([body_rotmat[:, :lelbow_idx],
        opt_lelbow.unsqueeze(1), opt_relbow.unsqueeze(1), 
        opt_lwrist.unsqueeze(1), opt_rwrist.unsqueeze(1), 
        body_rotmat[:, rwrist_idx+1:]], 1)
    return new_rotmat

def run_smplx(smplx_params, smplx_model, device):
    for k in smplx_params:
        smplx_params[k] = smplx_params[k].to(device)
    
    # create smplx class for inference
    out = smplx_model(
        betas=smplx_params['betas'],
        global_orient=smplx_params['root_pose'],
        body_pose=smplx_params['body_pose'],
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
    }
    return res

def run_smplx_with_mano(smplx_params, mano_params, smplx_model, device):
    for k in smplx_params:
        smplx_params[k] = smplx_params[k].to(device)
    
    lelbow_idx = 18
    relbow_idx = 19
    lwrist_idx = 20
    rwrist_idx = 21

    batch = smplx_params['root_pose'].size(0)
    root_pose_mat = aa_to_rotmat(smplx_params['root_pose'].reshape(-1, 3)).reshape(batch, -1, 3, 3)
    body_pose_mat = aa_to_rotmat(smplx_params['body_pose'].reshape(-1, 3)).reshape(batch, -1, 3, 3)
    full_body_rotmat = torch.cat((root_pose_mat, body_pose_mat), dim=1)
    root_rotmat_chain = batch_global_rotation(full_body_rotmat)

    lhand_pose = None 
    if mano_params["left"] is not None:
        lwrist_global_orient = aa_to_rotmat(mano_params["left"]["root_pose"].reshape(-1, 3)).reshape(batch, -1, 3, 3)
        lelbow_global_rotmat_inv = torch.linalg.inv(root_rotmat_chain[:, lelbow_idx])
        lwrist_local_rotmat = torch.matmul(lelbow_global_rotmat_inv, lwrist_global_orient)
        full_body_rotmat[:, lwrist_idx:(lwrist_idx+1)] = lwrist_local_rotmat

        lhand_pose = mano_params["left"]["hand_pose"].clone()
    
    rhand_pose = None 
    if mano_params["right"] is not None:
        rwrist_global_orient = aa_to_rotmat(mano_params["right"]["root_pose"].reshape(-1, 3)).reshape(batch, -1, 3, 3)           
        relbow_global_rotmat_inv = torch.linalg.inv(root_rotmat_chain[:, relbow_idx])
        rwrist_local_rotmat = torch.matmul(relbow_global_rotmat_inv, rwrist_global_orient)                
        full_body_rotmat[:, rwrist_idx:(rwrist_idx+1) ] = rwrist_local_rotmat

        rhand_pose = mano_params["right"]["hand_pose"].clone()
    
    # root_rotmat_chain = batch_global_rotation(full_body_rotmat)
    # print("=== lwrist", torch.equal(lwrist_global_orient, root_rotmat_chain[:, lwrist_idx]), flush=True)
    # print("=== rwrist", rwrist_global_orient, flush=True)
    # print("=== rchain", root_rotmat_chain[:, rwrist_idx], flush=True)
    # exit(0)
    tpose_out = smplx_model(return_verts=True)
    rest_pose = tpose_out.joints[:, :22].clone()
    full_body_rotmat = elbow_twist_adaptive_integration(rest_pose, full_body_rotmat)

    full_body_aa = matrix_to_axis_angle(full_body_rotmat)
    global_orient = full_body_aa[:, 0].clone()
    body_pose = full_body_aa[:, 1:22].clone()
    # create smplx class for inference
    out = smplx_model(
        betas=smplx_params['betas'],
        global_orient=global_orient,
        body_pose=body_pose,
        left_hand_pose=lhand_pose,
        right_hand_pose=rhand_pose,
        return_verts=True,
        return_full_pose=True 
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
        'full_pose': out.full_pose,
    }
    return res

def run_smplxlayer(smplx_params, smplx_layer, device):
    for k in smplx_params:
        smplx_params[k] = smplx_params[k].to(device)
    
    batch = smplx_params['root_pose'].size(0)
    root_pose_mat = smplx_params['root_pose'].clone()
    body_pose_mat = smplx_params['body_pose'].clone()
    lhand_pose_mat = smplx_params["left_hand_pose"].clone()
    rhand_pose_mat = smplx_params["right_hand_pose"].clone()
    
    out = smplx_layer(
        betas=smplx_params['betas'],
        global_orient=root_pose_mat,
        body_pose=body_pose_mat,
        left_hand_pose=lhand_pose_mat,
        right_hand_pose=rhand_pose_mat,
        transl=smplx_params['cam_trans'],
        return_verts=True,
        return_full_pose=True,
    )
    # import pdb; pdb.set_trace()
    # smplx inference 
    # coordinate convertion from bbox to full image
    verts = out.vertices #+ smplx_params['cam_trans'][:,None,:]
    joints = out.joints #+ smplx_params['cam_trans'][:,None,:]
    res = {
        'vertices': verts,
        'faces': smplx_layer.faces,
        'joints': joints,
        'full_pose': out.full_pose,
    }
    return res

def run_smplxlayer_with_mano(smplx_params, mano_params, smplx_layer, device):
    for k in smplx_params:
        smplx_params[k] = smplx_params[k].to(device)
    
    lelbow_idx = 18
    relbow_idx = 19
    lwrist_idx = 20
    rwrist_idx = 21

    batch = smplx_params['root_pose'].size(0)
    root_pose_mat = smplx_params['root_pose'].clone()
    body_pose_mat = smplx_params['body_pose'].clone()
    full_body_rotmat = torch.cat((root_pose_mat, body_pose_mat), dim=1)
    root_rotmat_chain = batch_global_rotation(full_body_rotmat)

    lhand_pose = None 
    if mano_params["left"] is not None:
        lwrist_global_orient = mano_params["left"]["root_pose"].clone()
        lelbow_global_rotmat_inv = torch.linalg.inv(root_rotmat_chain[:, lelbow_idx])
        lwrist_local_rotmat = torch.matmul(lelbow_global_rotmat_inv, lwrist_global_orient)
        full_body_rotmat[:, lwrist_idx:(lwrist_idx+1)] = lwrist_local_rotmat

        lhand_pose = mano_params["left"]["hand_pose"].clone()
    
    rhand_pose = None 
    if mano_params["right"] is not None:
        rwrist_global_orient = mano_params["right"]["root_pose"].clone()
        relbow_global_rotmat_inv = torch.linalg.inv(root_rotmat_chain[:, relbow_idx])
        rwrist_local_rotmat = torch.matmul(relbow_global_rotmat_inv, rwrist_global_orient)                
        full_body_rotmat[:, rwrist_idx:(rwrist_idx+1) ] = rwrist_local_rotmat

        rhand_pose = mano_params["right"]["hand_pose"].clone()
    
    tpose_out = smplx_layer(return_verts=True)
    rest_pose = tpose_out.joints[:, :22].clone()
    full_body_rotmat = elbow_twist_adaptive_integration(rest_pose, full_body_rotmat)
    # create smplx class for inference
    out = smplx_layer(
        betas=smplx_params['betas'],
        global_orient=full_body_rotmat[:, 0:1],
        body_pose=full_body_rotmat[:, 1:22],
        left_hand_pose=lhand_pose,
        right_hand_pose=rhand_pose,
        transl=smplx_params['cam_trans'],
        return_verts=True,
        return_full_pose=True,
    )
    # import pdb; pdb.set_trace()
    # smplx inference 
    # coordinate convertion from bbox to full image
    verts = out.vertices #+ smplx_params['cam_trans'][:,None,:]
    joints = out.joints #+ smplx_params['cam_trans'][:,None,:]
    res = {
        'vertices': verts,
        'faces': smplx_layer.faces,
        'joints': joints,
        'full_pose': out.full_pose,
    }
    return res

def cal_surface_area(verts, faces, visible_faces, vert_ids):
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

def cal_surface_area2d(verts, faces, visible_faces, vert_ids):
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
    
    v0_v1 = v1 - v0 
    v0_v2 = v2 - v0 
    cross_product = v0_v1[..., 0] * v0_v2[..., 1] - v0_v1[..., 1] * v0_v2[..., 0]
    areas = 0.5 * torch.abs(cross_product)
    return torch.sum(areas)


def select_best_view(smplx_param, mano_params, cam_params, vis_img_arr, device, smplx_model):
    smplx_mano_vert_path = '/mnt/petrelfs/luohuiwen/IMU/MV_MoCap/h3wb/human_models/smplx/MANO_SMPLX_vertex_ids.pkl'
    smplx_mano_vert_ids = pickle.load(open(smplx_mano_vert_path,'rb'))
     
    rarea_max, larea_max = 0, 0
    lview, rview = 3, 3
    view_num = len(cam_params)
    for view in range(view_num):
        vis_img = vis_img_arr[view].copy()
        img_h, img_w, _ = vis_img.shape 
        rasterizer = get_rasterizer(img_h, img_w)

        Rmat = cam_params[view]["R"]
        tvec = cam_params[view]["t"]
        focal = cam_params[view]["focal"]
        princpt = cam_params[view]["princpt"]

        intri_mat = np.eye(3, dtype=np.float32)
        intri_mat[0, 0] = focal[0]
        intri_mat[1, 1] = focal[1]
        intri_mat[0, 2] = princpt[0]
        intri_mat[1, 2] = princpt[1]
        intri_mat_pt = torch.from_numpy(intri_mat).unsqueeze(0).to(device)

        extri_mat = np.eye(4, dtype=np.float32)
        extri_mat[:3, :3] = Rmat.copy()
        extri_mat[:3, 3] = tvec.copy()
        extri_mat_pt = torch.from_numpy(extri_mat).unsqueeze(0).to(device)

        # res = run_smplx(smplx_param, smplx_model, device)
        res = run_smplx_with_mano(smplx_param, mano_params[view], smplx_model, device)
        faces = torch.from_numpy(res["faces"].astype(np.int64)).unsqueeze(0).to(device)
        verts = res["vertices"].clone()
        batch, verts_num, _ = verts.size()

        ones = torch.ones((batch, verts_num, 1), dtype=torch.float32, device=device)
        verts_homo = torch.cat([verts, ones], dim=2)
        verts_cam = torch.matmul(verts_homo, extri_mat_pt.permute(0, 2, 1))[..., :3]
        verts_cam_homo = verts_cam.clone()
        verts_cam_homo[:, :, :2] /= verts_cam_homo[:, :, 2:3]
        verts_img =  torch.cat([verts_cam_homo[..., :2], ones], dim=2)
        verts_img = torch.matmul(verts_img, intri_mat_pt.permute(0, 2, 1))[..., :2]
        vis_img = vis_img_arr[view]
        # for kpt2d in verts_img[0].cpu().detach().numpy():
        #     x = int(kpt2d[0])
        #     y = int(kpt2d[1])
        #     cv2.circle(vis_img, (x, y), 3, (0, 255, 0), -1)

        # render_img = render_mesh_pt3d(vis_img, verts_cam, faces, cam_params[view], rasterizer)
        # cv2.imwrite(f"./check_{view:02d}.jpg", render_img )
        verts_vis_arr, faces_vis_arr = check_visibility_pt3d(rasterizer, vis_img, verts_cam, faces, cam_params[view])
        # rarea = cal_surface_area2d(verts_img[0], faces[0], faces_vis_arr, smplx_mano_vert_ids['right_hand'])
        # larea = cal_surface_area2d(verts_img[0], faces[0], faces_vis_arr, smplx_mano_vert_ids['left_hand'])
        rarea = cal_surface_area(verts_cam[0], faces[0], faces_vis_arr, smplx_mano_vert_ids['right_hand'])
        larea = cal_surface_area(verts_cam[0], faces[0], faces_vis_arr, smplx_mano_vert_ids['left_hand'])
        # print(f"{view} area", rarea, larea, flush=True)
        
        if rarea > rarea_max:
            rarea_max = rarea
            rview = view
        if larea > larea_max:
            larea_max = larea
            lview = view

    # print("=== best", lview, rview, flush=True)
    # print("=== area", larea, rarea, flush=True)
    return lview, rview

