import sys 
from pathlib import Path
import torch
import argparse
import os
import os.path as osp 
import cv2
import smplx
from smplx.joint_names import JOINT_NAMES
import imageio 
import numpy as np
from tqdm import tqdm, trange
from itertools import combinations
import datetime

import sys 
hamer_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(hamer_path, "dependencies"))

from hamer.utils.geometry import perspective_projection, aa_to_rotmat, matrix_to_axis_angle
from hamer.mv.joint_vis import (
    get_rasterizer,
    select_best_view,
    run_smplxlayer_with_mano,
    run_smplxlayer,
)
from hamer.mv.joint_vis_orig import rigid_align_batched
import json
import copy
from typing import Dict, Optional
from scipy.spatial.transform import Rotation as R

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    BlendParams,
    look_at_view_transform,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardGouraudShader,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.utils import cameras_from_opencv_projection


def save_verts(verts, file_name='a.obj'):
    with open(file_name, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / cam_coord[:, 2] * f[0] + c[0]
    y = cam_coord[:, 1] / cam_coord[:, 2] * f[1] + c[1]
    z = cam_coord[:, 2]
    return np.stack((x, y, z), 1)

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)
    return cam_coord

def get_pose_c2w(pose_cam, c2w, return_mat=True):
    batch = pose_cam.size(0)
    device = pose_cam.device 
    pose_mat_cam = aa_to_rotmat(pose_cam.reshape(-1, 3)).reshape(batch, -1, 3, 3)
    pose_mat_world = torch.matmul(c2w.to(device), pose_mat_cam)
    if return_mat:
        return pose_mat_world.reshape(batch, 3, 3)

    pose_world = matrix_to_axis_angle(pose_mat_world).reshape(batch, 3)
    return pose_world


def get_hand_params(lhand_res, rhand_res, R_inv_dict=None, c2w=False):
    
    hand_dict = {"left": None, "right": None} 
    if lhand_res is not None and lhand_res['l_init_root_pose'].size().numel() > 0 :
        hand_dict["left"] = {}
        idx = 0
        root_pose = lhand_res['l_init_root_pose'][:, idx*3:(idx+1)*3].reshape(-1, 3)
        root_pose[..., 1] *= -1
        root_pose[..., 2] *= -1
        if c2w:
            hand_dict["left"]["root_pose"] = get_pose_c2w(root_pose, R_inv_dict["left"], return_mat=True)
        else:
            hand_dict["left"]["root_pose"] = aa_to_rotmat(root_pose)
        hand_pose = lhand_res['l_init_hand_pose'][:, idx*45:(idx+1)*45].reshape(15, 3)
        hand_pose[..., 1] *= -1
        hand_pose[..., 2] *= -1
        hand_dict["left"]["hand_pose"] = aa_to_rotmat(hand_pose).unsqueeze(0)
        hand_dict["left"]["shape"] = lhand_res['lshape'][idx:idx+1]

    if rhand_res is not None and rhand_res['r_init_root_pose'].size().numel() > 0 :
        # for k, v in rhand_res.items():
        #     if isinstance(v, torch.Tensor):
        #         print(k, v.size(), flush=True)
        hand_dict["right"] = {}
        idx = 0
        root_pose = rhand_res['r_init_root_pose'][:, idx*3:(idx+1)*3].reshape(-1, 3)
        if c2w:
            hand_dict["right"]["root_pose"] = get_pose_c2w(root_pose, R_inv_dict["right"], return_mat=True)
        else:
            hand_dict["right"]["root_pose"] = aa_to_rotmat(root_pose) 
        hand_pose = rhand_res['r_init_hand_pose'][:, idx*45:(idx+1)*45].reshape(15, 3)
        hand_dict["right"]["hand_pose"] = aa_to_rotmat(hand_pose).unsqueeze(0)
        hand_dict["right"]["shape"] = rhand_res['rshape'][idx:idx+1]
            
    return hand_dict        

def load_sv_data(sv_smplx_path, sv_hamer_path):
    sv_smplx_arr = np.load(sv_smplx_path, allow_pickle=True)
    sv_hamer_arr = np.load(sv_hamer_path, allow_pickle=True)

    sv_smplx_len = len(sv_smplx_arr)
    sv_hamer_len = len(sv_hamer_arr)

    max_len = max(sv_smplx_len, sv_hamer_len)
    out_sv_arr = []
    for i in range(max_len):
        out_sv_arr.append(dict())
    
    for i in range(max_len):
        if i < sv_smplx_len and "betas" in sv_smplx_arr[i]:
            smplx_fidx = sv_smplx_arr[i]['fidx']
            for k, v in sv_smplx_arr[i].items():
                if "fidx" in k or "kpt2d" in k:
                    continue 
                out_sv_arr[smplx_fidx][k] = v
        
        if i < sv_hamer_len and "left" in sv_hamer_arr[i]:
            hamer_fidx = sv_hamer_arr[i]['fidx']
            for k, v in sv_hamer_arr[i].items():
                if "fidx" in k:
                    continue 
                out_sv_arr[hamer_fidx][k] = v
        
    return out_sv_arr        

def get_high_conf_hand_num(hand_dict, kpt2d_key):
    if kpt2d_key not in hand_dict:
        return None  
    
    kpt2d = np.array(hand_dict[kpt2d_key])
    if 0 in kpt2d.shape:
        return None

    if kpt2d.ndim == 3:
        kpt2d = kpt2d[0].copy()

    conf = kpt2d[:, -1].copy()
    min_value = np.min(conf)
    max_value = np.max(conf)
    kpt2d[:, -1] = (conf - min_value) / (max_value - min_value)

    kpt2d_filter = kpt2d[kpt2d[:, -1] >= 0.75] ### 0.65 0.75
    return kpt2d_filter

def select_best_hand_view(all_sv_arr, fidx):
    cam_len = len(all_sv_arr)
    best_dict = {"left": dict(), "right": dict()}
    best_dict["left"]["max_num"] = 0
    best_dict["left"]["max_vi"] = 0
    best_dict["left"]["max_kpt2d"] = None

    best_dict["right"]["max_num"] = 0
    best_dict["right"]["max_vi"] = 0
    best_dict["right"]["max_kpt2d"] = None 

    for side, kpt2d_k in zip(["left", "right"], ["lkeyp", "rkeyp"]):
        for vi in range(cam_len):
            sv_dict = all_sv_arr[vi][fidx]
            if side not in sv_dict:
                continue 

            kpt2d_filter = get_high_conf_hand_num(sv_dict[side], kpt2d_k)
            if kpt2d_filter is None:
                kpt2d_num = 0 
            else:
                if kpt2d_filter.ndim > 2:
                    kpt2d_num = len(kpt2d_filter[0])
                else:
                    kpt2d_num = len(kpt2d_filter)

            if kpt2d_num > best_dict[side]["max_num"]:
                best_dict[side]["max_num"] = kpt2d_num
                best_dict[side]["max_vi"] = vi 
                best_dict[side]["max_kpt2d"] = kpt2d_filter 

            elif kpt2d_num == best_dict[side]["max_num"] and kpt2d_filter is not None:
                src_conf = np.sum(kpt2d_filter[..., -1])
                tgt_kpt2d = best_dict["left"]["max_kpt2d"]
                if tgt_kpt2d is None:
                    best_dict[side]["max_num"] = kpt2d_num
                    best_dict[side]["max_vi"] = vi 
                    best_dict[side]["max_kpt2d"] = kpt2d_filter
                else:
                    tgt_conf = np.sum(tgt_kpt2d[..., -1])
                    if src_conf > tgt_conf:
                        best_dict[side]["max_num"] = kpt2d_num
                        best_dict[side]["max_vi"] = vi 
                        best_dict[side]["max_kpt2d"] = kpt2d_filter 

    left_vi = best_dict["left"]["max_vi"]
    right_vi = best_dict["right"]["max_vi"]
    return left_vi, right_vi 

def render_mesh_cam(img, verts, faces, cam_param, rasterizer=None, alpha=0.8):
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
    if rasterizer is None: 
        raster_settings = RasterizationSettings(
            image_size=(img_h, img_w), 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            bin_size = 0,  # this setting controls whether naive or coarse-to-fine rasterization is used
            max_faces_per_bin = None  # this setting is for coarse rasterization
        )
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
    
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
    output_image = img * (1.0 - render_mask) + (img * (1.0 - alpha) + render_img * alpha) * render_mask
    output_image = output_image.astype(np.uint8)
    return output_image

def render_mesh_world(img, verts, faces, cam_param, sam_mask=None, alpha=0.8):
    device = verts.device
    img_h, img_w = img.shape[:2]
    image_size = torch.tensor([img_h, img_w]).unsqueeze(0).to(device)
    
    intri_mat = np.eye(3, dtype=np.float32)
    intri_mat[0][0] = cam_param['focal'][0]
    intri_mat[1][1] = cam_param['focal'][1]
    intri_mat[0][2] = cam_param['princpt'][0]
    intri_mat[1][2] = cam_param['princpt'][1]
    intri_mat_pt = torch.from_numpy(intri_mat).unsqueeze(0).float().to(device)

    R_pt = torch.from_numpy(cam_param['R']).unsqueeze(0).float().to(device)
    tvec_pt = torch.from_numpy(cam_param['t']).unsqueeze(0).float().to(device)
    cameras = cameras_from_opencv_projection(
        R=R_pt, tvec=tvec_pt, camera_matrix=intri_mat_pt, image_size=image_size
    )

    blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))
    raster_settings = RasterizationSettings(
        image_size=(img_h, img_w), 
        blur_radius=0.0, 
        faces_per_pixel=1, 
        bin_size = 0,  # this setting controls whether naive or coarse-to-fine rasterization is used
        max_faces_per_bin = None  # this setting is for coarse rasterization
    )
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )
    
    cam_center = -np.matmul(cam_param['R'].transpose(), cam_param['t'][..., None])
    light_pos = cam_center.copy().squeeze()
    light_pos[1] += 1.0
    lights = PointLights(device=device, location=(tuple(light_pos),), specular_color=((0.0, 0.0, 0.0),))
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
    output_image = img * (1.0 - render_mask) + (img * (1.0 - alpha) + render_img * alpha) * render_mask
    output_image = output_image * (1.0 - sam_mask) + (output_image * 0.5 + np.array([0, 0, 255]) * 0.5) * sam_mask
    output_image = output_image.astype(np.uint8)
    return output_image

def parse_args():
    parser = argparse.ArgumentParser(description="HaMeR demo code")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--sam_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    
    parser.add_argument('--use_mano', action='store_true')
    args = parser.parse_args()
    return args

def main():
    device = "cuda"
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print("=== use_mano", args.use_mano, flush=True)

    # Setup the renderer
    cur_file_path = os.path.abspath(__file__)
    # work_root = os.path.dirname(os.path.dirname(cur_file_path))
    # human_model_dir = os.path.join(work_root, "h3wb/human_models")
    work_root = os.path.dirname(os.path.dirname(cur_file_path))
    human_model_dir = os.path.join(work_root, "human_models", "human_model_files")
    smplx_model = smplx.SMPLXLayer(human_model_dir+'/smplx', gender='NEUTRAL', use_pca=False, use_face_contour=True)
    pose_mean = smplx_model.pose_mean.reshape(-1, 3)
    smplx_model.to(device)
    faces_tensor = torch.from_numpy(smplx_model.faces.astype(np.int32)).unsqueeze(0).to(device)

    cam_json_path = os.path.join(args.video_dir, "cam_info.json")
    with open(cam_json_path, 'r') as fin:
        cam_dict = json.load(fin)

    video_filenames = sorted([x for x in os.listdir(args.video_dir) if x.endswith(".mp4") or x.endswith(".mkv")])
    
    hamer_res_dir = os.path.join(args.data_dir, "hamer")
    sv_res_dir = os.path.join(args.data_dir, "smplestx")
    mv_res_dir = os.path.join(args.data_dir, "smplify/action")

    mv_smplx_path = os.path.join(mv_res_dir, "human_data_tri_smplx.npz")
    mv_smplx_dict = dict(np.load(mv_smplx_path, allow_pickle=True))
    mv_len = len(mv_smplx_dict["betas"])
    
    cam_info_arr = []
    all_sv_arr = []
    for video_fn in video_filenames:
        video_name = video_fn.split(".")[0]
        sv_smplx_path = os.path.join(sv_res_dir, f"{video_name}_res.npy")
        hamer_path = os.path.join(hamer_res_dir, f"{video_name}_res.npy")
        out_sv_arr = load_sv_data(sv_smplx_path, hamer_path)
        all_sv_arr.append(out_sv_arr)

        cam_info = cam_dict[video_fn]
        cam_param = {}
        cam_param['R'] = np.array(cam_info['R'], dtype=np.float32)
        cam_param['t'] = np.array(cam_info['T'], dtype=np.float32)
        K = cam_info['K']
        cam_param['focal'] = [K[0][0], K[1][1]]
        cam_param['princpt'] = [K[0][2], K[1][2]]

        cam_str = video_name.split("_")[-1]
        cam_param['cam_str'] = cam_str
        cam_info_arr.append(cam_param)

    cam_len = len(all_sv_arr)
    front_vi = 1
    sv_len = len(all_sv_arr[front_vi])

    video_path = os.path.join(args.video_dir, video_filenames[front_vi])
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    render_cam_dict = cam_info_arr[front_vi]
    v_root, ext = os.path.splitext(video_filenames[front_vi])
    sam_path = os.path.join(args.sam_dir, "mask", v_root)
    if os.path.exists(sam_path):
        sam_mask = [os.path.join(sam_path, x) for x in sorted(os.listdir(sam_path)) if x.endswith(".png")]
    else:
        sam_mask = None

    if args.use_mano:
        out_video_path = os.path.join(args.out_dir, f"render_smplx_with_mano_sam.mp4")
    else:
        out_video_path = os.path.join(args.out_dir, f"render_smplx_without_mano.mp4")

    writer = imageio.get_writer(
        out_video_path, 
        fps=fps, mode='I', format='FFMPEG', macro_block_size=1
    )

    prev_full_pose = None 
    prev_lhand_res = None 
    prev_rhand_res = None 
    out_arr = []
    hand_dicts = {}
    for fidx in trange(sv_len):
        if fidx >= len(mv_smplx_dict["fullpose"]):
            continue 

        if args.use_mano:
            lview, rview = select_best_hand_view(all_sv_arr, fidx)

            lhand_res = None 
            if "left" in all_sv_arr[lview][fidx]:
                lhand_res = {k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v for k, v in all_sv_arr[lview][fidx]["left"].items()}
                prev_lhand_res = lhand_res 
            elif prev_lhand_res is not None:
                print(f"=== fill {fidx} left with previous", flush=True)
                lhand_res = prev_lhand_res
            else:
                print('=== no left', lview, all_sv_arr[lview][fidx].keys(), flush=True)

            rhand_res = None 
            if "right" in all_sv_arr[rview][fidx]:
                rhand_res = {k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v for k, v in all_sv_arr[rview][fidx]["right"].items()}
                prev_rhand_res = rhand_res 
            elif prev_rhand_res is not None:
                print(f"=== fill {fidx} right with previous", flush=True)
                rhand_res = prev_rhand_res 
            else:
                print('=== no right', rview, all_sv_arr[rview][fidx].keys(), flush=True)

            left_cam_param = cam_info_arr[lview]
            R_inv_left = left_cam_param["R"].transpose()

            right_cam_param = cam_info_arr[rview]
            R_inv_right = right_cam_param["R"].transpose()

            R_inv_dict = {
                "left": torch.from_numpy(R_inv_left).reshape(1, 1, 3, 3),
                "right": torch.from_numpy(R_inv_right).reshape(1, 1, 3, 3)
            }
            hand_dict = get_hand_params(lhand_res, rhand_res, R_inv_dict=R_inv_dict, c2w=True)
            hand_dicts[fidx] = hand_dict
    
    import scipy.signal as signal
    def signal_filter1(input_array) :
        n = input_array.shape[0]
        flat = input_array.reshape(n, -1) 
        filtered = signal.medfilt(flat, kernel_size=(9, 1))
        smoothed = signal.savgol_filter(filtered, window_length=15, polyorder=1, axis=0)
        return smoothed.reshape(n, 1, 3, 3)
    
    def signal_filter2(input_array) :
        n = input_array.shape[0]
        flat = input_array.reshape(n, -1) 
        filtered = signal.medfilt(flat, kernel_size=(9, 1))
        smoothed = signal.savgol_filter(filtered, window_length=15, polyorder=1, axis=0)
        return smoothed.reshape(n, 1, 15, 3, 3)

    left_root_pose = []
    left_hand_pose = []
    right_root_pose = []
    right_hand_pose = []
    valid_left_keys = []
    valid_right_keys = []

    for k, v in hand_dicts.items() :
        if v["left"] is not None :
            left_root_pose.append(v["left"]["root_pose"].squeeze().cpu().numpy())
            left_hand_pose.append(v["left"]["hand_pose"].squeeze().cpu().numpy())
            valid_left_keys.append(k)
        if v["right"] is not None :
            right_root_pose.append(v["right"]["root_pose"].squeeze().cpu().numpy())
            right_hand_pose.append(v["right"]["hand_pose"].squeeze().cpu().numpy())
            valid_right_keys.append(k)
    left_root_pose = signal_filter1(np.array(left_root_pose))
    left_hand_pose = signal_filter2(np.array(left_hand_pose))
    right_root_pose = signal_filter1(np.array(right_root_pose))
    right_hand_pose = signal_filter2(np.array(right_hand_pose))
    
    for idx, k in enumerate(valid_left_keys):
        hand_dicts[k]["left"]["root_pose"] = torch.tensor(left_root_pose[idx]).to(device)
        hand_dicts[k]["left"]["hand_pose"] = torch.tensor(left_hand_pose[idx]).to(device)
    for idx, k in enumerate(valid_right_keys):
        hand_dicts[k]["right"]["root_pose"] = torch.tensor(right_root_pose[idx]).to(device)
        hand_dicts[k]["right"]["hand_pose"] = torch.tensor(right_hand_pose[idx]).to(device)
    
    
    prev_full_pose = None 
    prev_lhand_res = None 
    prev_rhand_res = None 
    out_arr = []
    for fidx in trange(sv_len):
        if fidx >= len(mv_smplx_dict["fullpose"]):
            continue 

        smplx_fullpose = torch.from_numpy(mv_smplx_dict["fullpose"][fidx])
        smplx_fullpose += pose_mean
        smplx_fullpose_mat = aa_to_rotmat(smplx_fullpose).unsqueeze(0)

        smplx_dict_pt = {}
        smplx_dict_pt["root_pose"] = smplx_fullpose_mat[:, 0:1].clone()
        smplx_dict_pt["body_pose"] = smplx_fullpose_mat[:, 1:22].clone()
        smplx_dict_pt["left_hand_pose"] = smplx_fullpose_mat[:, 25:40].clone()
        smplx_dict_pt["right_hand_pose"] = smplx_fullpose_mat[:, 40:55].clone()
        smplx_dict_pt["cam_trans"] = torch.from_numpy(mv_smplx_dict["transl"][fidx]).float().unsqueeze(0)
        smplx_dict_pt["betas"] = torch.from_numpy(mv_smplx_dict["betas"][fidx]).float().unsqueeze(0)

        if args.use_mano:
            lview, rview = select_best_hand_view(all_sv_arr, fidx)

            lhand_res = None 
            if "left" in all_sv_arr[lview][fidx]:
                lhand_res = {k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v for k, v in all_sv_arr[lview][fidx]["left"].items()}
                prev_lhand_res = lhand_res 
            elif prev_lhand_res is not None:
                print(f"=== fill {fidx} left with previous", flush=True)
                lhand_res = prev_lhand_res
            else:
                print('=== no left', lview, all_sv_arr[lview][fidx].keys(), flush=True)

            rhand_res = None 
            if "right" in all_sv_arr[rview][fidx]:
                rhand_res = {k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v for k, v in all_sv_arr[rview][fidx]["right"].items()}
                prev_rhand_res = rhand_res 
            elif prev_rhand_res is not None:
                print(f"=== fill {fidx} right with previous", flush=True)
                rhand_res = prev_rhand_res 
            else:
                print('=== no right', rview, all_sv_arr[rview][fidx].keys(), flush=True)

            left_cam_param = cam_info_arr[lview]
            R_inv_left = left_cam_param["R"].transpose()

            right_cam_param = cam_info_arr[rview]
            R_inv_right = right_cam_param["R"].transpose()

            R_inv_dict = {
                "left": torch.from_numpy(R_inv_left).reshape(1, 1, 3, 3),
                "right": torch.from_numpy(R_inv_right).reshape(1, 1, 3, 3)
            }
            hand_dict = hand_dicts[fidx]

            mv_res = run_smplxlayer_with_mano(smplx_dict_pt, hand_dict, smplx_model, device)
        else:
            mv_res = run_smplxlayer(smplx_dict_pt, smplx_model, device)

        prev_full_pose = mv_res["full_pose"].clone()
        
        ##### visualization
        ret, bgr_img = cap.read()
        vis_img = bgr_img.copy()
        if sam_mask is not None and len(sam_mask) == sv_len:
            sam_m = cv2.imread(sam_mask[fidx], cv2.IMREAD_UNCHANGED)
        else :
            sam_m = np.zeros_like(vis_img[..., 0])
        sam_m = sam_m[..., None]
        mv_render_img = render_mesh_world(vis_img, mv_res['vertices'], faces_tensor, render_cam_dict, sam_m)
        mv_render_img = cv2.cvtColor(mv_render_img, cv2.COLOR_BGR2RGB)
        cv2.putText(mv_render_img, "%04d" % fidx, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 1, cv2.LINE_AA)
        writer.append_data(mv_render_img)
        
        ##### pack data
        full_pose_mat_np = mv_res["full_pose"][0].cpu().detach().numpy()
        out_frame_dict = {
            "fidx": fidx,
            "multi_view":{
                "betas": mv_smplx_dict["betas"][fidx],
                "expression": mv_smplx_dict["expression"][fidx],
                "transl": mv_smplx_dict["transl"][fidx],
                "global_orient": full_pose_mat_np[0:1],
                "body_pose": full_pose_mat_np[1:22],
                "jaw_pose": full_pose_mat_np[22:23],
                "leye_pose": full_pose_mat_np[23:24],
                "reye_pose": full_pose_mat_np[24:25],
                "left_hand_pose": full_pose_mat_np[25:40],
                "right_hand_pose": full_pose_mat_np[40:55],
                "cams": {},
            },
            "single_view": {}
        }

        for vi in range(cam_len):
            cam_param = cam_info_arr[vi]
            cam_str = cam_param['cam_str']
            out_frame_dict["multi_view"]["cams"][cam_str] = cam_param

            sv_dict = all_sv_arr[vi][fidx]
            if "full_pose" not in sv_dict:
                continue 

            smplx_dict_pt = {}
            smplx_fullpose = torch.from_numpy(sv_dict["full_pose"]).unsqueeze(0)
            smplx_dict_pt["root_pose"] = smplx_fullpose[:, 0:1].clone()
            smplx_dict_pt["body_pose"] = smplx_fullpose[:, 1:22].clone()
            smplx_dict_pt["left_hand_pose"] = smplx_fullpose[:, 25:40].clone()
            smplx_dict_pt["right_hand_pose"] = smplx_fullpose[:, 40:55].clone()
            smplx_dict_pt["cam_trans"] = torch.from_numpy(sv_dict["transl"]).float().unsqueeze(0)
            smplx_dict_pt["betas"] = torch.from_numpy(sv_dict["betas"]).float().unsqueeze(0)

            lhand_res = None 
            rhand_res = None 
            if args.use_mano:
                if "left" in sv_dict:
                    lhand_res = {k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v for k, v in sv_dict["left"].items()}
                if "right" in sv_dict:
                    rhand_res = {k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v for k, v in sv_dict["right"].items()}
                
                hand_dict = get_hand_params(lhand_res, rhand_res)
                sv_res = run_smplxlayer_with_mano(smplx_dict_pt, hand_dict, smplx_model, device)
            else:
                sv_res = run_smplxlayer(smplx_dict_pt, smplx_model, device)

            full_pose_mat_np = sv_res['full_pose'][0].cpu().detach().numpy()
            out_frame_dict["single_view"][cam_str] = {
                "betas": sv_dict["betas"],
                "expression": sv_dict["expression"],
                "transl": sv_dict["transl"],
                "global_orient": full_pose_mat_np[0:1],
                "body_pose": full_pose_mat_np[1:22],
                "jaw_pose": full_pose_mat_np[22:23],
                "leye_pose": full_pose_mat_np[23:24],
                "reye_pose": full_pose_mat_np[24:25],
                "left_hand_pose": full_pose_mat_np[25:40],
                "right_hand_pose": full_pose_mat_np[40:55],
                "cams": {"focal": sv_dict["focal"], "princpt": sv_dict["princpt"]}
            }
        
        out_arr.append(out_frame_dict)
        

    cap.release()
    writer.close()

    if args.use_mano:
        out_npy_path = os.path.join(args.out_dir, f"pack_smplx_with_mano.npy")
    else:
        out_npy_path = os.path.join(args.out_dir, f"pack_smplx_without_mano.npy")

    np.save(out_npy_path, out_arr)
    

if __name__ == '__main__':
    main()
    print("=== done", flush=True)