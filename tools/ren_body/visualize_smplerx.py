import copy
import logging
import numpy as np
import torch
import mmcv
import cv2
import io
import sys
import torch.nn.functional as F
sys.path.append('/mnt/petrelfs/lufan/zoehuman')
sys.path.append('/mnt/petrelfs/lufan/zoehuman/tools')
sys.path.append('/mnt/petrelfs/lufan/zoehuman/xrmocap')
# from mmcv.runner import load_checkpoint
from torchvision.transforms import Compose
from tqdm import tqdm
from typing import List, Tuple, Union, overload
from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrprimer.utils.log_utils import get_logger

from xrmocap.data_structure.body_model import SMPLData
from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.io.image import (
    get_n_frame_from_mview_src, load_clip_from_mview_src,
)
from xrmocap.model.architecture.builder import build_architecture
from xrmocap.model.registrant.builder import SMPLify, build_registrant
from xrmocap.transform.image.builder import build_image_transform
from xrmocap.transform.image.shape import get_affine_trans_aug
from xrmocap.transform.keypoints3d.optim.builder import (
    BaseOptimizer, build_keypoints3d_optimizer,
)

from xrmocap.utils.geometry import get_scale
from xrmocap.utils.mvp_utils import norm2absolute, process_dict

import argparse
import os
import os.path as osp
from pytorch3d.io import IO

from xrmocap.transform.convention.keypoints_convention import convert_keypoints, get_keypoint_idx
from mmhuman3d.data.data_structures.human_data import HumanData
from xrmocap.core.estimation.builder import build_estimator
from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.model.registrant.builder import SMPLify, build_registrant

# from tools.visualization.visualize_project_smplx import plot_keypoints
from visualization.visualize_project_smplx import plot_keypoints
from zoehuman.utils.path_utils import Existence, check_path_existence
from xrmocap.core.visualization.visualize_keypoints3d import visualize_project_keypoints3d

from xrprimer.data_structure.camera import FisheyeCameraParameter
from zoehuman.core.visualization.visualize_smpl import visualize_smpl_distortion

from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes
from tqdm import tqdm
from mmhuman3d.core.visualization.visualize_smpl import \
    visualize_smpl_calibration

from xrprimer.transform.camera.distortion import undistort_images
from mmhuman3d.core.visualization.visualize_smpl import \
    visualize_smpl_calibration
# yapf: enable
from petrel_client.client import Client
conf_path = '~/petreloss.conf'
client = Client(conf_path)

def parse_args():
    parser = argparse.ArgumentParser('SMPLify tools')
    # path
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--annots_file', type=str,
        help='Path to the annots.npy file',
        default=None)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--init_smpl_dir', type=str,
        help='Path to the init smpl dict, e.g. apose betas',
        default=None)
    parser.add_argument('--background_dir', type=str,
        help='Path to the background image file',
        default=None)

    # triangulation config
    parser.add_argument('--triangulate', action='store_true')
    parser.add_argument(
        '--cam_parameters_type',
        type=str,
        help='Type of camera parameters.',
        choices=['smc', 'chessboard', 'dump', 'lightstage', 'ren_body_0418'],
        default='ren_body_0418')
    parser.add_argument(
        '--image_data_dir',
        type=str,
        help='Path to human_data directory.',
        default='')
    parser.add_argument(
        '--human_data_dir',
        type=str,
        help='Path to human_data directory.',
        default='')
    parser.add_argument(
        '--human_data_prefix',
        type=str,
        help='Prefix string of human_data files.',
        default='human_data_')
    
    # smpl config
    parser.add_argument('--src_convention', type=str, default='smplx')
    parser.add_argument('--tgt_convention', type=str, default='smpl')

    parser.add_argument(
        '--model',
        type=str,
        default='smpl',
        choices=['smpl', 'smplh', 'smplx', 'manol', 'manor'])
    parser.add_argument(
        '--gender',
        type=str,
        default='neutral',
        choices=['neutral', 'male', 'female'])
    parser.add_argument('--sequence',type=str,default='')
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--kid', 
        help='Whether data is a kid.',
        action='store_true')
    parser.add_argument(
        '--kid_data_dir',
        type=str,
        help='Path to kid template directory.',
        default='mmhuman3d/data/body_models/smplx/smplx_kid_template.npy')


    # visualization
    parser.add_argument('--vis_smpl', action='store_true')
    parser.add_argument('--vis_kps', action='store_true')
    parser.add_argument('--save_mesh', action='store_true')
    parser.add_argument('--frame_interval', type=int, default=30)
    parser.add_argument('--view_idxs', nargs='+', 
        help='Indexes of views to be rendered')

    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--read_file', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--start_t', type=int, default=None)
    parser.add_argument('--end_t', type=int, default=None)
    parser.add_argument('--output_finetune', action='store_true')
    parser.add_argument('--smplerx_pose_dir', type=str, default=None)

    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    estimator_config = mmcv.Config.fromfile(args.config)
    estimator_config.smplify.body_model.update(dict(type = args.model.upper()))
    estimator_config.smplify.body_model.update(dict(gender = args.gender))

    body_model_type = ['smpl', 'smplx']
    smplify_type =  ['smplify', 'smplifyx']
    assert estimator_config.smplify.body_model.type.lower() in body_model_type
    assert estimator_config.smplify.type.lower() in smplify_type
    assert body_model_type.index(estimator_config.smplify.body_model.type.lower()) == \
        smplify_type.index(estimator_config.smplify.type.lower())
    
    ## for kid data
    if args.kid:
        estimator_config.smplify.body_model.update(dict(age = 'kid'))
        estimator_config.smplify.body_model.update(dict(kid_template_path = args.kid_data_dir))
        # it is recommended to use gender= 'male' for male kids and gender='neutral' for female kids
        if args.gender == 'female':
            estimator_config.smplify.body_model.update(dict(gender = 'neutral'))
        logger.info(f"is kid! gender: {estimator_config.smplify.body_model.gender}; template: " 
            f"{estimator_config.smplify.body_model.kid_template_path}")

    logger = get_logger()

    smplerx_fns = sorted(os.listdir(args.smplerx_pose_dir))
    fullposes = []
    transls = []
    betas = []

    for smplerx_fn in smplerx_fns:
        data = np.load(os.path.join(args.smplerx_pose_dir, smplerx_fn))
        global_orient = data['global_orient'].squeeze()
        body_pose = data['body_pose'].squeeze()
        left_hand_pose = data['left_hand_pose'].squeeze()
        right_hand_pose = data['right_hand_pose'].squeeze()
        jaw_pose = data['jaw_pose'].squeeze()
        leye_pose = data['leye_pose'].squeeze()
        reye_pose = data['reye_pose'].squeeze()
        beta = data['betas'].squeeze()
        expression = data['expression'].squeeze()
        transl = data['transl'].squeeze() / 5000.
        print(transl)
        print(global_orient.shape, body_pose.shape, jaw_pose.shape, leye_pose.shape, reye_pose.shape, left_hand_pose.shape, right_hand_pose.shape)
        fullpose = np.concatenate([global_orient[None], body_pose, jaw_pose[None], leye_pose[None], reye_pose[None], left_hand_pose, right_hand_pose], axis=0)
        fullposes.append(fullpose)
        betas.append(beta)
        transls.append(transl)
    
    fullpose = np.stack(fullposes, axis=0)
    fullpose = fullpose.reshape(-1, 165)
    betas = np.stack(betas, axis=0)
    transl = np.stack(transls, axis=0)

    vis_reso = 2
    
    for view in args.view_idxs: # 10, 25
        logger.info(f'Visualization of view {view}')
        # load background
        image_array, n_frame = get_image_array(args.background_dir, view)

        H = image_array.shape[1]
        W = image_array.shape[2]

        image_array_th = torch.from_numpy(image_array).float()
        image_array_th = image_array_th.permute(0, 3, 1, 2) / 255.
        image_array_th = F.interpolate(image_array_th, (H // vis_reso, W // vis_reso), mode='bilinear')
        image_array_th = image_array_th.permute(0, 2, 3, 1) * 255.
        image_array = image_array_th.numpy().astype(np.uint8)

        # if args.finetune:
        #     image_array = image_array[start_t:end_t]
        #     n_frame = end_t - start_t
        if n_frame == 0:
            logger.error("Frame number is 0, please check the image directory")
            raise ValueError
        src_image = image_array

        # load camera
        camera_parameter = FisheyeCameraParameter(name=view)
        K, R, T, dist_coeff_k, dist_coeff_p, dist_coeff_dict = \
            get_camera_param_from_annots(args.annots_file, view)
        
        K = K / vis_reso
        K[2,2] = 1.

        camera_parameter.set_KRT(K, R, T)
        camera_parameter.set_dist_coeff(dist_coeff_k,dist_coeff_p)
        camera_parameter.inverse_extrinsic()
        camera_parameter.set_resolution(image_array.shape[1], image_array.shape[2]) # height, width

        # undistort cam and images
        corrected_cam, corrected_img = undistort_images(camera_parameter, image_array)
        K = np.asarray(corrected_cam.get_intrinsic())
        R = np.asarray(corrected_cam.get_extrinsic_r())
        T = np.asarray(corrected_cam.get_extrinsic_t())

        print(fullpose.shape, betas.shape, transl.shape, image_array.shape)

        print(estimator_config.smplify.body_model)

        # visualize smpl
        results = visualize_smpl_calibration(
            poses=fullpose,
            betas=betas,
            transl=transl,
            K=K,
            R=R,
            T=T,
            overwrite=True,
            body_model_config=estimator_config.smplify.body_model,
            output_path=os.path.join(args.output_dir,f'xr_smplx_view{view}/'),
            image_array=corrected_img,
            resolution=(corrected_img.shape[1], corrected_img.shape[2]),
            return_tensor=True,
            alpha=0.8,
            batch_size=5,
            plot_kps=False,
            vis_kp_index=False,
            verbose=True)


def get_image_array(background_dir, view):
    frame_dir = os.path.join(background_dir, view)
    n_frame = 0
    if frame_dir.startswith('s3://'):
        frame_list = list(client.list(frame_dir))
        n_frame = len(frame_list)
        frame_list.sort()
        abs_frame_list = [
            os.path.join(frame_dir, frame_name) for frame_name in frame_list
        ]
        image_list = []
        for image_path in abs_frame_list:
            img_bytes = client.get(image_path)
            img_mem_view = memoryview(img_bytes)
            img_array = np.frombuffer(img_mem_view, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            image_list.append(img)
        image_array = np.asarray(image_list)
    else:
        image_array = None
    return image_array, n_frame

def get_camera_param_from_annots(annots_file, view):
    if annots_file.startswith('s3://'):
        annots_file = io.BytesIO(client.get(annots_file))
    ren_body_cam_dict = np.load(
    annots_file, allow_pickle=True).item()['cams']
    
    camera_para_dict = {
        'RT': ren_body_cam_dict[view]['RT'].reshape(1, 4, 4),
        'K': ren_body_cam_dict[view]['K'].reshape(1, 3, 3),
        } if '00' in ren_body_cam_dict.keys() else \
        {
        'RT': ren_body_cam_dict['RT'][int(view)].reshape(1, 4, 4),
        'K': ren_body_cam_dict['K'][int(view)].reshape(1, 3, 3),
        }

    extrinsic = camera_para_dict['RT'][0, :, :]  # 4x4 mat
    intrinsic = camera_para_dict['K'][0, :, :]  # 3x3 mat
    r_mat_inv = extrinsic[:3, :3]
    r_mat = np.linalg.inv(r_mat_inv)
    t_vec = extrinsic[:3, 3:]
    t_vec = -np.dot(r_mat, t_vec).reshape((3))

    dist_array = ren_body_cam_dict[view]['D'] \
            if '00' in ren_body_cam_dict.keys() else \
            np.array(ren_body_cam_dict['D'][int(view)])
    dist_keys = ['k1', 'k2', 'p1', 'p2', 'k3']
    dist_coeff_k =[]
    dist_coeff_p = []

    distortion_coefficients = {}
    for dist_index, dist_key in enumerate(dist_keys):
        if 'k' in dist_key:
            dist_coeff_k.append(dist_array[dist_index])
        if 'p' in dist_key:
            dist_coeff_p.append(dist_array[dist_index])
        distortion_coefficients[dist_key] = float(dist_array[dist_index])

    return intrinsic, r_mat, t_vec, dist_coeff_k, dist_coeff_p, distortion_coefficients

if __name__ == '__main__':
    main()