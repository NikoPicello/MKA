# yapf: disable

import argparse
import os
import os.path as osp
import json 
import copy
import logging
import numpy as np
import mmcv
import cv2
import io
import sys
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
# from mmcv.runner import load_checkpoint
from tqdm import tqdm
from typing import List, Tuple, Union, overload

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path, "dependencies"))
sys.path.append(os.path.join(root_path, 'tools'))
sys.path.append(os.path.join(root_path, "dependencies", 'xrmocap'))

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
from xrmocap.transform.convention.keypoints_convention import convert_keypoints, get_keypoint_idx
from xrmocap.utils.geometry import get_scale
from xrmocap.utils.mvp_utils import norm2absolute, process_dict
from xrmocap.core.estimation.builder import build_estimator
from xrmocap.core.visualization.visualize_keypoints3d import visualize_project_keypoints3d

from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.core.visualization.visualize_smpl import \
    visualize_smpl_calibration
# from tools.visualization.visualize_project_smplx import plot_keypoints
from visualization.visualize_project_smplx import plot_keypoints_h36m

from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrprimer.transform.camera.distortion import undistort_images
from xrprimer.utils.log_utils import get_logger

from zoehuman.core.visualization.visualize_smpl import visualize_smpl_distortion
from zoehuman.utils.path_utils import Existence, check_path_existence

from pytorch3d.io import IO
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes

def parse_args():
    parser = argparse.ArgumentParser('SMPLify tools')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--kps3d_file', type=str, required=True)
    parser.add_argument('--cam_file', type=str, required=True)
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--init_smpl_file', type=str, default=None)
    # smpl config
    parser.add_argument('--src_convention', type=str, default='smplx')
    parser.add_argument('--tgt_convention', type=str, default='smplx')

    parser.add_argument(
        '--model',
        type=str,
        default='smplx',
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

    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--read_file', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--start_t', type=int, default=0)
    parser.add_argument('--end_t', type=int, default=-1)
    parser.add_argument('--output_finetune', action='store_true')

    args = parser.parse_args()
    return args

def main():
    # get configs and update
    args = parse_args()        
    logger = get_logger()

    print("==== loading parameters", flush=True)
    estimator_config = mmcv.Config.fromfile(args.config)
    estimator_config.smplify.body_model.update(dict(type = args.model.upper()))
    estimator_config.smplify.body_model.update(dict(gender = args.gender))

    body_model_type = ['smpl', 'smplx']
    smplify_type =  ['smplify', 'smplifyx']
    print("==== loading estimator", flush=True)
    assert estimator_config.smplify.body_model.type.lower() in body_model_type
    assert estimator_config.smplify.type.lower() in smplify_type
    assert body_model_type.index(estimator_config.smplify.body_model.type.lower()) == \
        smplify_type.index(estimator_config.smplify.type.lower())
    
    ### 0. read images and cameras

    ### 1. estimate 2D kps and triangulate for kps3d
    # build mview sperson estimator
    # import pdb; pdb.set_trace()
    logger.info(estimator_config.smplify.body_model)
    mview_sp_smpl_estimator = build_estimator(dict(estimator_config))
    logger.info("Estimator built")
    
    logger.info(f"Loading keypoints 3D from old pipeline: {args.kps3d_file}")
    kps3d_file = args.kps3d_file
    if kps3d_file.startswith('s3://'):
        kps3d_file = io.BytesIO(client.get(kps3d_file))
    human_data = HumanData.fromfile(kps3d_file)

    frame_num, joint_num, _ = human_data["keypoints3d"].shape
    start_t = max(0, args.start_t)
    end_t = min(args.end_t, frame_num) if args.end_t > 0 else frame_num 
    
    # keypoints_src = human_data['keypoints3d'][..., :3]
    if args.finetune:
        keypoints_src_mask_raw = human_data['keypoints3d_mask']
        keypoints_src_raw = human_data['keypoints3d'][..., :3]
        # end_t = keypoints_src_raw.shape[0]
        keypoints_src = keypoints_src_raw[start_t:end_t] # #[186:205] # [69:84]
        n_frame = keypoints_src_raw.shape[0]
        keypoints_src_mask_raw = np.repeat(keypoints_src_mask_raw[np.newaxis,:], n_frame, axis=0)
        keypoints_src_mask_raw = np.expand_dims(keypoints_src_mask_raw, axis=1)
        keypoints_src_raw = np.expand_dims(keypoints_src_raw, axis=1)
    else:
        keypoints_src = human_data['keypoints3d'][start_t:end_t, :, :3]
        keypoints_src_mask = human_data['keypoints3d_mask']
    
    n_frame = keypoints_src.shape[0]
    keypoints_src_mask = human_data['keypoints3d_mask']
    keypoints_src_mask = np.repeat(keypoints_src_mask[np.newaxis,:], n_frame, axis=0)
    keypoints_src_mask = np.expand_dims(keypoints_src_mask,axis=1)
    keypoints_src = np.expand_dims(keypoints_src,axis=1)
    if not args.tgt_convention == estimator_config.smplify.body_model['keypoint_convention']:
        args.tgt_convention = estimator_config.smplify.body_model['keypoint_convention']
        logger.warning(f"Overwrite tgt_convention with f{estimator_config.smplify.body_model['keypoint_convention']}")

    keypoints3d = Keypoints(            
            dtype='numpy',
            kps=keypoints_src,
            mask=keypoints_src_mask,
            convention=args.src_convention)
    
    if args.finetune:
        save_keypoints3d_raw = Keypoints(            
            dtype='numpy',
            kps=keypoints_src_raw,
            mask=keypoints_src_mask_raw,
            convention=args.src_convention)
        
    if mview_sp_smpl_estimator.kps3d_optimizers is not None:
        for optimizer in mview_sp_smpl_estimator.kps3d_optimizers:
            save_keypoints3d = optimizer.optimize_keypoints3d(
                keypoints3d)
    else:
        save_keypoints3d = keypoints3d

    # Visualize and output in human_data format
    save_keypoints3d_file = {}
    if args.finetune:
        save_keypoints3d_file['convention'] = save_keypoints3d_raw.get_convention()
        save_keypoints3d_file['keypoints3d'] = save_keypoints3d_raw.get_keypoints().squeeze()
        save_keypoints3d_file['keypoints3d_mask'] = save_keypoints3d_raw.get_mask().squeeze()
    else:
        save_keypoints3d_file['convention'] = save_keypoints3d.get_convention()
        save_keypoints3d_file['keypoints3d'] = save_keypoints3d.get_keypoints().squeeze()
        save_keypoints3d_file['keypoints3d_mask'] = save_keypoints3d.get_mask().squeeze()

    output_dir_npz = args.output_dir
    os.makedirs(output_dir_npz, exist_ok=True)
    kps_path = osp.join(output_dir_npz, f'human_data_optimized_keypoints3d.npz')
    keypoints_path = osp.join(output_dir_npz, f'human_data_optimized_xr_keypoints3d.npz')
    np.savez(kps_path, **save_keypoints3d_file)
    save_keypoints3d.dump(keypoints_path)
    logger.info(f'optimized keypoints with convention {save_keypoints3d.get_convention()} '
        f'saved to {kps_path}.')

    if args.src_convention != args.tgt_convention:
        convert_keypoints3d = convert_keypoints(
                keypoints = keypoints3d,
                dst = args.tgt_convention,
                approximate = True
        )
    else:
        convert_keypoints3d = keypoints3d
    logger.info(f"Keypoints converted, current convention: {convert_keypoints3d.get_convention()}")
    # print(convert_keypoints3d['keypoints'])
    
    ### 2. optimize kps3d/ camera selection
    # 3D keypoints optimization
    # import pdb; pdb.set_trace()
    if mview_sp_smpl_estimator.kps3d_optimizers is not None:
        for optimizer in mview_sp_smpl_estimator.kps3d_optimizers:
            convert_keypoints3d = optimizer.optimize_keypoints3d(
                convert_keypoints3d)

    ### 3. fit smpl
    # read respective init smpl parameters for non apose sequence
    init_smpl_dict = {}
    if args.init_smpl_file is not None:
        keys = ['betas'] # TODO only betas for now
        if args.init_smpl_file.startswith('s3://'):
            init_smpl_f = io.BytesIO(client.get(init_smpl_file))
            apose_dict = dict(np.load(init_smpl_f))
            init_smpl_f.close()
        else:
            apose_dict = dict(np.load(args.init_smpl_file, allow_pickle=True))
        
        apose_dict_init = apose_dict.copy()
        print("==== init keys", apose_dict_init.keys(), flush=True)
        if 'fullpose' in apose_dict_init.keys():            
            if args.finetune:
                apose_dict_init['fullpose'] = apose_dict_init['fullpose'][start_t:end_t]
                apose_dict_init['transl'] = apose_dict_init['transl'][start_t:end_t]
                apose_dict_init['expression'] = apose_dict_init['expression'][start_t:end_t]
                if apose_dict_init['betas'].shape[0] != 1:
                    apose_dict_init['betas'] = apose_dict_init['betas'][start_t:end_t]

            fullpose = apose_dict_init['fullpose']
            n_frame = fullpose.shape[0]
            apose_dict_init['global_orient'] = fullpose[:, 0].reshape(n_frame, 3)
            apose_dict_init['body_pose'] = fullpose[:, 1:22].reshape(n_frame, 63)
            apose_dict_init['jaw_pose'] = fullpose[:, 22].reshape(n_frame, 3)
            apose_dict_init['leye_pose'] = fullpose[:, 23].reshape(n_frame, 3)
            apose_dict_init['reye_pose'] = fullpose[:, 24].reshape(n_frame, 3)
            apose_dict_init['left_hand_pose'] = fullpose[:, 25:40].reshape(n_frame, 45)
            apose_dict_init['right_hand_pose'] = fullpose[:, 40:55].reshape(n_frame, 45)

        for key in keys:
            # if key == 'gender':
            #     continue
            init_smpl_dict[key] = torch.tensor(apose_dict_init[key])
            logger.info(f'Loading initial {key} from {args.init_smpl_file}')
    
    smpl_data, adhoc_data = mview_sp_smpl_estimator.estimate_smpl(
        keypoints3d = convert_keypoints3d, init_smpl_dict=init_smpl_dict,
        return_joints=True, return_verts=args.save_mesh)
    
    smpl_keypoints3d = Keypoints(            
            dtype='numpy',
            kps=adhoc_data['joints'][:, np.newaxis],
            convention=args.tgt_convention)

    smpl_data.set_gender(args.gender)
    logger.info("SMPLify done")

    if args.finetune and args.init_smpl_file is not None:
        smpl_data_after = smpl_data.copy()
        smpl_data = apose_dict.copy()
        smpl_data['fullpose'][start_t:end_t] = smpl_data_after['fullpose']
        smpl_data['transl'][start_t:end_t] = smpl_data_after['transl']
        smpl_data['expression'][start_t:end_t] = smpl_data_after['expression']
        # smpl_data['betas'][start_t:end_t] = smpl_data_after['betas']

    save_smplx = smpl_data.copy()
    fullpose = save_smplx.pop('fullpose')
    n_frame = fullpose.shape[0]

    save_smplx['global_orient'] = fullpose[:, 0].reshape(n_frame, 3)
    save_smplx['body_pose'] = fullpose[:, 1:22].reshape(n_frame, 63)
    save_smplx['jaw_pose'] = fullpose[:, 22].reshape(n_frame, 3)
    save_smplx['leye_pose'] = fullpose[:, 23].reshape(n_frame, 3)
    save_smplx['reye_pose'] = fullpose[:, 24].reshape(n_frame, 3)
    save_smplx['left_hand_pose'] = fullpose[:, 25:40].reshape(n_frame, 45)
    save_smplx['right_hand_pose'] = fullpose[:, 40:55].reshape(n_frame, 45)

    ### 4.save results # TODO: save the same file as old version
    output_dir_npz = args.output_dir
    os.makedirs(output_dir_npz, exist_ok=True)
    # stem, _ = osp.splitext(osp.basename(args.kps3d_file))
    npz_path = osp.join(output_dir_npz, f'human_data_tri_{args.model}.npz')
    save_smplx_path = osp.join(output_dir_npz, f'human_data_{args.model}.npz')

    joints_path = osp.join(output_dir_npz, f'human_data_tri_{args.model}_adhoc.npz')
    for key, value in adhoc_data.items():
        if isinstance(value, torch.Tensor):
            adhoc_data[key] = value.detach().cpu().numpy()
    
    if args.save_mesh:
        face = mview_sp_smpl_estimator.smplify.body_model.faces
        if isinstance(face, np.ndarray):
            face = torch.from_numpy(face.astype(np.int32))
        vertices = torch.tensor(adhoc_data['vertices']).clone()
        adhoc_data.pop('vertices', None)

    adhoc_data['keypoints3d'] = convert_keypoints3d.get_keypoints().squeeze()
    adhoc_data['keypoints3d_mask'] = convert_keypoints3d.get_mask().squeeze()

    if args.finetune and args.init_smpl_file is not None:
        adhoc_data_after = adhoc_data.copy()
        if args.init_smpl_file.startswith('s3://'):
            adhoc_data = io.BytesIO(client.get(init_smpl_file))
            adhoc_data = dict(np.load(adhoc_data))
        else:
            adhoc_data = dict(np.load(init_smpl_file, allow_pickle=True))
        
        adhoc_data['joints'][start_t:end_t] = adhoc_data_after['joints']
        adhoc_data['keypoints3d'][start_t:end_t] = adhoc_data_after['keypoints3d']
        adhoc_data['keypoints3d_mask'][start_t:end_t] = adhoc_data_after['keypoints3d_mask']

    # for key in adhoc_data.keys():
    #     print(key, adhoc_data[key].shape)
    if 'scale' in smpl_data.keys():
        smpl_data.pop('scale')
    if 'scale' in adhoc_data.keys():
        adhoc_data.pop('scale')
    np.savez(npz_path, scale=args.scale, **smpl_data)
    np.savez(joints_path, scale=args.scale, **adhoc_data)
    np.savez(save_smplx_path, **save_smplx)
    logger.info(f"File saved as {npz_path} \n{joints_path}")
    
    # save mesh
    if args.save_mesh:
        output_dir_mesh = os.path.join(args.output_dir, 'mesh')
        os.makedirs(output_dir_mesh, exist_ok=True)

        for frame_index in tqdm(range(vertices.shape[0])):
            #  vertices.shape: [1, 10475, 3]
            #  faces.shape: [20908, 3]
            meshes = Meshes(
                verts=vertices[frame_index:frame_index + 1, :, :],
                faces=face.view(1, -1, 3),
                textures=TexturesVertex(
                    verts_features=torch.FloatTensor((
                        1, 1, 1)).view(1, 1, 3).repeat(
                        1, vertices.shape[-2], 1)))
            mesh_path = os.path.join(output_dir_mesh,
                                     f'{frame_index:06d}.obj')
            IO().save_mesh(data=meshes, path=mesh_path)

        logger.info(f"Mesh saved as {output_dir_mesh}")

    # TODO 
    ### 5.visualization
    # output_dir = os.path.dirname(args.output_video_path)
    # results_path = os.path.join(output_dir, f'smpl_vis_results_view{view}.npz')
    # np.savez(results_path, results=results)
    # results = np.load(results_path)['results']

    n_frame = smpl_data['fullpose'].shape[0]
    fullpose = smpl_data['fullpose'].reshape(n_frame, -1)
    transl = smpl_data['transl']
    betas = smpl_data['betas']

    body_model_config = dict(model_path='human_models/human_model_files')
    body_model_config['use_pca'] = False
    # body_model_config['use_face_contour'] = True
    # body_model_config['flat_hand_mean'] = True

    estimator_config.smplify.body_model.update(body_model_config)
    # import pdb; pdb.set_trace()
    logger.info(estimator_config.smplify.body_model)

    if args.finetune and args.output_finetune:
        fullpose = fullpose[start_t:end_t]
        transl = transl[start_t:end_t]
        if betas.shape[0] != 1:
            betas = betas[start_t:end_t]
        n_frame = end_t - start_t
    
    render_cam_len = 2
    cam_dict_list = []
    with open(args.cam_file, 'r') as fin:
        cam_dict = json.load(fin)
    for video_fn, v_dict in cam_dict.items():
        R = np.array(v_dict["R"], dtype=np.float32)
        T = np.array(v_dict["T"], dtype=np.float32)
        K = np.array(v_dict["K"], dtype=np.float32)

        video_path = os.path.join(args.video_dir, video_fn)
        cam_dict_list.append({
            "video_path": video_path,
            "K": K,
            "R": R,
            "T": T ,
        })

        if len(cam_dict_list) == render_cam_len:
            break 

    # vis_reso = len(cam_dict_list)
    vis_reso = 1
    for ci, cam_dict in enumerate(cam_dict_list):
        cam_str = "cam%03d" % ci
        video_path = cam_dict["video_path"]
        logger.info(f'Visualization of view {video_path}')
        # load background
        image_array, n_frame = get_image_array(video_path)
        H = image_array.shape[1]
        W = image_array.shape[2]

        image_array_th = torch.from_numpy(image_array).float()
        image_array_th = image_array_th.permute(0, 3, 1, 2) / 255.
        image_array_th = F.interpolate(image_array_th, (H // vis_reso, W // vis_reso), mode='bilinear')
        image_array_th = image_array_th.permute(0, 2, 3, 1) * 255.
        image_array = image_array_th.numpy().astype(np.uint8)

        image_array = image_array[start_t:end_t]
        n_frame = end_t - start_t

        # if args.finetune:
        #     image_array = image_array[start_t:end_t]
        #     n_frame = end_t - start_t
        if n_frame == 0:
            logger.error("Frame number is 0, please check the image directory")
            raise ValueError
        src_image = image_array

        K = cam_dict["K"]
        K = K / vis_reso
        K[2,2] = 1.

        R = cam_dict["R"]
        T = cam_dict["T"]
        # print("===== old_camera", flush=True)
        camera_parameter = FisheyeCameraParameter(name=cam_str)
        camera_parameter.set_KRT(K, R, T)
        camera_parameter.set_resolution(image_array.shape[1], image_array.shape[2]) # height, width
        
        RT = np.eye(4, dtype=np.float32)
        RT[:3, :3] = R.copy()
        RT[:3, 3] = T.copy()
        inv_RT = np.linalg.inv(RT)
        
        # visualize smpl
        if args.vis_smpl:
            print("==== visualize smplx", flush=True)
            results = visualize_smpl_calibration(
                poses=fullpose,
                betas=betas,
                transl=transl,
                K=K,
                R=inv_RT[:3, :3],
                T=inv_RT[:3, 3],
                overwrite=True,
                body_model_config=estimator_config.smplify.body_model,
                output_path=os.path.join(args.output_dir,f'xr_smplx_{cam_str}/'),
                image_array=image_array,
                resolution=(H, W),
                return_tensor=True,
                alpha=0.8,
                batch_size=1,
                plot_kps=False,
                vis_kp_index=False,
                verbose=True)

            src_image = results

        # plot kps3d
        if args.vis_kps:
            out_kpt3d_dir = os.path.join(args.output_dir,f'kpt3d_{cam_str}')
            os.makedirs(out_kpt3d_dir, exist_ok=True)
            frame_idxs = range(0, n_frame, args.frame_interval)
            for idx in frame_idxs:
                img = plot_keypoints_h36m(cam_dict, src_image, adhoc_data, idx, cam_str)
                out_img_path = os.path.join(out_kpt3d_dir, "%06d.png" % idx)
                cv2.imwrite(out_img_path, img)

            if args.finetune and args.output_finetune:
                save_keypoints3d_raw = save_keypoints3d

            if args.finetune:
                save_res = visualize_project_keypoints3d(
                    keypoints=save_keypoints3d_raw,
                    cam_param=camera_parameter,
                    output_path=os.path.join(args.output_dir,f'human_data_optimized_keypoints3d_{cam_str}/'),
                    img_arr=np.array(src_image),
                    overwrite = True,
                    return_array=True)
            else:
                save_res = visualize_project_keypoints3d(
                    keypoints=save_keypoints3d,
                    cam_param=camera_parameter,
                    output_path=os.path.join(args.output_dir,f'human_data_optimized_keypoints3d_{cam_str}/'),
                    img_arr=np.array(src_image),
                    overwrite = True,
                    return_array=True)

    return True, True, True



def get_image_array(video_path, to_rgb=False):
    out_img_arr = []
    n_frame = 0
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break 
        if to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_img_arr.append(frame)   
        n_frame += 1
    
    cap.release()
    return np.stack(out_img_arr, axis=0), n_frame


if __name__ == '__main__':
    print("=== start", flush=True)
    success, seq, e = main()