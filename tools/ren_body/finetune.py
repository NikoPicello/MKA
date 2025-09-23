# yapf: disable
import sys
# sys.path.append('/home/hehonglin/nvme/renbody_release/zoehuman/xrmocap')
sys.path.append('/nvme/lufan/Projects/RenBody/Renbody_benchmark/keypoint_smpl/zoehuman')
sys.path.append('/nvme/lufan/Projects/RenBody/Renbody_benchmark/keypoint_smpl/zoehuman/tools')
sys.path.append('/nvme/lufan/Projects/RenBody/Renbody_benchmark/keypoint_smpl/zoehuman/xrmocap_v1/xrmocap')
import copy
import logging
import numpy as np
import torch
import mmcv
import cv2
from mmcv.runner import load_checkpoint
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
# sys.path.append('/home/hehonglin/nvme/renbody_release/zoehuman')
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

def parse_args():
    parser = argparse.ArgumentParser('SMPLify tools')
    # path
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--kps3d_file', type=str, required=True)
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
    parser.add_argument(
        '--mask_data_dir',
        type=str,
        help='Path to Mask directory.', default=None)


    # visualization
    parser.add_argument('--vis_smpl', action='store_true')
    parser.add_argument('--vis_kps', action='store_true')
    parser.add_argument('--save_mesh', action='store_true')
    parser.add_argument('--frame_interval', type=int, default=30)
    parser.add_argument('--view_idxs', nargs='+', 
        help='Indexes of views to be rendered')

    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--read_file', action='store_true')

    args = parser.parse_args()
    return args

def main():
    # get configs and update
    args = parse_args()

    logger = get_logger()

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
        

    ### 0. read images and cameras

    ### 1. estimate 2D kps and triangulate for kps3d
    # build mview sperson estimator
    # import pdb; pdb.set_trace()
    logger.info(estimator_config.smplify.body_model)
    mview_sp_smpl_estimator = build_estimator(dict(estimator_config))
    logger.info("Estimator built")

    if args.triangulate:
        logger.info("Using xrmocap triangulation, with cam selector")
        # load HumanData
        human_data_list = []
        annot = np.load(args.annots_file, allow_pickle=True).item()
        human_data_key_list = sorted(list(annot['cams'].keys())) \
            if args.cam_parameters_type == 'ren_body_0418' \
            else sorted(os.listdir(args.image_data_dir))
        
        for human_data_key in human_data_key_list:
            tmp_human_data_path = os.path.join(
                args.human_data_dir,
                f'{args.human_data_prefix}{human_data_key}.npz')
            if check_path_existence(tmp_human_data_path, 'file') != \
                    Existence.FileExist:
                raise FileNotFoundError(
                    f'HumanData file not found: {tmp_human_data_path}')
            human_data = HumanData.fromfile(tmp_human_data_path)
            # convert HumanData to Keypoints2D
            keypoints_src_mask = human_data['keypoints2d_mask']
            keypoints_src = human_data['keypoints2d'][..., :3]
            n_frame = keypoints_src.shape[0]
            keypoints_src_mask = np.repeat(keypoints_src_mask[np.newaxis,:], n_frame, axis=0)
            keypoints_src_mask = np.expand_dims(keypoints_src_mask,axis=1)
            keypoints_src = np.expand_dims(keypoints_src,axis=1)
            
            keypoints2d = Keypoints(            
                dtype='numpy',
                kps=keypoints_src,
                mask=keypoints_src_mask,
                convention=args.src_convention)

            human_data_list.append(keypoints2d)
        assert len(human_data_list) >= 2, 'HumanData fewer than 2!'


        # load Camera Parameters
        assert check_path_existence(args.annots_file, 'auto') == \
            Existence.FileExist

        cam_parameters_key_list = sorted(list(annot['cams'].keys())) \
            if args.cam_parameters_type == 'ren_body_0418' \
            else sorted(os.listdir(args.image_data_dir))
        camera_parameter_list = []
        for cam_key in cam_parameters_key_list:
            camera_parameter = FisheyeCameraParameter(name=cam_key)
            K, R, T, dist_coeff_k, dist_coeff_p, dist_coeff_dict = \
                get_camera_param_from_annots(args.annots_file, cam_key)
            camera_parameter.set_KRT(K, R, T)
            camera_parameter.set_dist_coeff(dist_coeff_k,dist_coeff_p)
            camera_parameter.inverse_extrinsic()
            camera_parameter_list.append(camera_parameter)

        assert len(camera_parameter_list) == len(human_data_list),\
            'numbers of cameras and HumanData do not match'

        logger.info(">>> triangulate")
        # triangulate
        keypoints3d = mview_sp_smpl_estimator.estimate_keypoints3d(
            camera_parameter_list,human_data_list)
        
        
        kps3d_file_mocap = args.kps3d_file
        kps3d_file_mocap.replace('optim','xrmocap')
        head, _ = os.path.split(kps3d_file_mocap)
        if not os.path.exists(head):
            os.makedirs(head)
        keypoints3d.dump(kps3d_file_mocap)
        logger.info(f"Saved to {kps3d_file_mocap}")

        # temp = dict(np.load(args.kps3d_file))
        # keypoints3d = Keypoints(            
        #         dtype='numpy',
        #         kps=temp['keypoints'],
        #         mask=temp['mask'],
        #         convention=str(temp['convention']))
        # keypoints3d = Keypoints()
        # import pdb; pdb.set_trace()
        # logger.info(args.kps3d_file)
        # keypoints3d.fromfile(args.kps3d_file)
        
    # read pose3d and convert to kps3d
    else:
        annot = np.load(args.annots_file, allow_pickle=True).item()
        cam_parameters_key_list = sorted(list(annot['cams'].keys())) \
            if args.cam_parameters_type == 'ren_body_0418' \
            else sorted(os.listdir(args.image_data_dir))
        # annot = np.load(args.annots_file, allow_pickle=True).item()
        camera_parameter_list = []
        for cam_key in cam_parameters_key_list:
            camera_parameter = FisheyeCameraParameter(name=cam_key)
            K, R, T, dist_coeff_k, dist_coeff_p, dist_coeff_dict = \
                get_camera_param_from_annots(args.annots_file, cam_key)
            camera_parameter.set_KRT(K, R, T)
            camera_parameter.set_dist_coeff(dist_coeff_k,dist_coeff_p)
            camera_parameter.inverse_extrinsic()
            camera_parameter_list.append(camera_parameter)
        logger.info(f"Loading keypoints 3D from old pipeline: {args.kps3d_file}")
        human_data = HumanData.fromfile(args.kps3d_file)
        keypoints_src_mask = human_data['keypoints3d_mask']
        keypoints_src = human_data['keypoints3d'][..., :3]
        n_frame = keypoints_src.shape[0]
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
            
        if mview_sp_smpl_estimator.kps3d_optimizers is not None:
            for optimizer in mview_sp_smpl_estimator.kps3d_optimizers:
                save_keypoints3d = optimizer.optimize_keypoints3d(
                    keypoints3d)
        else:
            save_keypoints3d = keypoints3d

        # Visualize and output in human_data format
        save_keypoints3d_file = {}
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
    
    ### 2. optimize kps3d/ camera selection
    # 3D keypoints optimization
    # import pdb; pdb.set_trace()
    if mview_sp_smpl_estimator.kps3d_optimizers is not None:
        for optimizer in mview_sp_smpl_estimator.kps3d_optimizers:
            convert_keypoints3d = optimizer.optimize_keypoints3d(
                convert_keypoints3d)

    temp_kps = convert_keypoints3d.get_keypoints().copy()
    temp_msk = convert_keypoints3d.get_mask().copy()
    # set extra keypoint mappings
    name_dic = {'left_foot': 'left_bigtoe',
                'right_foot': 'right_bigtoe' }
    for k, v in name_dic.items(): 
        k_idx = get_keypoint_idx(
                    k, convention=args.tgt_convention)
        v_idx = get_keypoint_idx(
                    v, convention=args.tgt_convention)
        temp_kps[:,:,k_idx,:] = temp_kps[:,:,v_idx,:]
        temp_msk[:,:,k_idx] = temp_msk[:,:,v_idx]
        logger.info(f"Set keypoint {k} {k_idx} with {v} {v_idx}")
    
    convert_keypoints3d.set_keypoints(temp_kps)
    convert_keypoints3d.set_mask(temp_msk)

    ### 3. fit smpl

    # read respective init smpl parameters for non apose sequence
    init_smpl_dict = {}
    if args.init_smpl_dir is not None :# and 'apose' not in args.sequence:
        # keys = ['betas'] # TODO only betas for now
        
        # name, yf = args.sequence.split("_")[-3:-1] # sequence format: name_yf1_dz1
        name = 'chekang' # '_'.join(args.sequence.split("_")[:-2])
        yf = 'yf1' # args.sequence.split("_")[-2]
        dz = 'dz1'
        smpl_output_dir = os.path.basename(args.output_dir)
        # stem, _ = osp.splitext(osp.basename(args.kps3d_file))
        # args.init_smpl_dir = os.path.join(args.init_smpl_dir,
        #     f'{name}_{yf}_{dz}', 'smpl','smplx_xrmocap',f'human_data_tri_smplx.npz')
        if not os.path.isfile(args.init_smpl_dir):
            e = f"apose not exist in {args.init_smpl_dir}"
            seq = ','.join(args.output_dir.split('/')[-4:-1])
            return False, seq, e
        
        pose_dict = dict(np.load(args.init_smpl_dir))
        fullpose = pose_dict['fullpose']
        n_frame = fullpose.shape[0]
        pose_dict['global_orient'] = fullpose[:, 0].reshape(n_frame, 3)
        pose_dict['body_pose'] = fullpose[:, 1:22].reshape(n_frame, 63)
        pose_dict['jaw_pose'] = fullpose[:, 22].reshape(n_frame, 3)
        pose_dict['leye_pose'] = fullpose[:, 23].reshape(n_frame, 3)
        pose_dict['reye_pose'] = fullpose[:, 24].reshape(n_frame, 3)
        pose_dict['left_hand_pose'] = fullpose[:, 25:40].reshape(n_frame, 45)
        pose_dict['right_hand_pose'] = fullpose[:, 40:55].reshape(n_frame, 45)
        keys = list(pose_dict.keys())
        keys.remove('gender')
        print(keys)
        for key in keys:
            init_smpl_dict[key] = torch.tensor(pose_dict[key])
            logger.info(f'Loading initial {key} from {args.init_smpl_dir}')
    
    # put the mask
    smpl_data, adhoc_data = mview_sp_smpl_estimator.estimate_smpl(
        keypoints3d = convert_keypoints3d, init_smpl_dict=init_smpl_dict, mask_dir=args.mask_data_dir, annots_file=args.annots_file, camera_params=camera_parameter_list,
        return_joints=True, return_verts=args.save_mesh)

    smpl_keypoints3d = Keypoints(            
            dtype='numpy',
            kps=adhoc_data['joints'][:, np.newaxis],
            convention=args.tgt_convention)

    smpl_data.set_gender(args.gender)
    logger.info("SMPLify done")

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
    stem, _ = osp.splitext(osp.basename(args.kps3d_file))
    npz_path = osp.join(output_dir_npz, f'{stem}_{args.model}.npz')
    save_smplx_path = osp.join(output_dir_npz, f'human_data_{args.model}.npz')

    joints_path = osp.join(output_dir_npz, f'{stem}_{args.model}_adhoc.npz')
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

    np.savez(npz_path, scale=args.scale, **smpl_data)
    np.savez(joints_path, scale=args.scale, **adhoc_data)
    np.savez(save_smplx_path, **save_smplx)
    logger.info(f"File saved as {npz_path} \n{joints_path}")
    
    # save mesh
    if args.save_mesh:
        output_dir_mesh = os.path.join(args.output_dir, '..', 'mesh')
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

    fullpose = smpl_data['fullpose'].reshape(n_frame, -1)
    transl = smpl_data['transl']
    betas = smpl_data['betas']

    body_model_config = dict(model_path='mmhuman3d/data/body_models')
    body_model_config['use_pca'] = False
    # body_model_config['use_face_contour'] = True
    # body_model_config['flat_hand_mean'] = True

    estimator_config.smplify.body_model.update(body_model_config)
    # import pdb; pdb.set_trace()
    logger.info(estimator_config.smplify.body_model)
    
    for view in args.view_idxs: # 10, 25
        logger.info(f'Visualization of view {view}')
        # load background
        image_array, n_frame = get_image_array(args.background_dir, view)
        if n_frame == 0:
            logger.error("Frame number is 0, please check the image directory")
            raise ValueError
        src_image = image_array

        # load camera
        camera_parameter = FisheyeCameraParameter(name=view)
        K, R, T, dist_coeff_k, dist_coeff_p, dist_coeff_dict = \
            get_camera_param_from_annots(args.annots_file, view)
        camera_parameter.set_KRT(K, R, T)
        camera_parameter.set_dist_coeff(dist_coeff_k,dist_coeff_p)
        camera_parameter.inverse_extrinsic()
        camera_parameter.set_resolution(image_array.shape[1], image_array.shape[2]) # height, width

        # undistort cam and images
        corrected_cam, corrected_img = undistort_images(camera_parameter, image_array)
        K = np.asarray(corrected_cam.get_intrinsic())
        R = np.asarray(corrected_cam.get_extrinsic_r())
        T = np.asarray(corrected_cam.get_extrinsic_t())

        # visualize smpl
        if args.vis_smpl:
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
                vis_kp_index=False)

            src_image = results

        # plot kps3d
        if args.vis_kps:
            frame_idxs = range(0, n_frame, args.frame_interval)
            annots_file = np.load(args.annots_file, allow_pickle=True).item()
            for idx in frame_idxs:
                results = plot_keypoints(annots_file, src_image, args.output_dir, idx, view)
                # try:
                #     results = plot_keypoints(args.annots_file, src_image, args.output_dir, idx, view)
                # except: 
                #     e = "Nan loss"
                #     seq = ','.join(args.output_dir.split('/')[-4:-1])
                #     return False, seq, e
        
            # smpl_regression_res = visualize_project_keypoints3d(
            #     keypoints=smpl_keypoints3d,
            #     cam_param=corrected_cam,
            #     output_path=os.path.join(args.output_dir,f'human_data_smpl_keypoints3d_view{view}.mp4'),
            #     img_arr=np.array(src_image),
            #     overwrite = True,
            #     return_array=True)
        
            # tri_res = visualize_project_keypoints3d(
            #     keypoints=convert_keypoints3d,
            #     cam_param=camera_parameter,
            #     output_path=os.path.join(args.output_dir,f'xr_kps3d_tri_view{view}.mp4'),
            #     img_arr=image_array,
            #     overwrite = True,
            #     return_array=True)

            save_res = visualize_project_keypoints3d(
                keypoints=save_keypoints3d,
                cam_param=corrected_cam,
                output_path=os.path.join(args.output_dir,f'human_data_optimized_keypoints3d_view{view}/'),
                img_arr=np.array(src_image),
                overwrite = True,
                return_array=True)
    return True, True, True
        
        
def get_camera_param_from_annots(annots_file, view):
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

def get_image_array(background_dir, view):
    frame_dir = os.path.join(background_dir, view)
    n_frame = 0
    if len(frame_dir) > 0 and \
        check_path_existence(frame_dir, 'dir') == \
        Existence.DirectoryExistNotEmpty:
        frame_list = os.listdir(frame_dir)
        n_frame = len(frame_list)
        frame_list.sort()
        abs_frame_list = [
            os.path.join(frame_dir, frame_name) for frame_name in frame_list
        ]
        image_list = [
            cv2.imread(filename=image_path) for image_path in abs_frame_list
        ]
        image_array = np.asarray(image_list)
    else:
        image_array = None
    return image_array, n_frame


if __name__ == '__main__':

    success, seq, e = main()
    if not success:
        print(f">>>Not success ({e}): {seq}")
        error_file = open(f'/mnt/cache/yinwanqi/01-project/zoehuman/data/data_list/fail_list.txt', "a")
        error_file.write(f'{seq},{e}\n')
        error_file.close()
