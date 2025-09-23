# yapf: disable
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

from tools.visualization.visualize_project_smplx import plot_keypoints
from zoehuman.utils.path_utils import Existence, check_path_existence
from xrmocap.core.visualization.visualize_keypoints3d import visualize_project_keypoints3d

from xrprimer.data_structure.camera import FisheyeCameraParameter
from zoehuman.core.visualization.visualize_smpl import visualize_smpl_distortion

from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes
from tqdm import tqdm
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
    
    # config
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


    # visualization
    parser.add_argument('--vis_smpl', action='store_true')
    parser.add_argument('--vis_kps', action='store_true')
    parser.add_argument('--save_mesh', action='store_true')
    parser.add_argument('--frame_interval', type=int, default=30)
    parser.add_argument('--view_idxs', nargs='+', 
        help='Indexes of views to be rendered')

    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--read_file', type=int, default=999)

    args = parser.parse_args()
    return args

def main(args, logger):
    logger = get_logger(logger)
    # get configs and update
    
    estimator_config = mmcv.Config.fromfile(args.config)
    estimator_config.smplify.body_model.update(dict(type = args.model.upper()))
    estimator_config.smplify.body_model.update(dict(gender = args.gender))
    
    # TODO update config according to annots and sequence type
    # if args.annots_file is not None:
    #     actor_info = np.load(args.annots_file, allow_pickle=True).item()['info']
    #     args.scale = 170 / actor_info['height'] if actor_info['gender'] == 'female' else 180 / actor_info['height']
    #     logger.info(f'update scale: {args.scale}')
    #     estimator_config.smplify.update(dict(scale=args.scale))

    body_model_type = ['smpl', 'smplx']
    smplify_type =  ['smplify', 'smplifyx']
    assert estimator_config.smplify.body_model.type.lower() in body_model_type
    assert estimator_config.smplify.type.lower() in smplify_type
    assert body_model_type.index(estimator_config.smplify.body_model.type.lower()) == \
        smplify_type.index(estimator_config.smplify.type.lower())

    # TODO
    ### 0. read images and cameras
    ### 1. estimate 2D kps and triangulate for kps3d
    
    # read pose3d and convert to kps3d
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
    
    if args.src_convention != args.tgt_convention:
        convert_keypoints3d = convert_keypoints(
                keypoints = keypoints3d,
                dst = args.tgt_convention,
                approximate = True
        )
    else:
        convert_keypoints3d = keypoints3d
    logger.info(f"Keypoints converted, current convention: {convert_keypoints3d.get_convention()}")
    
    # set extra keypoint mapptings
    name_dic = {'left_foot': 'left_bigtoe',
                'right_foot': 'right_bigtoe' }
    temp_kps = convert_keypoints3d.get_keypoints().copy()
    temp_msk = convert_keypoints3d.get_mask().copy()
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

    ### 2. optimize kps3d/ camera selection

    ### 3. fit smpl
    # build mview sperson estimator
    mview_sp_smpl_estimator = build_estimator(dict(estimator_config))
    logger.info("Estimator built")

    # read respective init smpl parameters for non apose sequence
    init_smpl_dict = {}
    if args.init_smpl_dir is not None and 'apose' not in args.sequence:
        keys = ['betas'] # TODO only betas for now
        
        # name, yf = args.sequence.split("_")[-3:-1] # sequence format: name_name_name_fy1_dz1
        name = '_'.join(args.sequence.split("_")[:-2])
        yf = args.sequence.split("_")[-2]
        smpl_output_dir = os.path.basename(args.output_dir)
        stem, _ = osp.splitext(osp.basename(args.kps3d_file))
        args.init_smpl_dir = os.path.join(args.init_smpl_dir,
            f'{name}_apose_{yf}',smpl_output_dir,f'{stem}_{args.model}.npz')
        
        if not os.path.isfile(args.init_smpl_dir):
            e = f"apose not exist in {args.init_smpl_dir}"
            return False, e

        apose_dict = dict(np.load(args.init_smpl_dir))
        for key in keys:
            init_smpl_dict[key] = torch.tensor(apose_dict[key])
            logger.info(f'Loading initial {key} from {args.init_smpl_dir}')
        

    smpl_data, adhoc_data = mview_sp_smpl_estimator.estimate_smpl(
        keypoints3d = convert_keypoints3d, init_smpl_dict=init_smpl_dict,
        return_joints=True, return_verts=args.save_mesh)

    smpl_keypoints3d = Keypoints(            
            dtype='numpy',
            kps=adhoc_data['joints'][:, np.newaxis],
            convention=args.tgt_convention)

    smpl_data.set_gender(args.gender)
    logger.info("SMPLify done")


    ### 4.save results # TODO: save the same file as old version
    output_dir_npz = args.output_dir
    os.makedirs(output_dir_npz, exist_ok=True)
    stem, _ = osp.splitext(osp.basename(args.kps3d_file))
    npz_path = osp.join(output_dir_npz, f'{stem}_{args.model}.npz')

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
    body_model_config['use_face_contour'] = True
    body_model_config['flat_hand_mean'] = True

    estimator_config.smplify.body_model.update(body_model_config)
    
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
        K = np.asarray(camera_parameter.get_intrinsic())
        R = np.asarray(camera_parameter.get_extrinsic_r())
        T = np.asarray(camera_parameter.get_extrinsic_t())

        # visualize smpl
        if args.vis_smpl:
            results = visualize_smpl_distortion(
                poses=fullpose,
                betas=betas,
                transl=transl,
                K=K,
                R=R,
                T=T,
                dist_coeffs=dist_coeff_dict,
                overwrite=True,
                body_model_config=estimator_config.smplify.body_model,
                output_path=os.path.join(args.output_dir,f'xr_smplx_view{view}.mp4'),
                image_array=image_array,
                resolution=(image_array.shape[1], image_array.shape[2]),
                return_tensor=True,
                plot_kps=False,
                vis_kp_index=False)
            
            # speed up, do not rectify images, kps might not match the image
            # results = visualize_smpl_calibration(
            #     poses=fullpose,
            #     betas=betas,
            #     transl=transl,
            #     K=K,
            #     R=R,
            #     T=T,
            #     overwrite=True,
            #     body_model_config=estimator_config.smplify.body_model,
            #     resolution=(image_array.shape[1], image_array.shape[2]),
            #     output_path=os.path.join(args.output_dir,f'xr_smplx_view{view}.mp4'),
            #     image_array=image_array,
            #     return_tensor=True,
            #     plot_kps=False,
            #     vis_kp_index=False)

            src_image = results

        # plot kps3d
        if args.vis_kps:
            frame_idxs = range(0, n_frame, args.frame_interval)
            for idx in frame_idxs:
                try:
                    results = plot_keypoints(args.annots_file, src_image, args.output_dir, idx, view)
                except: 
                    e = "Nan loss"
                    return False, e
        # plot kps3d xrmocap
        # smpl_regression_res = visualize_project_keypoints3d(
        #     keypoints=smpl_keypoints3d,
        #     cam_param=camera_parameter,
        #     output_path=os.path.join(args.output_dir,f'xr_kps3d_smplx_view{view}.mp4'),
        #     img_arr=image_array,
        #     overwrite = True,
        #     return_array=True)
        
        # tri_res = visualize_project_keypoints3d(
        #     keypoints=convert_keypoints3d,
        #     cam_param=camera_parameter,
        #     output_path=os.path.join(args.output_dir,f'xr_kps3d_tri_view{view}.mp4'),
        #     img_arr=image_array,
        #     overwrite = True,
        #     return_array=True)
        
    return True, "done"
        
        
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

def read_file(args, line):
    # format 20220929/duping_f/duping_yf2_dz8
    USER = 'yinwanqi'
    elements = line.split('/')
    date = elements[0]
    person_name = elements[1]
    gender_file = person_name[-1]
    sequence_elements = elements[2].strip('\n').split("_")
    dz_name = f'{person_name[:-2]}_{sequence_elements[-2]}_{sequence_elements[-1]}'

    if gender_file=='f':
        gender = 'female'
    elif gender_file=='m':
        gender = 'male'

    input_dir=f'/mnt/lustre/share_data/chengwei/RenBody/{date}/{person_name}'
    output_dir=f'/mnt/lustre/share_data/{USER}/zoehuman_ren_body/{date}/{person_name}/{dz_name}'
    output_dir_final=f'/mnt/lustre/share_data/{USER}/zoehuman_ren_body_sub200/{date}/{person_name}/{dz_name}'
    renbody_out_dir=f'/mnt/lustre/share_data/{USER}/RenBody_smpl/{date}/{person_name}/{dz_name}'
    betas_dir=f'/mnt/lustre/share_data/{USER}/zoehuman_ren_body/{date}/{person_name}'
    background_dir=f'{input_dir}/{dz_name}/image/'
    annots_file=f'{input_dir}/annots.npy'
    if 'apose' in dz_name:
        args.config = 'configs/smplify/mview_sperson_smplify_renbody.py'
    else:
        args.config = 'configs/smplify/mview_sperson_smplify_renbody_seq.py'

    args.kps3d_file = f'{output_dir}/pose_3d/optim/human_data_tri.npz'
    args.annots_file = annots_file
    args.output_dir = f'{output_dir_final}/smplx_xrmocap'
    args.init_smpl_dir = betas_dir
    args.background_dir = background_dir
    args.gender = gender
    args.sequence = dz_name

    return args



if __name__ == '__main__':

    # file = open('/mnt/cache/yinwanqi/RenBody_random_sets_new/random_set_200.txt','r')
    file = open('/mnt/cache/yinwanqi/01-project/zoehuman/data/data_list/sub2_redo_3.txt','r')

    lines = file.readlines()
    logger = get_logger()
    args = parse_args()

    if args.read_file != 999:
        total_id = len(lines)
        id_range = int(total_id/4)
        min_id = int(args.read_file * id_range)
        max_id = int((args.read_file+1) * id_range)
        
        lines = lines[min_id: max_id]
        logger.info(f">>>Process [{min_id},{max_id}], total {len(lines)} ids")
    else: 
        min_id = 0
        max_id = len(lines)
        logger.info(f">>>Process [{min_id},{max_id}], total {len(lines)} ids")
    
    for id, line in enumerate(lines):
        logger.info(f">>>Subset Id: {min_id+id}")
        logger.info(">>>Update args from file")
        args = read_file(args, line)
        logger.info(args)
        if not os.path.isfile(args.kps3d_file):
            logger.info(f">>>Not exist (no 3dkps file): {args.kps3d_file}")
            error_file = open(f'/mnt/cache/yinwanqi/RenBody_random_sets/200_not_exist_{str(args.read_file)}.txt', "a")
            error_file.write(line)
            error_file.close()
            continue
        success, e = main(args, logger)
        if not success:
            logger.info(f">>>Not success ({e}): {line}")
            error_file = open(f'/mnt/cache/yinwanqi/RenBody_random_sets/200_not_success_{str(args.read_file)}.txt', "a")
            error_file.write(line + ":" + e + "\n")
            # error_file.write(e)
        
    # main()
