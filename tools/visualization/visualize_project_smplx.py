import argparse
import os

import cv2
import numpy as np
# TODO: temp
import torch
from tqdm import tqdm

from zoehuman.core.cameras.camera_parameters import CameraParameter
from zoehuman.core.visualization.visualize_smpl import \
    visualize_smpl_distortion  # prevent yapf isort conflict
from zoehuman.utils import str_or_empty
from zoehuman.utils.path_utils import Existence, check_path_existence

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.transform.convention.keypoints_convention import convert_keypoints

from mmhuman3d.data.data_structures.human_data import HumanData
# TODO: temp
def undistort(intrinsic: np.ndarray, width: int, height: int,
              dist_coeffs: dict):
    dist_coeff_list = [
        dist_coeffs.get('k1', 0.0),
        dist_coeffs.get('k2', 0.0),
        dist_coeffs.get('p1', 0.0),
        dist_coeffs.get('p2', 0.0),
        dist_coeffs.get('k3', 0.0),
        dist_coeffs.get('k4', 0.0),
        dist_coeffs.get('k5', 0.0),
        dist_coeffs.get('k6', 0.0),
    ]
    resolution_wh = np.array([width, height])
    corrected_intrinsic, _ = cv2.getOptimalNewCameraMatrix(
        intrinsic, np.array(dist_coeff_list), resolution_wh, 0, resolution_wh)
    return corrected_intrinsic


def perspective_projection(points, rotation, translation, fx, fy,
                           camera_center):
    """This function computes the perspective projection of a set of points.

    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


def plot_keypoints_h36m(cam_dict, img_arr, adhoc_data, idx, cam_str):
    # draw 2D keypoints for one image
    if isinstance(img_arr[idx], torch.Tensor):
        image = img_arr[idx].detach().cpu().numpy()
    else:
        image = img_arr[idx]
    
    # adhoc_data_path = "/mnt/petrelfs/luohuiwen/IMU/SMPLest-X/outputs_human36m_mv/kpt3d/s_09_act_02_subact_01/optim_kpt3d.npz"
    # adhoc_data = HumanData.fromfile(adhoc_data_path)

    kps3d = adhoc_data['keypoints3d'] 
    kps3d_mask = adhoc_data['keypoints3d_mask']
    joints = adhoc_data['joints']
    scale = adhoc_data['scale'] if hasattr(adhoc_data, 'scale') else 1.0
    
    K = cam_dict["K"]
    R = cam_dict["R"]
    T = cam_dict["T"]
    RT = np.eye(4, dtype=np.float32)
    RT[:3, :3] = R.copy()
    RT[:3, 3] = T.copy()
    camera_para_dict = {
        "RT": RT.reshape(1, 4, 4),
        "K": K.reshape(1, 3, 3),
    }

    kps2d = project_kps3d(camera_para_dict, K, kps3d, inv_ext=False)
    joints_2d = project_kps3d(camera_para_dict, K, joints*scale, inv_ext=False)

    kps2d = kps2d[idx]
    kps3d_mask = kps3d_mask[idx].squeeze()
    joints_2d = joints_2d[idx]
    for i, kp2d_plot in enumerate(kps2d):
        # if i in [0, 3, 6, 12]: # pelvis, spine1, spine2, neck
        x, y = kp2d_plot
        kp2d_plot = int(x), int(y)
        cv2.circle(image, kp2d_plot, radius=4, color=(0, 0, 255), thickness=-1)

        # x, y = joints_2d[i]
        # joint_plot = int(x), int(y)
        # if kps3d_mask[i] > 0.:
        #     cv2.circle(image, kp2d_plot, radius=4, color=(0, 255, 0), thickness=2) # green
        #     cv2.circle(image, joint_plot, radius=4, color=(255, 0, 0), thickness=2) # blue
        #     cv2.line(image, kp2d_plot, joint_plot, color=(0, 0, 255), thickness=1) # red
        # else:
        #     cv2.circle(image, joint_plot, radius=2, color=(255, 0, 0), thickness=2)
            
    # cv2.imwrite(os.path.join(output_dir, f'smplify_{cam_str}_{idx}.jpg'), image)
    return image


def plot_keypoints_record(cam_dict, results, output_path, idx, view):
    output_dir = output_path

    # draw 2D keypoints for one image
    if isinstance(results[idx], torch.Tensor):
        image = results[idx].detach().cpu().numpy()
    else:
        image = results[idx]

    # obtain 3D keypoints
    adhoc_data_path = os.path.join(output_dir, 'human_data_tri_smplx_adhoc.npz')
    adhoc_data = dict(np.load(adhoc_data_path))

    kps3d = adhoc_data['keypoints3d']
    kps3d_mask = adhoc_data['keypoints3d_mask']
    joints = adhoc_data['joints']
    scale = adhoc_data['scale'] if hasattr(adhoc_data, 'scale') else 1.0
    
    K = cam_dict["K"]
    R = cam_dict["R"]
    T = cam_dict["T"]
    RT = np.eye(4, dtype=np.float32)
    RT[:3, :3] = R.copy()
    RT[:3, 3] = T.copy()
    camera_para_dict = {
        "RT": RT.reshape(1, 4, 4),
        "K": K.reshape(1, 3, 3),
    }
    kps2d = project_kps3d(camera_para_dict, K, kps3d, inv_ext=False)
    joints_2d = project_kps3d(camera_para_dict, K, joints*scale, inv_ext=False)

    kps2d = kps2d[idx]
    kps3d_mask = kps3d_mask[idx].squeeze()
    joints_2d = joints_2d[idx]
    for i, kp2d_plot in enumerate(kps2d):
        # if i in [0, 3, 6, 12]: # pelvis, spine1, spine2, neck
        x, y = kp2d_plot
        kp2d_plot = int(x), int(y)
        x, y = joints_2d[i]
        joint_plot = int(x), int(y)
        if kps3d_mask[i] > 0.:
            cv2.circle(image, kp2d_plot, radius=4, color=(0, 255, 0), thickness=2) # green
            cv2.circle(image, joint_plot, radius=4, color=(255, 0, 0), thickness=2) # blue
            cv2.line(image, kp2d_plot, joint_plot, color=(0, 0, 255), thickness=1) # red
        else:
            cv2.circle(image, joint_plot, radius=2, color=(255, 0, 0), thickness=2)
            
    # cv2.imwrite(os.path.join(output_dir, f'smplify_view{view}_{idx}.jpg'), image)
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    return results


# TODO: temp
def plot_keypoints(annots_file, results, output_path, idx, view):
    output_dir = output_path

    # draw 2D keypoints for one image
    if isinstance(results[idx], torch.Tensor):
        image = results[idx].detach().cpu().numpy()
    else:
        image = results[idx]

    # obtain 3D keypoints
    # kps3d_human_data_path = os.path.join(output_dir, '../pose_3d', 'optim',
    #                                     'human_data_tri.npz')
    # kps3d = np.load(kps3d_human_data_path)['keypoints3d']
    # kps3d_mask = np.load(kps3d_human_data_path)['keypoints3d_mask']

    # read pose3d and convert to kps3d
    # import pdb; pdb.set_trace()
    adhoc_data_path = os.path.join(output_dir, 'human_data_tri_smplx_adhoc.npz')
    adhoc_data = dict(np.load(adhoc_data_path))

    kps3d = adhoc_data['keypoints3d']
    kps3d_mask = adhoc_data['keypoints3d_mask']
    joints = adhoc_data['joints']
    scale = adhoc_data['scale'] if hasattr(adhoc_data, 'scale') else 1.0

    # obtain camera parameters
    # annots_path = '/home/caizhongang/github/zoehuman/' + \
    #     'data_temp/renbody/sensebee_datalist_80636/annots.npy'
    # ren_body_cam_dict = np.load(annots_file, allow_pickle=True).item()['cams']
    ren_body_cam_dict = annots_file['cams']
    camera_parameter = CameraParameter(name=view)
    camera_para_dict = {
        'RT': ren_body_cam_dict[view]['RT'].reshape(1, 4, 4),
        'K': ren_body_cam_dict[view]['K'].reshape(1, 3, 3),
    }
    camera_parameter.load_from_lightstage(camera_para_dict, 0)
    dist_array = ren_body_cam_dict[view]['D']
    dist_keys = ['k1', 'k2', 'p1', 'p2', 'k3']
    distortion_coefficients = {}
    for dist_index, dist_key in enumerate(dist_keys):
        camera_parameter.set_value(dist_key, float(dist_array[dist_index]))
        distortion_coefficients[dist_key] = float(dist_array[dist_index])
    # import pdb; pdb.set_trace()
    corrected_intrinsic = undistort(
        intrinsic=camera_para_dict['K'][0],
        width=image.shape[1],
        height=image.shape[0],
        dist_coeffs=distortion_coefficients)
    # RT = np.linalg.inv(camera_para_dict['RT'][0])

    # # project kps3d
    # N, K, _ = kps3d.shape
    # kps3d = kps3d[..., :3].reshape(-1, 3)
    # xyz1 = np.concatenate([kps3d, np.ones([kps3d.shape[0], 1])],
    #                       axis=-1)  # (n, 4)
    # xyz1 = xyz1.T  # (4, n)
    # xyz1_cam = RT @ xyz1
    # xyz = xyz1_cam[:3, :]  # (3, n)
    # kps2d = corrected_intrinsic @ xyz
    # kps2d = kps2d.T  # (n, 3)
    # kps2d = kps2d[:, :2] / kps2d[:, 2, None]
    # kps2d = kps2d.reshape(N, K, 2)
    # print(f'kps2d shape: {kps2d.shape}')
    # import pdb; pdb.set_trace()
    kps2d = project_kps3d(camera_para_dict, corrected_intrinsic, kps3d)
    joints_2d = project_kps3d(camera_para_dict, corrected_intrinsic, joints*scale)

    kps2d = kps2d[idx]
    kps3d_mask = kps3d_mask[idx].squeeze()
    joints_2d = joints_2d[idx]
    for i, kp2d_plot in enumerate(kps2d):
        # if i in [0, 3, 6, 12]: # pelvis, spine1, spine2, neck
        x, y = kp2d_plot
        kp2d_plot = int(x), int(y)
        x, y = joints_2d[i]
        joint_plot = int(x), int(y)
        if kps3d_mask[i] > 0.:
            cv2.circle(image, kp2d_plot, radius=4, color=(0, 255, 0), thickness=2) # green
            # cv2.putText(
            #     image,
            #     f'{i}',
            #     kp2d_plot,
            #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #     fontScale=1,
            #     color=(0, 0, 255),
            #     thickness=1)
            cv2.circle(image, joint_plot, radius=4, color=(255, 0, 0), thickness=2) # blue
            cv2.line(image, kp2d_plot, joint_plot, color=(0, 0, 255), thickness=1) # red
        else:
        #     cv2.circle(image, kp2d_plot, radius=2, color=(0, 255, 0), thickness=2)
            cv2.circle(image, joint_plot, radius=2, color=(255, 0, 0), thickness=2)
            # cv2.circle(image, joint_plot, radius=4, color=(0, 0, 255), thickness=4)

    cv2.imwrite(os.path.join(output_dir, f'smplify_view{view}_{idx}.jpg'), image)
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    return results


def project_kps3d(camera_para_dict, corrected_intrinsic, kps3d, inv_ext=True):
    if kps3d.ndim == 3:
        N, K, _ = kps3d.shape
    elif kps3d.ndim == 4:
        N, _, K, _ = kps3d.shape
    if inv_ext:
        RT = np.linalg.inv(camera_para_dict['RT'][0])
    else:
        RT = camera_para_dict['RT'][0]

    kps3d = kps3d[..., :3].reshape(-1, 3)
    xyz1 = np.concatenate([kps3d, np.ones([kps3d.shape[0], 1])],
                          axis=-1)  # (n, 4)
    xyz1 = xyz1.T  # (4, n)
    xyz1_cam = RT @ xyz1
    xyz = xyz1_cam[:3, :]  # (3, n)
    kps2d = corrected_intrinsic @ xyz
    kps2d = kps2d.T  # (n, 3)
    kps2d = kps2d[:, :2] / kps2d[:, 2, None]
    kps2d = kps2d.reshape(N, K, 2)
    # print(f'>>>Projected kps2d, {kps2d.shape}')
    return kps2d


def main(args):
    # load background
    frame_dir = args.background_dir
    if len(frame_dir) > 0 and \
            check_path_existence(frame_dir, 'dir') == \
            Existence.DirectoryExistNotEmpty:
        frame_list = os.listdir(frame_dir)
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
    # load smplx and prepare body model
    ren_body_cam_dict = np.load(
        args.annots_file, allow_pickle=True).item()['cams']
    camera_parameter = CameraParameter(name=args.view)
    camera_para_dict = {
        'RT': ren_body_cam_dict[args.view]['RT'].reshape(1, 4, 4),
        'K': ren_body_cam_dict[args.view]['K'].reshape(1, 3, 3),
    } if '00' in ren_body_cam_dict.keys() else \
        {
            'RT': ren_body_cam_dict['RT'][int(args.view)].reshape(1, 4, 4),
            'K': ren_body_cam_dict['K'][int(args.view)].reshape(1, 3, 3),
        }
    camera_parameter.load_from_lightstage(camera_para_dict, 0)
    dist_array = ren_body_cam_dict[args.view]['D'] \
        if '00' in ren_body_cam_dict.keys() else \
        np.array(ren_body_cam_dict['D'][int(args.view)])
    dist_keys = ['k1', 'k2', 'p1', 'p2', 'k3']
    distortion_coefficients = {}
    for dist_index, dist_key in enumerate(dist_keys):
        camera_parameter.set_value(dist_key, float(dist_array[dist_index]))
        distortion_coefficients[dist_key] = float(dist_array[dist_index])
    
    smplx_dict = np.load(args.smplx_path)
    transl = smplx_dict['transl']
    betas = smplx_dict['betas']
    body_model_config = dict(model_path='mmhuman3d/data/body_models')
    body_model_config['use_pca'] = False
    body_model_config['gender'] = str(smplx_dict['gender'])
    body_model_config['use_face_contour'] = True
    body_model_config['flat_hand_mean'] = True
    body_model_config['type'] = 'smplx'

    # zoehuman
    global_orient = smplx_dict['global_orient']
    frame_number = global_orient.shape[0]
    body_pose = smplx_dict['body_pose'].reshape(frame_number, -1)
    poses = np.concatenate((global_orient, body_pose), axis=1)
    jaw_pose = smplx_dict['jaw_pose']
    leye_pose = smplx_dict['leye_pose']
    reye_pose = smplx_dict['reye_pose']
    left_hand_pose = smplx_dict['left_hand_pose'].reshape(frame_number, -1)
    right_hand_pose = smplx_dict['right_hand_pose'].reshape(frame_number, -1)
    poses = np.concatenate((poses, jaw_pose, leye_pose, reye_pose,
                            left_hand_pose, right_hand_pose),
                           axis=1)

    # with xrmocap
    # poses = smplx_dict['fullpose']
    # frame_number = poses.shape[0]
    # poses = poses.reshape(frame_number, -1)

    camera_parameter.inverse_extrinsics()
    K, R, T = camera_parameter.get_KRT()

    plot_kps = True  # if visual kypts
    # TODO: temp. local: insuff memory
    if args.render_idx is not None:
        idx = args.render_idx
        poses = poses[None, idx].copy()
        transl = transl[None, idx].copy()
        image_array = image_array[None, idx].copy()
        # del poses, transl, image_array

    results = visualize_smpl_distortion(
        poses=poses,
        betas=betas,
        transl=transl,
        K=K,
        R=R,
        T=T,
        dist_coeffs=distortion_coefficients,
        overwrite=True,
        body_model_config=body_model_config,
        output_path=args.output_video_path,
        image_array=image_array,
        resolution=(image_array.shape[1], image_array.shape[2]),
        return_tensor=True,
        plot_kps=args.viskps,
        vis_kp_index=False)
    
    # output_dir = os.path.dirname(args.output_video_path)
    # results_path = os.path.join(output_dir, f'smpl_vis_results_view{args.view}.npz')
    # np.savez(results_path, results=results)
    # results = np.load(results_path)['results']

    # idxs = range(0,frame_number,30)
    # if plot_kps:
    #     for idx in idxs:
    #         results = plot_keypoints(args.annots_file, results, output_dir, idx, args.view)
    # else:
    #     image = results[0].detach().cpu().numpy()
    #     output_dir = os.path.dirname(args.output_video_path)
    #     cv2.imwrite(os.path.join(output_dir, 'smplify.jpg'), image)


def setup_parser():
    parser = argparse.ArgumentParser(description='')
    # input args
    parser.add_argument(
        '--smplx_path',
        type=str,
        help='Path to the input smplx npz file.',
        default='')
    parser.add_argument(
        '--annots_file',
        type=str,
        help='Path to the annots.npy file',
        default='')
    parser.add_argument(
        '--view', type=str, help='Which view, like 00.', default='00')
    parser.add_argument('--gender', type=str, default='neutral')
    parser.add_argument('--render_idx', type=int, default=None)
    # output args
    parser.add_argument(
        '--output_video_path',
        type=str,
        help='Path to the output video.',
        default='./default_output.mp4')
    # optional args
    parser.add_argument(
        '--background_dir',
        type=str_or_empty,
        help='Path to the directory with background images in it.',
        default='')
    parser.add_argument(
        '--viskps',
        action='store_true',
        help='If checked, visualize smpl kps.',
        default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    main(args)
