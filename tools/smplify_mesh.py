import argparse
import os

import mmcv
import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes

from zoehuman.core.cameras.camera_parameters import CameraParameter
from zoehuman.core.conventions.keypoints_mapping import convert_kps
from zoehuman.core.visualization.visualize_smpl import \
    visualize_smpl_calibration  # prevent yapf isort conflict
from zoehuman.data.data_structures.human_data import HumanData
from zoehuman.models.builder import build_registrant
from zoehuman.utils import str_or_empty
from zoehuman.utils.path_utils import (  # prevent yapf isort conflict
    Existence, check_path_existence, check_path_suffix,
)


def prepare_keypoints3d(human_data_path, dst_convention, device):
    """"""
    if check_path_existence(human_data_path, 'file') == \
                Existence.FileExist:
        human_data_3d = HumanData.fromfile(human_data_path)
    else:
        raise FileNotFoundError(f'HumanData3d not found:\n{human_data_path}')
    # load keypoints
    keypoints3d, keypoints3d_mask = convert_kps(
        keypoints=human_data_3d['keypoints3d'],
        mask=human_data_3d['keypoints3d_mask'],
        src='human_data',
        dst=dst_convention)
    keypoints3d = torch.from_numpy(keypoints3d[:, :, :3]).to(
        dtype=torch.float32, device=device)
    keypoints3d_conf = torch.from_numpy(np.expand_dims(keypoints3d_mask,
                                                       0)).to(
                                                           dtype=torch.float32,
                                                           device=device)
    return keypoints3d, keypoints3d_conf, human_data_3d


def prepare_mesh(mesh_path, device):
    if check_path_existence(mesh_path, 'file') ==\
            Existence.FileExist:
        if check_path_suffix(mesh_path, ['.obj']):
            mesh_raw = load_objs_as_meshes([
                mesh_path,
            ], device=device)
        else:
            raise NotImplementedError('Only obj mesh is accpetable.')
    else:
        raise FileNotFoundError(f'Mesh file not found:\n{mesh_path}')
    return mesh_raw


def prepare_camera(cam_param_path):
    if len(cam_param_path) <= 0:
        return None, None
    else:
        camera_param_list = []
        camera_pytorch3d_list = []
        if check_path_existence(cam_param_path, 'dir') == \
                Existence.DirectoryExistNotEmpty:
            file_names = os.listdir(cam_param_path)
            file_names.sort()
            for file_name in file_names:
                camera_parameter = CameraParameter(name=file_name)
                camera_parameter.load(os.path.join(cam_param_path, file_name))
                camera_parameter.inverse_extrinsics()
                camera_param_list.append(camera_parameter)
                camera_pytorch3d_list.append(
                    camera_parameter.export_to_perspective_cameras())

        else:
            raise FileNotFoundError('CameraParameter not found.')
    return camera_param_list, camera_pytorch3d_list


def main(args):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    keypoints_convention = 'smpl_45' if args.model_type == 'smpl'\
        else 'smplx'
    keypoints3d, keypoints3d_conf, human_data_3d = \
        prepare_keypoints3d(
            args.human_data_3d_path,
            dst_convention=keypoints_convention, device=device)
    meshes = prepare_mesh(args.mesh_path, device=device)
    # meshes = None
    cam_param_list, cam_pytorch3d_list = \
        prepare_camera(args.cam_param_dir)
    # build
    if args.model_type == 'smpl':
        cfg = mmcv.Config.fromfile('configs/smplify/smplify.py')
    else:
        raise NotImplementedError
    smplify_config = dict(cfg)
    smplify_config['verbose'] = True
    smplify_config['use_one_betas_per_video'] = True
    smplify_config['num_epochs'] = args.num_epochs
    smplify_config['camera'] = None if cam_pytorch3d_list is None \
        else cam_pytorch3d_list[0]
    smplify = build_registrant(smplify_config)
    # run
    smplify_output = smplify(
        keypoints3d=keypoints3d,
        keypoints3d_conf=keypoints3d_conf,
        human_mesh=meshes)
    result_human_data = human_data_3d
    result_human_data[args.model_type] = {}
    smpl_dict = result_human_data[args.model_type]
    for k, v in smplify_output.items():
        if isinstance(v, torch.Tensor):
            np_v = v.detach().cpu().numpy()
            assert not np.any(np.isnan(np_v)), f'{k} fails.'
            smpl_dict[k] = np_v
    result_human_data.dump(
        os.path.join(args.output_dir, f'human_data_{args.model_type}.npz'))
    if args.visualize and cam_param_list is not None:
        for cam_index, cam_param in enumerate(cam_param_list):
            tmp_params = cam_param.get_KRT(k_dim=4)
            for p_index, tmp_param in enumerate(tmp_params):
                tmp_params[p_index] = \
                    torch.Tensor(tmp_param).unsqueeze(0)
            if cam_index == 0:
                params = tmp_params
            else:
                for p_index, _ in enumerate(tmp_params):
                    params[p_index] = \
                        torch.cat(
                            (params[p_index], tmp_params[p_index]),
                            dim=0)
        height, width = \
            cam_param_list[0].get_value('H'),\
            cam_param_list[0].get_value('W')
        resolution = (height, width)
        overlay_path = os.path.join(args.output_dir,
                                    f'{args.model_type}_overlay.mp4')
        global_orient = torch.Tensor(smpl_dict['global_orient']).to(device)
        n = global_orient.shape[0]
        body_pose = torch.Tensor(smpl_dict['body_pose']).reshape(n,
                                                                 -1).to(device)
        transl = torch.Tensor(smpl_dict['transl']).to(device)
        betas = torch.Tensor(smpl_dict['betas']).to(device)
        poses = torch.cat((global_orient, body_pose), dim=1)
        if len(args.frames_dir) <= 0:
            frame_list = None
        else:
            frame_list = []
            file_list = sorted(os.listdir(args.frames_dir))
            for file_name in file_list:
                frame_path = os.path.join(args.frames_dir, file_name)
                if check_path_suffix(
                        frame_path, allowed_suffix=['.png', '.jpg']):
                    frame_list.append(frame_path)
        visualize_smpl_calibration(
            poses=poses.repeat(len(cam_param_list), 1),
            betas=betas.repeat(len(cam_param_list), 1),
            transl=transl.repeat(len(cam_param_list), 1),
            body_model=smplify.body_model,
            K=params[0],
            R=params[1],
            T=params[2],
            overwrite=True,
            device=device,
            output_path=overlay_path,
            frame_list=frame_list,
            resolution=resolution)


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Run smplify for mesh and ' +
        'keypoints3d extracted from its renderer images.')
    # input args
    parser.add_argument(
        '--mesh_path',
        type=str_or_empty,
        help='Path to input mesh file.',
        default='')
    parser.add_argument(
        '--human_data_3d_path',
        type=str,
        help='Path to the HumanData file for keypoints3d.',
        default='')
    parser.add_argument(
        '--cam_param_dir',
        type=str_or_empty,
        help='Path to the directory for CameraParameter dumped file.' +
        'It shall be data_dir/camera_parameters from ' +
        'tools/render_detect_human_meshes.py.',
        default='')
    parser.add_argument(
        '--frames_dir',
        type=str_or_empty,
        help='Path to the frames directory for visualization.',
        default='')
    # model args
    parser.add_argument(
        '--model_type',
        type=str_or_empty,
        choices=['smpl', 'smplx'],
        help='Which registrant to use.',
        default='smpl')
    parser.add_argument(
        '--num_epochs', type=int, help='Epoch number.', default=3)
    # output args
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to the directory saving ' + 'all possible output files.',
        default='./default_output')
    # output args
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='If checked, visualize result.',
        default=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    main(args)
