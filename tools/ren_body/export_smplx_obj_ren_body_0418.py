import argparse
import os

import mmcv
import numpy as np
import torch
from pytorch3d.io import IO
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes
from tqdm import tqdm

from zoehuman.models.builder import build_body_model
from zoehuman.utils import str_or_empty
from zoehuman.utils.path_utils import Existence, check_path_existence


def prepare_body_model(device, gender='neutral'):
    smplify_config = dict(
        mmcv.Config.fromfile('mmhuman3d/configs/smplify/smplifyx.py'))
    body_model_config = smplify_config['body_model']
    body_model_config['model_path'] = os.path.join(
        'mmhuman3d', body_model_config['model_path'])
    model_type = body_model_config.get('type', 'smplx')
    body_model_config.update(type=model_type.lower())
    body_model_config.update(gender=gender.lower())
    body_model_config.update(
        dict(use_pca=False, use_face_contour=True, flat_hand_mean=True))
    if body_model_config['model_path'] is not None:
        body_model = build_body_model(body_model_config)
    else:
        raise FileNotFoundError('Wrong model_path.'
                                ' File or directory does not exist.')
    return body_model.to(device)


def prepare_mesh(poses, betas, transl, start, end, body_model):
    NUM_JOINTS = body_model.NUM_JOINTS
    NUM_DIM = 3 * (NUM_JOINTS + 1)
    joints = None
    if poses.shape[-1] != NUM_DIM:
        raise ValueError(f'Please make sure your poses is {NUM_DIM} dims in'
                         f'the last axis. Your input shape: {poses.shape}')
    poses = poses.view(poses.shape[0], -1, (NUM_JOINTS + 1) * 3)
    num_frames, num_person, _ = poses.shape
    full_pose = poses[start:end]
    # slice the input poses, betas, and transl.
    betas = betas[start:end] if betas is not None else None
    if betas.shape[0] == 1:
        betas = betas.repeat(full_pose.shape[0], 1)
    transl = transl[start:end] if transl is not None else None
    pose_dict = body_model.tensor2dict(
        full_pose=full_pose, betas=betas, transl=transl)
    for k, v in pose_dict.items():
        pose_dict[k] = v.to(poses.device)
    # get new num_frames
    num_frames = full_pose.shape[0]
    model_output = body_model(**pose_dict)
    vertices = model_output['vertices']
    faces = body_model.faces_tensor
    joints = model_output['joints']
    return vertices, faces, joints, num_frames, num_person


def main(args):
    # check output path
    exist_result = check_path_existence(args.output_dir, 'dir')
    if exist_result == Existence.MissingParent:
        raise FileNotFoundError
    elif exist_result == Existence.DirectoryNotExist:
        os.mkdir(args.output_dir)
    gender = 'neutral'
    device_name = 'cuda'
    scale = 1.0
    body_model = prepare_body_model(gender=gender, device=device_name)
    smplx_path = args.smplx_path
    mesh_dir = args.output_dir
    # load smpl
    smplx_data = np.load(smplx_path)
    global_orient = smplx_data['global_orient']
    n = global_orient.shape[0]
    body_pose = smplx_data['body_pose'].reshape(n, -1)
    poses = np.concatenate((global_orient, body_pose), axis=1)
    transl = torch.from_numpy(smplx_data['transl']).to(device_name)
    betas = torch.from_numpy(smplx_data['betas']).to(device_name)
    jaw_pose = smplx_data['jaw_pose']
    leye_pose = smplx_data['leye_pose']
    reye_pose = smplx_data['reye_pose']
    left_hand_pose = smplx_data['left_hand_pose'].reshape(n, -1)
    right_hand_pose = smplx_data['right_hand_pose'].reshape(n, -1)
    poses = np.concatenate((poses, jaw_pose, leye_pose, reye_pose,
                            left_hand_pose, right_hand_pose),
                           axis=1)
    poses = torch.from_numpy(poses).to(device_name)
    # apply to mesh
    vertices, faces, _, _, _ = prepare_mesh(poses, betas, transl, 0,
                                            poses.shape[0], body_model)
    vertices /= scale
    for frame_index in tqdm(range(poses.shape[0])):
        meshes = Meshes(
            verts=vertices[frame_index:frame_index + 1, :, :],
            faces=faces.clone().view(1, -1, 3),
            textures=TexturesVertex(
                verts_features=torch.FloatTensor((
                    1, 1, 1)).view(1, 1, 3).repeat(1, vertices.shape[-2], 1)))
        mesh_path = os.path.join(mesh_dir, f'{frame_index:06d}.obj')
        IO().save_mesh(data=meshes, path=mesh_path)
    return 0


def setup_parser():
    parser = argparse.ArgumentParser(description='')
    # input args
    parser.add_argument(
        '--smplx_path',
        type=str_or_empty,
        help='Path to the smplx npz file.',
        default='')
    # output args
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to the directory saving meshes.',
        default='./default_output')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    main(args)
