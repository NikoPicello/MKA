import argparse
import os
import os.path as osp
import pickle
import time

import numpy as np
import torch
from body_models import batch_rodrigues, load_model
from config import CONFIG
from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_pose
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.body_models.utils import batch_transform_to_camera_frame
from mmhuman3d.models.builder import build_body_model
# from mmhuman3d.utils.transforms import aa_to_rotmat, rotmat_to_aa
from pipeline import smpl_from_keypoints3d
from pipeline.weight import load_weight_pose, load_weight_shape
from pytorch3d.io import IO
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes
from tqdm import tqdm

from zoehuman.core.conventions.keypoints_mapping import convert_kps


def prepare_body_model(device, gender='neutral'):
    body_model_config = dict(
        type='SMPLX',
        gender='neutral',
        num_betas=10,
        use_face_contour=True,
        keypoint_src='smplx',
        keypoint_dst='smplx',
        model_path='mmhuman3d/data/body_models/smplx/',
        batch_size=1)
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


def remove_nan(x):
    njoints = x.shape[1]
    x_nan = np.isnan(x).any(axis=-1)
    nx = x[~x_nan, ...]
    idxs = np.argwhere(~x_nan)
    start = idxs[:, 0].min()
    end = idxs[:, 0].max()
    nframe = end - start + 1
    assert nframe > 0
    nx = nx.reshape(nframe, njoints, 3)
    return nx, start, end


def load_body_model(args):
    # load body model
    weight_pose = load_weight_pose(args.model)
    weight_shape = load_weight_shape(args.model)
    print('Loading {}, {}'.format(args.model, args.gender))
    body_model = load_model(args.gender, model_type=args.model)
    print('Load body model successfully')
    return body_model, weight_pose, weight_shape


def smplify3d(output_folder,
              kp3d_path,
              args,
              body_model=None,
              weight_pose=None,
              weight_shape=None):
    config = CONFIG[args.body]
    opt_scale = args.opt_scale
    scale = args.scale
    model_dir = 'mmhuman3d/data/body_models/'

    if body_model is None:
        body_model, weight_pose, weight_shape = load_body_model(args)

    input_folder = osp.dirname(kp3d_path).strip()
    input_folder = input_folder[:input_folder.rfind('/')]

    # read keypoints3d
    if kp3d_path.endswith('.pickle') or kp3d_path.endswith('.pkl'):
        with open(kp3d_path, 'rb') as f:
            keypoints_ori = pickle.load(f, encoding='latin1')
        keypoints_ori_mask = None
    elif kp3d_path.endswith('.npz'):
        # human_data_3d = np.load(kp3d_path)
        human_data_3d = HumanData.fromfile(kp3d_path)
        keypoints_ori = human_data_3d['keypoints3d'][..., :3]
        if 'keypoints3d_mask' in human_data_3d:
            keypoints_ori_mask = human_data_3d['keypoints3d_mask']
        else:
            keypoints_ori_mask = human_data_3d['mask']
    else:
        raise TypeError('Unsupported suffix:', kp3d_path)

    print('input', keypoints_ori.shape)

    # # TODO: temp test for bow
    # import pdb; pdb.set_trace()
    # keypoints_ori = keypoints_ori[None, 60].repeat(150, axis=0)

    if keypoints_ori.ndim == 3:
        keypoints_ori = keypoints_ori[:, None, :, :]
    assert keypoints_ori.ndim == 4

    nperson = keypoints_ori.shape[1]
    print('#person:', nperson)
    print('tgt_convention:', args.tgt_convention)

    gender = args.gender
    for pid in range(nperson):
        keypoints_src = keypoints_ori[:, pid, :, :]
        keypoints_src_mask = keypoints_ori_mask

        # keypoints_src, start, end = remove_nan(keypoints_src)
        print('keypoints_src', keypoints_src.shape)

        if args.tgt_convention == 'smpl':
            keypoints_mid, mask_mid = convert_kps(
                keypoints_src,
                src=args.src_convention,
                dst='smpl',
                mask=keypoints_src_mask)

            keypoints, mask = convert_kps(
                keypoints_mid,
                src='smpl',
                dst='openpose_25',
                mask=mask_mid,
                approximate=True)

        elif args.model == 'smplx':
            print('using smplx')
            keypoints, mask = convert_kps(
                keypoints_src,
                src=args.src_convention,
                dst='openpose_118',
                mask=keypoints_src_mask)
            print('success')

        else:
            keypoints, mask = convert_kps(
                keypoints_src,
                src=args.src_convention,
                dst=args.tgt_convention,
                mask=keypoints_src_mask,
                approximate=True)

        print('mean', keypoints.mean())
        print('shape', keypoints.shape)
        print('mask', len(mask), mask)

        N, J, D = keypoints.shape
        assert D == 3
        keypoints_conf = np.tile(mask.reshape(1, -1), [N, 1])[..., None]

        print('conf', keypoints_conf.shape)
        keypoints3d = np.concatenate((keypoints, keypoints_conf), axis=-1)
        print('kp3d', keypoints3d.shape)

        # keypoints3d should be [N, J, 4], the last dimension is confidence

        # fitting
        t0 = time.time()

        # # TODO: temp. bad case: ankle
        # np.save(kp3d_save_path, keypoints3d)

        # TODO: temp. bad case: ankle
        weight_shape = {'s3d': 1., 'reg_shapes': 1e-3}  # fix betas
        # weight_pose = {'k3d': 1.0,
        #                 'k3d_hand': 5.0,
        #                 'k3d_face': 2.0,
        #                 'reg_poses_zero': 0.01,
        #                 'smooth_body': 0.5,
        #                 'smooth_poses': 0.1,
        #                 'smooth_hand': 0.001,
        #                 'reg_hand': 0.0001,
        #                 'reg_expr': 0.01,
        #                 'reg_head': 0.01,
        #                 'k2d': 0.0001}

        body_params = smpl_from_keypoints3d(
            body_model,
            keypoints3d,
            config,
            args,
            weight_shape=weight_shape,
            weight_pose=weight_pose,
            opt_scale=opt_scale,
            scale=scale)
        t1 = time.time()
        print(f'{t1 - t0} s')

        global_orient = body_params['Rh']
        transl = body_params['Th']
        betas = body_params['shapes']
        # scale = 0.25
        # if opt_scale:
        #     scale = body_params['scale']

        name = osp.basename(kp3d_path)
        name = name[:name.rfind('.')]

        output_folder_npz = osp.join(output_folder,
                                     f'{args.model}_{args.gender}')
        os.makedirs(output_folder_npz, exist_ok=True)

        ofn = f'{name}_{args.model}'
        if nperson > 1:
            ofn = f'{ofn}_p{pid}'

        npz_path = osp.join(output_folder_npz.strip(), f'{ofn}.npz')

        vis_path = osp.join(output_folder_npz.strip(), f'{ofn}.mp4')

        if args.model == 'smpl':
            body_pose = body_params['poses'][:, 3:]
            poses = dict(
                body_pose=body_pose,
                betas=betas,
                transl=transl,
                global_orient=global_orient)
            # compute correct transl
            smpl_model = load_model(
                gender=gender,
                use_cuda=True,
                model_type='smpl',
                skel_type='body25',
                device=None,
                model_path='data/smplx')
            # get transl_new
            _, keypoints3d = smpl_model(
                return_verts=True,
                return_joints=True,
                return_tensor=True,
                poses=body_params['poses'],
                shapes=betas,
                Rh=global_orient,
                Th=transl)
            keypoints3d = keypoints3d.detach().cpu()
            j0 = keypoints3d[:, 0, :]
            rot = batch_rodrigues(torch.from_numpy(global_orient))
            transl_new = torch.from_numpy(transl) - j0 + torch.einsum(
                'bij,bj->bi', rot, j0)
            transl_new = transl_new.detach().cpu().numpy()
            # get keypoints3d
            _, keypoints3d = smpl_model(
                return_verts=True,
                return_joints=True,
                return_tensor=True,
                poses=body_params['poses'],
                shapes=betas,
                Rh=global_orient,
                Th=transl_new)
            keypoints3d = keypoints3d.detach().cpu()

            keypoints3d = keypoints3d * scale

            print(f'Save to {npz_path} ...')
            np.savez(
                npz_path,
                body_pose=body_pose,
                betas=betas,
                transl=transl_new,
                keypoints3d=keypoints3d,
                global_orient=global_orient,
                scale=scale)

            if args.vis_smpl:
                visualize_smpl_pose(
                    poses=poses,
                    body_model_config=dict(
                        model_path=model_dir, model_type=args.model),
                    output_path=vis_path,
                    convention='opencv',
                    # convention='pytorch3d',
                    orbit_speed=1,
                    overwrite=True)
        elif args.model == 'smplx':
            print('smplx with gender', gender)
            body_pose = body_params['poses']
            expression = body_params['expression']
            smpl_model = load_model(
                gender=gender,
                use_cuda=True,
                model_type='smplx',
                skel_type='body25',
                device=None,
                model_path='mmhuman3d/data/body_models')
            verts, keypoints3d = smpl_model(
                return_verts=True,
                return_joints=True,
                return_tensor=True,
                poses=body_pose,
                shapes=betas,
                Rh=global_orient,
                Th=transl,
                expression=expression)
            # correct tranls
            j0 = keypoints3d[:, 0, :].detach().cpu()
            rot = batch_rodrigues(torch.from_numpy(global_orient))
            transl_new = torch.from_numpy(transl) - j0 + torch.einsum(
                'bij,bj->bi', rot, j0)
            transl_new = transl_new.detach().cpu().numpy()
            verts = verts.detach().cpu().numpy()
            keypoints3d = keypoints3d.detach().cpu().numpy()
            keypoints3d = keypoints3d * scale
            verts = verts * scale
            print('keypoints3d', keypoints3d.shape)
            print('body_pose', body_pose.shape)
            print('transl', transl.shape)
            print('expression', expression.shape)
            full_pose = body_model.full_poses(body_pose).reshape(-1, 55, 3)

            full_pose[:, 0, :] = global_orient

            # full_pose = full_pose.detach().cpu().numpy()

            body_pose = full_pose[:, 1:22, :]  # N, 21, 3
            jaw_pose = full_pose[:, 22, :]  # N, 1,  3
            leye_pose = full_pose[:, 23, :]  # N, 1,  3
            reye_pose = full_pose[:, 24, :]  # N, 1,  3
            left_hand_pose = full_pose[:, 25:40, :]  # N, 15, 3
            right_hand_pose = full_pose[:, 40:55, :]  # N, 15, 3

            print('full_pose', full_pose.shape)
            print('body_pose', body_pose.shape)
            print('jaw_pose', jaw_pose.shape)
            print('leye_pose', leye_pose.shape)
            print('reye_pose', reye_pose.shape)
            print('left_hand_pose', left_hand_pose.shape)
            print('right_hand_pose', right_hand_pose.shape)
            # save results
            npz_path = osp.join(output_folder_npz.strip(), f'{ofn}.npz')
            print('save to', npz_path)
            # np.savez(
            #     npz_path,
            #     body_pose=body_pose,
            #     jaw_pose=jaw_pose,
            #     leye_pose=leye_pose,
            #     reye_pose=reye_pose,
            #     left_hand_pose=left_hand_pose,
            #     right_hand_pose=right_hand_pose,
            #     betas=betas,
            #     transl=transl_new,
            #     global_orient=global_orient,
            #     expression=expression,
            #     keypoints3d=keypoints3d,
            #     scale=scale,
            #     # start=start,
            #     # end=end)
            # )

            import pdb
            pdb.set_trace()
            # TODO: temp fix for bow
            body_pose = body_params['poses']
            verts, keypoints3d = smpl_model(
                return_verts=True,
                return_joints=True,
                return_tensor=True,
                poses=body_pose,
                shapes=betas,
                # Rh=global_orient,
                # Th=transl,
                expression=expression)
            # correct tranls
            j0 = keypoints3d[:, 0, :].detach().cpu()
            batch_size = global_orient.shape[0]
            extrinsic = np.eye(4)[None].repeat(batch_size, axis=0)
            # extrinsic[:, :3, :3] = aa_to_rotmat(global_orient)
            extrinsic[:, :3, 3] = transl
            new_gloabl_orient, new_transl = batch_transform_to_camera_frame(
                global_orient=body_pose[:, :3],
                transl=np.zeros((batch_size, 3)),
                pelvis=j0.detach().cpu().numpy(),
                extrinsic=extrinsic)
            full_pose = body_model.full_poses(body_pose).reshape(-1, 55, 3)
            body_pose = full_pose[:, 1:22, :]  # N, 21, 3
            keypoints3d = keypoints3d.detach().cpu().numpy()
            np.savez(
                npz_path,
                body_pose=body_pose,
                jaw_pose=jaw_pose,
                leye_pose=leye_pose,
                reye_pose=reye_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                betas=betas,
                transl=new_transl,
                global_orient=new_gloabl_orient,
                expression=expression,
                keypoints3d=keypoints3d,
                scale=scale,
                # start=start,
                # end=end)
            )
            # TODO: temp fix for bow ends

            if args.vis_smpl:
                visualize_smpl_pose(
                    verts=verts,
                    body_model_config=dict(
                        model_path=model_dir,
                        use_flat_hand=True,
                        use_pca=False,
                        use_face_contour=True),
                    model_type=args.model,
                    output_path=vis_path,
                    convention='opencv',
                    orbit_speed=2,
                    overwrite=True)

            if args.save_mesh:
                output_folder_mesh = osp.join(output_folder, 'mesh')
                os.makedirs(output_folder_mesh, exist_ok=True)
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                else:
                    device = torch.device('cpu')

                body_model = prepare_body_model(gender=gender, device=device)
                n = global_orient.shape[0]
                body_pose = body_pose.reshape(n, -1)
                left_hand_pose = left_hand_pose.reshape(n, -1)
                right_hand_pose = right_hand_pose.reshape(n, -1)
                poses = np.concatenate((global_orient, body_pose), axis=1)
                poses = np.concatenate((poses, jaw_pose, leye_pose, reye_pose,
                                        left_hand_pose, right_hand_pose),
                                       axis=1)
                transl = transl_new
                poses = torch.from_numpy(poses).to(device)
                transl = torch.from_numpy(transl).to(device)
                betas = torch.from_numpy(betas).to(device)
                vertices, faces, _, _, _ = prepare_mesh(
                    poses, betas, transl, 0, poses.shape[0], body_model)
                vertices = vertices * scale

                print(f'save {n} frames to {output_folder_mesh}')
                t0 = time.time()
                for frame_index in tqdm(range(poses.shape[0])):
                    #  vertices.shape: [1, 10475, 3]
                    #  faces.shape: [20908, 3]
                    meshes = Meshes(
                        verts=vertices[frame_index:frame_index + 1, :, :],
                        faces=faces.clone().view(1, -1, 3),
                        textures=TexturesVertex(
                            verts_features=torch.FloatTensor((1, 1, 1)).view(
                                1, 1, 3).repeat(1, vertices.shape[-2], 1)))
                    mesh_path = os.path.join(output_folder_mesh,
                                             f'{frame_index:06d}.obj')
                    IO().save_mesh(data=meshes, path=mesh_path)
                t1 = time.time()
                print('success!', t1 - t0)
        else:
            raise TypeError


def get_gender(input_path):
    gender_dict = {}  # construct gender dict from file
    name = input_path.split('/')[7]
    name = name.split('_')[0]
    if name in gender_dict:
        gender = gender_dict[name]
        gender = 'female' if gender == 0 else 'male'
    else:
        gender = 'neutral'
    print(f'Dealing with {name}({gender})')
    return gender


def load_parser():
    parser = argparse.ArgumentParser('SMPLify tools')
    parser.add_argument('--kp3d_path', type=str, required=True)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--src_convention', type=str, default='smplx')
    parser.add_argument('--tgt_convention', type=str, default='smpl')

    parser.add_argument(
        '--body',
        type=str,
        default='body25',
        choices=[
            'body15', 'body25', 'h36m', 'bodyhand', 'bodyhandface', 'handl',
            'handr', 'total'
        ])
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
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--verbose', action='store_true')

    # visualization
    parser.add_argument('--vis_smpl', action='store_true')
    parser.add_argument('--save_mesh', action='store_true')
    parser.add_argument('--opt_scale', action='store_true')
    return parser


def parse_parser(parser):
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    parser = load_parser()
    args = parse_parser(parser)

    if osp.isdir(args.kp3d_path):
        raise NotImplementedError
    elif osp.isfile(args.kp3d_path):
        if args.kp3d_path.endswith('.txt'):
            with open(args.kp3d_path) as f:
                lst = [x.strip() for x in f.readlines()]
        else:
            lst = [args.kp3d_path]
        for input_path in lst:
            try:
                print(input_path)
                # set gender
                # args.gender = get_gender(input_path)
                # manually set for renbody output_folder
                output_folder = '/'.join(input_path.split('/')[:-3])
                smplify3d(output_folder, input_path, args)
            except Exception as e:
                print(e)
                with open('logs/error.txt', 'a') as of:
                    of.write(f'{input_path}, {e}')
    else:
        raise TypeError('Unsupported input file: ', args.kp3d_path)
