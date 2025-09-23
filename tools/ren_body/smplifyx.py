import argparse
import os
import os.path as osp

import mmcv
import numpy as np
import torch
# from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
# from mmhuman3d.core.evaluation import keypoint_mpjpe
# from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_pose
from mmhuman3d.data.data_structures.human_data import HumanData
# from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.models.registrants.builder import build_registrant
# from mmhuman3d.utils.transforms import aa_to_rotmat, rotmat_to_aa
from pytorch3d.io import IO
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes
from tqdm import tqdm

from zoehuman.core.conventions.keypoints_mapping import convert_kps

# additional updates:
# 1. body model keypoints_dst aligned
# 2. use the same betas for all frames
# 3.
# 4.


def parse_args():
    parser = argparse.ArgumentParser('SMPLify tools')
    parser.add_argument('--kp3d_path', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
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

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    smplify_config = mmcv.Config.fromfile(args.config)
    # unpdata smplify_config from args
    smplify_config.body_model.update(dict(type=args.model))
    smplify_config.body_model.update(dict(gender=args.gender))
    assert smplify_config.body_model.type.lower() in ['smpl', 'smplx']
    assert smplify_config.type.lower() in ['smplify', 'smplifyx']

    # set cudnn_benchmark
    if smplify_config.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load 3D keypoints
    human_data = HumanData.fromfile(args.kp3d_path)
    keypoints_src_mask = human_data['keypoints3d_mask']
    keypoints_src = human_data['keypoints3d'][..., :3]

    # map 3D keypoints
    keypoints, mask = convert_kps(
        keypoints_src,
        mask=keypoints_src_mask,
        src=args.src_convention,
        dst=smplify_config.body_model['keypoint_dst'])
    keypoints_conf = np.repeat(mask[None], keypoints.shape[0], axis=0)

    keypoints = torch.tensor(keypoints, dtype=torch.float32, device=device)
    keypoints_conf = torch.tensor(
        keypoints_conf, dtype=torch.float32, device=device)

    # update batch size
    smplify_config.body_model.update(dict(batch_size=keypoints.shape[0]))

    # create registrant
    smplify = build_registrant(dict(smplify_config))

    # run registrant
    smplify_output = smplify(
        keypoints3d=keypoints,
        keypoints3d_conf=keypoints_conf,
        return_verts=args.save_mesh,
        return_joints=True)
    smplify_output.update(smplify.body_model(
        return_verts=args.save_mesh, **smplify_output))
    smplify_output = {k: v.detach().cpu() for k, v in smplify_output.items()}

    # save results
    output_folder_npz = args.output_folder
    os.makedirs(output_folder_npz, exist_ok=True)
    stem, _ = osp.splitext(osp.basename(args.kp3d_path))
    npz_path = osp.join(output_folder_npz, f'{stem}_smplx.npz')

    if args.save_mesh:
        face = smplify.body_model.faces
        if isinstance(face, np.ndarray):
            face = torch.from_numpy(face.astype(np.int32))
        vertices = smplify_output['vertices'].clone()
        smplify_output.pop('vertices', None)
    np.savez(
        npz_path,
        # keypoints3d=keypoints3d, # TODO: 55 keypoints?
        scale=args.scale,
        gender=args.gender,
        **smplify_output)

    if args.save_mesh:
        output_folder_mesh = os.path.join(args.output_folder, '..', 'mesh')
        os.makedirs(output_folder_mesh, exist_ok=True)
        for frame_index in tqdm(range(vertices.shape[0])):
            #  vertices.shape: [1, 10475, 3]
            #  faces.shape: [20908, 3]
            meshes = Meshes(
                verts=vertices[frame_index:frame_index + 1, :, :],
                faces=face.view(1, -1, 3),
                textures=TexturesVertex(
                    verts_features=torch.FloatTensor((
                        1, 1,
                        1)).view(1, 1, 3).repeat(1, vertices.shape[-2], 1)))
            mesh_path = os.path.join(output_folder_mesh,
                                     f'{frame_index:06d}.obj')
            IO().save_mesh(data=meshes, path=mesh_path)

    # # TODO: visualize
    # if args.show_path is not None:
    #     # visualize smpl pose
    #     body_model_dir = os.path.dirname(args.body_model_dir.rstrip('/'))
    #     body_model_config.update(
    #         model_path=body_model_dir,
    #         model_type=smplify_config.body_model.type.lower())
    #     visualize_smpl_pose(
    #         poses=poses,
    #         body_model_config=body_model_config,
    #         output_path=args.show_path,
    #         orbit_speed=1,
    #         overwrite=True)


if __name__ == '__main__':
    main()
