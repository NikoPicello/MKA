import argparse
import os

import numpy as np

from zoehuman.utils.path_utils import Existence, check_path_existence


def main(args):
    # check input path
    if check_path_existence(args.dz_dir,
                            'dir') != Existence.DirectoryExistNotEmpty:
        raise FileNotFoundError
    if check_path_existence(args.annots_path, 'file') != Existence.FileExist:
        raise FileNotFoundError
    # check output path
    exist_result = check_path_existence(args.output_dir, 'dir')
    if exist_result == Existence.MissingParent:
        raise FileNotFoundError
    elif exist_result == Existence.DirectoryNotExist:
        os.makedirs(args.output_dir)
    if args.vis_kp2d:
        run_visualize_kp2d(args)
    if args.vis_kp3d:
        run_visualize_kp3d(args)
    if args.vis_project:
        run_visualize_project(args)


def run_visualize_kp2d(args):
    views_dir = os.path.join(args.dz_dir, 'image')
    pose_2d_dir = os.path.join(args.output_dir, 'pose_2d')
    if args.views == 'all':
        view_list = [f'{x:02d}' for x in range(60)]
    else:
        view_list = args.views.strip().split('_')
    val = 0
    for view_str in view_list:
        hd_path = os.path.join(pose_2d_dir, f'human_data_{view_str}.npz')
        video_path = os.path.join(pose_2d_dir, f'human_data_{view_str}.mp4')
        background_dir = os.path.join(views_dir, view_str)
        if check_path_existence(background_dir, 'dir') != \
                Existence.DirectoryExistNotEmpty:
            continue
        cmd_str = f'python -u tools/visualization/visualize_kp2d.py \
                    {hd_path} {video_path} \
                    --background_dir {background_dir}'

        val = os.system(cmd_str) + val
    return val


def run_visualize_kp3d(args):
    type_list = ['optim', 'no_optim']
    pose3d_dir = os.path.join(args.output_dir, 'pose_3d')
    val = 0
    for type_str in type_list:
        hd_path = os.path.join(pose3d_dir, type_str, 'human_data_tri.npz')
        video_path = os.path.join(pose3d_dir, type_str, 'human_data_tri.mp4')
        cmd_str = f'python -u tools/visualization/visualize_kp3d.py \
                    --human_data_path {hd_path} \
                    --output_video_path {video_path}'

        val = os.system(cmd_str) + val
    return val


def run_visualize_project(args):
    type_list = ['optim', 'no_optim']
    pose3d_dir = os.path.join(args.output_dir, 'pose_3d')
    annots_dict = np.load(args.annots_path, allow_pickle=True).item()
    cam_dict = annots_dict['cams']
    view_keys = []
    human_data_keys = ''
    cam_param_keys = []
    # images in view_key[n] must match camera in cam_param_index[n]
    for view_index in range(60):
        view_str = f'{view_index:02d}'
        if view_str in cam_dict:
            view_keys.append(view_str)
            cam_param_keys.append(view_str)
            human_data_keys += f'_{view_index:02d}'
    human_data_keys = human_data_keys[1:]
    cam_parameters_keys = human_data_keys
    views_dir = os.path.join(args.dz_dir, 'image')
    if args.views == 'all':
        view_list = [f'{x:02d}' for x in range(60)]
    else:
        view_list = args.views.strip().split('_')
    val = 0
    for type_str in type_list:
        hd_path = os.path.join(pose3d_dir, type_str, 'human_data_tri.npz')
        output_dir = os.path.join(pose3d_dir, type_str, 'project')
        cmd_str = f'python -u tools/ren_body/project_ren_body_0418.py \
                    --human_data_path {hd_path} \
                    --cam_parameters_path {args.annots_path} \
                    --cam_parameters_type ren_body_0418 \
                    --cam_parameters_keys {cam_parameters_keys} \
                    --output_dir {output_dir}'

        val = os.system(cmd_str) + val
        if val == 0:
            pose_2d_dir = output_dir
            for view_str in view_list:
                hd_path = os.path.join(pose_2d_dir,
                                       f'human_data_{view_str}.npz')
                video_path = os.path.join(pose_2d_dir,
                                          f'human_data_{view_str}.mp4')
                background_dir = os.path.join(views_dir, view_str)
                if check_path_existence(background_dir, 'dir') != \
                        Existence.DirectoryExistNotEmpty:
                    continue
                cmd_str = f'python -u tools/visualization/visualize_kp2d.py \
                            {hd_path} {video_path} \
                            --background_dir {background_dir}'

                val = os.system(cmd_str) + val
    return val


def setup_parser():
    parser = argparse.ArgumentParser(description='')
    # input args
    parser.add_argument(
        '--dz_dir',
        type=str,
        help='Path to the directory named with dz.',
        default='')
    parser.add_argument(
        '--annots_path',
        type=str,
        help='Path to the annots.npy file',
        default=0)
    parser.add_argument(
        '--views', type=str, help='Which view, like 00_01_02.', default='all')
    # output args
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to the directory saving ' + 'all possible output files.',
        default='./default_output')
    parser.add_argument(
        '--vis_kp2d',
        action='store_true',
        help='If checked, visualize keypoints2d.',
        default=False)
    parser.add_argument(
        '--vis_kp3d',
        action='store_true',
        help='If checked, visualize keypoints3d.',
        default=False)
    parser.add_argument(
        '--vis_project',
        action='store_true',
        help='If checked, visualize keypoints2d.',
        default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    main(args)
