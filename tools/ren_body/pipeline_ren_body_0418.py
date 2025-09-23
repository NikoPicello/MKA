import argparse
import os
import sys
import io
# sys.path.append('/nvme/lufan/Projects/RenBody/Renbody_benchmark/keypoint_smpl/zoehuman')
sys.path.append('/nvme/lufan/Projects/zoehuman')
# sys.path.append('/nvme/lufan/Projects/RenBody/Renbody_benchmark/keypoint_smpl/zoehuman/tools')

import numpy as np

from zoehuman.utils.path_utils import Existence, check_path_existence
from petrel_client.client import Client
conf_path = '~/petreloss.conf'
client = Client(conf_path)


def main(args):
    # check input path
    # if check_path_existence(args.dz_dir,
    #                         'dir') != Existence.DirectoryExistNotEmpty:
    #     raise FileNotFoundError
    # if check_path_existence(args.annots_path, 'file') != Existence.FileExist:
    #     raise FileNotFoundError
    # check output path
    if client.contains(os.path.join(args.prev_save_dir, 'pose_2d')):
        raise FileExistsError
    os.makedirs(args.output_dir, exist_ok=True)
    exist_result = check_path_existence(args.output_dir, 'dir')
    if exist_result == Existence.MissingParent:
        raise FileNotFoundError
    elif exist_result == Existence.DirectoryNotExist:
        os.makedirs(args.output_dir)
    
    # det_res = run_detect(args)
    _ = run_triangulate(args)
    # if det_res == 0:
        # tri_res = run_triangulate(args) + det_res
    # if tri_res == 0:
    #     smplx_res = run_smplx(args) + tri_res
    # if smplx_res == 0:
    #     vis_res = run_vis_smplx(args) + smplx_res
    # print(vis_res)


def run_detect(args):
    # image dir not found in confluence, but seen in file system
    # not confirmed
    views_dir = os.path.join(args.dz_dir, 'image')
    pose_2d_dir = os.path.join(args.output_dir, 'pose_2d')
    os.makedirs(pose_2d_dir, exist_ok=True)
    ind_start_view = 0
    ind_end_view = 48
    # det model
    det_cfg = '/nvme/lufan/Projects/RenBody/Renbody_benchmark/keypoint_smpl/zoehuman/weights/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
    det_weight = '/nvme/lufan/Projects/RenBody/Renbody_benchmark/keypoint_smpl/zoehuman/weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    # pose model
    pose_cfg = '/nvme/lufan/Projects/RenBody/Renbody_benchmark/keypoint_smpl/zoehuman/weights/mmpose_wholebody_cfg/hrnet_w48_coco_wholebody_384x288_dark_plus.py'  # noqa: E501
    pose_weight = '/nvme/lufan/Projects/RenBody/Renbody_benchmark/keypoint_smpl/zoehuman/weights/hrnet_w48_wholebody_384x288_dark_plus-8701e1ce_20210426.pth'  # noqa: E501
    # never vis and never drop according to data 0417
    vis_option = '--visualize' if False else ''
    drop_option = '--drop_location 1 --drop_number 1' if False else ''
    cmd_str = f'python -u /nvme/lufan/Projects/RenBody/Renbody_benchmark/keypoint_smpl/zoehuman/tools/ren_body/detect_pose_ren_body_0418.py \
                {det_cfg} \
                {det_weight} \
                {pose_cfg} \
                {pose_weight} \
                --views_dir {views_dir} \
                --output_dir {pose_2d_dir} \
                --start_view {ind_start_view} \
                --end_view {ind_end_view} {vis_option} {drop_option}'

    val = os.system(cmd_str)
    net_start_view = 48
    net_end_view = 60
    cmd_str = f'python -u /nvme/lufan/Projects/RenBody/Renbody_benchmark/keypoint_smpl/zoehuman/tools/ren_body/detect_pose_ren_body_0418.py \
                {det_cfg} \
                {det_weight} \
                {pose_cfg} \
                {pose_weight} \
                --views_dir {views_dir} \
                --output_dir {pose_2d_dir} \
                --start_view {net_start_view} \
                --end_view {net_end_view} {vis_option}'

    val = os.system(cmd_str) + val
    return val


def run_triangulate(args):
    # human_data_dir = os.path.join(args.output_dir, 'pose_2d')
    # human_data_dir = os.path.join(args.output_dir, 'pose_2d_proj_merge_2')
    smc_id = '0013_01'
    human_data_dir = f'/nvme/lufan/Projects/zoehuman/tmp_out/zoehuman_ren_body_full_apose/smc_kpts_coco_annos/{smc_id}'
    output_dir =  f'/nvme/lufan/Projects/zoehuman/tmp_out/zoehuman_ren_body_full_apose/smc_kpts3d_coco_annos_v2/{smc_id}'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # output_dir = os.path.join(args.output_dir, 'pose_3d_merge')
    human_data_2d_list = sorted(os.listdir(human_data_dir))
    args.annots_path = f'/nvme/lufan/data/renbody/{smc_id}.smc'

    # if args.annots_path.startswith('s3://'):
    #     annots_f = io.BytesIO(client.get(args.annots_path))
    #     annots_dict = np.load(annots_f, allow_pickle=True).item()
    #     cam_dict = annots_dict['cams']
    #     annots_f.close()
    # else:
    #     annots_dict = np.load(args.annots_path, allow_pickle=True).item()
    #     cam_dict = annots_dict['cams']
    view_keys = []
    human_data_keys = ''
    cam_param_keys = []
    # images in view_key[n] must match camera in cam_param_index[n]
    for view_index in range(60):
        hd_name = f'human_data_{view_index:02d}.npz'
        view_str = f'{view_index:02d}'
        # if hd_name in human_data_2d_list and\
        #         view_str in cam_dict:
        if hd_name in human_data_2d_list:
            view_keys.append(view_str)
            cam_param_keys.append(view_str)
            human_data_keys += f'_{view_index:02d}'
    human_data_keys = human_data_keys[1:]
    cam_parameters_keys = human_data_keys
    vis_option = '--visualize --project' if args.visualize else ''
    # cmd_str = f'python -u tools/ren_body/triangulate_ren_body_0418.py \
    #             --human_data_dir {human_data_dir} \
    #             --human_data_prefix human_data_ \
    #             --human_data_keys {human_data_keys} \
    #             --cam_parameters_path {args.annots_path} \
    #             --cam_parameters_type ren_body_0418 \
    #             --cam_parameters_keys {cam_parameters_keys} \
    #             --keypoints_thr auto \
    #             --output_dir {output_dir} {vis_option}'
    cmd_str = f'python -u tools/ren_body/triangulate_ren_body_0418.py \
                --human_data_dir {human_data_dir} \
                --human_data_prefix human_data_ \
                --human_data_keys {human_data_keys} \
                --cam_parameters_path {args.annots_path} \
                --cam_parameters_type smc_new \
                --cam_parameters_keys {cam_parameters_keys} \
                --keypoints_thr auto \
                --output_dir {output_dir} {vis_option}'

    val = os.system(cmd_str)
    return val


def run_smplx(args):
    os.makedirs('logs', exist_ok=True)
    kp3d_human_data_path = os.path.join(args.output_dir, 'pose_3d', 'optim',
                                        'human_data_tri.npz')
    output_folder = os.path.join(args.output_dir, 'smplx_neutral')
    cmd_str = f'python mocap/smplify_renbody/smplify3d.py \
                --vis_smpl \
                --kp3d_path {kp3d_human_data_path} \
                --output_folder {output_folder} \
                --model "smplx" \
                --gender "neutral" \
                --src_convention "human_data" \
                --tgt_convention "openpose_118"'

    val = os.system(cmd_str)
    return val


def run_export_obj(args):
    smplx_path = os.path.join(args.output_dir, 'smplx_neutral',
                              'human_data_tri_smplx.npz')
    mesh_dir = os.path.join(args.output_dir, 'meshes')
    cmd_str = f'python -u tools/ren_body/export_smplx_obj_ren_body_0418.py \
                --smplx_path {smplx_path} \
                --output_dir {mesh_dir}'

    val = os.system(cmd_str)
    return val


def run_vis_obj(args):
    cam_key = '25'
    mesh_dir = os.path.join(args.output_dir, 'meshes')
    background_dir = os.path.join(args.dz_dir, 'image', cam_key)
    vis_dir = os.path.join(args.output_dir, 'meshes_overlay')

    cmd_str = f'python -u tools/ren_body/visualize_obj_ren_body_0418.py \
                --obj_dir {mesh_dir} \
                --background_dir {background_dir} \
                --annots_path {args.annots_path} \
                --cam_parameters_key {cam_key} \
                --vis_dir {vis_dir} '

    val = os.system(cmd_str)
    return val


def run_vis_smplx(args):
    cam_key = '25'
    smplx_dir = os.path.join(args.output_dir, 'smplx_neutral')
    smplx_path = os.path.join(smplx_dir, 'human_data_tri_smplx.npz')
    background_dir = os.path.join(args.dz_dir, 'image', cam_key)
    vis_path = os.path.join(smplx_dir,
                            f'human_data_tri_smplx_view{cam_key}.mp4')

    cmd_str = f'python -u tools/visualization/visualize_project_smplx.py \
                --smplx_path {smplx_path} \
                --annots_path {args.annots_path} \
                --view {cam_key} \
                --output_video_path {vis_path} \
                --background_dir {background_dir} '

    val = os.system(cmd_str)
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
    # output args
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to the directory saving ' + 'all possible output files.',
        default='./default_output')
    parser.add_argument(
        '--prev_save_dir',
        type=str,
        help='Path to the directory saving ' + 'all possible output files.',
        default='./default_output')
    parser.add_argument(
        '--save_dir',
        type=str,
        help='Path to the directory saving ' + 'all possible output files.',
        default='./default_output')
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='If checked, visualize poses.',
        default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    seq_id = '20220420/wuyinghao_m/wuyinghao_yf1_dz2'
    date = seq_id.split('/')[0]
    person_name = seq_id.split('/')[1]
    args.dz_dir = f's3://transfer/RenBody_luohuiwen/chengwei_RenBody/{seq_id}'
    args.annots_path = f's3://transfer/RenBody_luohuiwen/chengwei_RenBody/{date}/{person_name}/annots.npy'
    args.prev_save_dir = f's3://transfer/RenBody_luohuiwen/yinwanqi_zoehuman_ren_body_full/{seq_id}/smplx_xrmocap/'
    # args.prev_save_dir = f''
    args.save_dir = f's3://transfer/RenBody_lufan/zoehuman_ren_body_full/{seq_id}/smplx_xrmocap/'
    # args.output_dir = f'/nvme/lufan/ckpts/renbody/test/zoehuman_ren_body_full_kps/{seq_id}/smplx_xrmocap/'
    args.output_dir = f'/nvme/lufan/ckpts/renbody/test/zoehuman_ren_body_full_kps/{seq_id}/'

    main(args)
