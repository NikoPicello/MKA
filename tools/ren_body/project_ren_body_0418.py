import argparse
import json
import os
import cv2
import sys
from mmhuman3d.core.conventions.keypoints_mapping import KEYPOINTS_FACTORY, convert_kps
# sys.path.append('/nvme/lufan/Projects/RenBody/Renbody_benchmark/keypoint_smpl/zoehuman')
sys.path.append('/nvme/lufan/Projects/zoehuman')

import numpy as np
from mocap.multi_view_3d_keypoint.triangulate_scene import TriangulateScene
from zoehuman.core.visualization.visualize_keypoints2d import visualize_kp2d

from zoehuman.core.cameras.camera_parameters import CameraParameter
# from zoehuman.data.data_structures import SMCReader
from smc_reader_4k4d import SMCReader
from zoehuman.data.data_structures.human_data import HumanData
from zoehuman.utils.path_utils import (  # prevent yapf
    Existence, check_path_existence, check_path_suffix,
)

from petrel_client.client import Client
conf_path = '~/petreloss.conf'
client = Client(conf_path)

import io

def main(args):
    # check output path
    # exist_result = check_path_existence(args.output_dir, 'dir')
    # if exist_result == Existence.MissingParent:
    #     raise FileNotFoundError
    # elif exist_result == Existence.DirectoryNotExist:
    #     os.mkdir(args.output_dir)

    # load Camera Parameters
    # assert check_path_existence(args.cam_parameters_path, 'auto') == \
    #     Existence.FileExist
    # cam_parameters_key_list = args.cam_parameters_keys.split('_')
    print(args.cam_parameters_path)
    cam_parameters_key_list = [f'{index:02d}' for index in range(60)]
    camera_parameter_list = load_camera_parameters(args.cam_parameters_type,
                                                   args.cam_parameters_path,
                                                   cam_parameters_key_list)
    
    # smc_id = '0013_01'
    # cam_parameters_path = f'/nvme/lufan/data/renbody/{smc_id}.smc'
    # camera_parameter_list = load_camera_parameters('smc_new',
    #                                                cam_parameters_path,
    #                                                cam_parameters_key_list)
    # print(cam_parameters_key_list)
    
    scene = TriangulateScene(camera_parameter_list, 'auto')

    if args.human_data_path.startswith('s3://'):
        human_data_path = io.BytesIO(client.get(args.human_data_path))
        human_data_3d = HumanData.fromfile(human_data_path)
        human_data_path.close()
    else:
        human_data_path = args.human_data_path
        human_data_3d = HumanData.fromfile(human_data_path)
    
    # human_data_3d = HumanData.fromfile(f'/nvme/lufan/Projects/zoehuman/tmp_out/zoehuman_ren_body_full_apose/smc_kpts3d_coco_annos_v1/{smc_id}/no_optim/human_data_tri_fused.npz')
    # project keypoints3d back to keypoints2d(HumanData)
    projected_human_data_list = scene.project(human_data_3d)
    # args.output_dir = f'/nvme/lufan/Projects/zoehuman/tmp_out/zoehuman_ren_body_full_apose/src_pose_2d_new/{smc_id}'
    os.makedirs(args.output_dir, exist_ok=True)
    project_dir = args.output_dir
    # smc_reader = SMCReader(cam_parameters_path)
    # if check_path_existence(project_dir, 'dir') == \
    #         Existence.DirectoryNotExist:
    #     os.mkdir(project_dir)
    for index, human_data_2d in enumerate(projected_human_data_list):
        print('Processing ', index)
        human_data_2d_coco = convert_kps(human_data_2d['keypoints2d'], 'human_data', 'coco_wholebody')
        tmp_human_data_path = os.path.join(project_dir,
                                           f'human_data_{index:02d}.npz')
        print(human_data_2d['keypoints2d'].shape)
        human_data_2d.dump(tmp_human_data_path)

        # random_idx = np.arange(start=0, stop=len(frame_list), step=30)
        random_idx = np.arange(start=0, stop=150, step=30)
        # random_idx = np.arange(start=0, stop=10, step=1)
        # start_idx = 140
        if index < 48:
            cam_group = 'Camera_5mp'
        else:
            cam_group = 'Camera_12mp'

        # image_array = []
        # for idx in random_idx:
        #     img = smc_reader.get_img(Camera_group=cam_group, Camera_id=str(index), Image_type='color', Frame_id=str(idx))
        #     image_array.append(img)
        # image_array = np.stack(image_array, axis=0)

        image_path = f'{args.human_image_path}/{index:02d}'
        img_fns = sorted(list(client.list(image_path)))
        frame_list = [os.path.join(image_path, img_fn) for img_fn in img_fns]
        # random_idx = np.random.choice(len(frame_list), 5, replace=False)
        random_idx = np.arange(start=0, stop=len(frame_list), step=30)
        frame_list = [frame_list[idx] for idx in random_idx]
        image_array = []
        for frame in frame_list:
            if frame.startswith('s3://'):
                img_bytes = client.get(frame)
                img_mem_view = memoryview(img_bytes)
                img_array = np.frombuffer(img_mem_view, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                image_array.append(img)
        image_array = np.stack(image_array, axis=0)

        img_hw = image_array.shape[1:3]
        img_h_down = img_hw[0] // 4
        img_w_down = img_hw[1] // 4
    
        # visualize
        if args.visualize:
            if args.vis_type == 'image':
                vis_path = os.path.join(args.output_dir,
                                        f'keypoints2d_{index:02d}_vis')
            elif args.vis_type == 'video':
                vis_path = os.path.join(args.output_dir,
                                        f'keypoints2d_{index:02d}_vis.mp4')
            else:
                raise KeyError(f'Wrong visualization type: {args.vis_type}')
            # visualize 2D keypoints on source frame
            # print(human_data_2d['keypoints2d'].shape)
            # print(human_data_2d['keypoints2d_mask'].shape)
            visualize_kp2d(
                kp2d=human_data_2d['keypoints2d'][random_idx],
                output_path=vis_path,
                data_source='human_data',
                frame_list=None,
                image_array=image_array,
                mask=human_data_2d['keypoints2d_mask'],
                overwrite=True,
                resolution=(img_h_down, img_w_down),
                disable_tqdm=True)
            
    return 0


def load_camera_parameters(cam_parameters_type, cam_parameters_path,
                           cam_parameters_key_list):
    camera_para_list = []
    if cam_parameters_type == 'smc':
        assert check_path_suffix(cam_parameters_path, ['.smc']) is True
        smc_reader = SMCReader(cam_parameters_path)
        for camera_key in cam_parameters_key_list:
            temp_camera_parameter = CameraParameter(name=camera_key)
            temp_camera_parameter.load_kinect_from_smc(smc_reader,
                                                       int(camera_key))
            camera_para_list.append(temp_camera_parameter)
    elif cam_parameters_type == 'smc_new':
        smc_reader = SMCReader(cam_parameters_path)
        for camera_key in cam_parameters_key_list:
            if int(camera_key) < 48:
                cam_id = 'Camera_5mp'
            else:
                cam_id = 'Camera_12mp'
            # print(smc_reader.get_Calibration_all()[cam_id].keys())
            camera_key_ = str(int(camera_key))
            calib = smc_reader.get_Calibration_all()[cam_id][camera_key_]
            
            temp_camera_parameter = CameraParameter(name=camera_key)
            camera_para_dict = {
                'RT':
                calib['RT'].reshape(1, 4, 4),
                'K': calib['K'].reshape(1, 3, 3),
            }
            temp_camera_parameter.load_from_lightstage(camera_para_dict, 0)
            dist_array = calib['D']
            dist_keys = [
                'k1',
                'k2',
                'p1',
                'p2',
                'k3',
            ]
            for dist_index, dist_key in enumerate(dist_keys):
                temp_camera_parameter.set_value(dist_key,
                                                float(dist_array[dist_index]))
            camera_para_list.append(temp_camera_parameter)
    elif cam_parameters_type == 'chessboard':
        assert check_path_suffix(cam_parameters_path, ['.json']) is True
        camera_para_json_dict = json.load(open(cam_parameters_path))
        for camera_key in cam_parameters_key_list:
            temp_camera_parameter = CameraParameter(name=camera_key)
            temp_camera_parameter.load_from_chessboard(
                camera_para_json_dict[camera_key], camera_key)
            camera_para_list.append(temp_camera_parameter)
    elif cam_parameters_type == 'dump':
        assert check_path_suffix(cam_parameters_path, []) is True
        for camera_key in cam_parameters_key_list:
            temp_camera_parameter_path = os.path.join(
                cam_parameters_path, f'camera_parameter_{camera_key}.json')
            temp_camera_parameter_dict = \
                json.load(open(temp_camera_parameter_path))
            temp_camera_parameter = CameraParameter(name=camera_key)
            temp_camera_parameter.load_from_dict(temp_camera_parameter_dict)
            camera_para_list.append(temp_camera_parameter)
    elif cam_parameters_type == 'lightstage':
        assert check_path_suffix(cam_parameters_path, ['.npy']) is True
        camera_para_dict = np.load(
            cam_parameters_path, allow_pickle=True).item()['cams']
        for camera_key in cam_parameters_key_list:
            temp_camera_parameter = CameraParameter(name=camera_key)
            temp_camera_parameter.load_from_lightstage(camera_para_dict,
                                                       int(camera_key))
            camera_para_list.append(temp_camera_parameter)
    elif cam_parameters_type == 'ren_body_0418':
        assert check_path_suffix(cam_parameters_path, ['.npy']) is True
        if cam_parameters_path.startswith('s3://'):
            f = io.BytesIO(client.get(cam_parameters_path))
            ren_body_0418_cam_dict = np.load(
                f,
                allow_pickle=True).item()['cams']
            f.close()
        else:
            ren_body_0418_cam_dict = np.load(
                cam_parameters_path, allow_pickle=True).item()['cams']
        # print(ren_body_0418_cam_dict.keys())
        for camera_key in cam_parameters_key_list:
            temp_camera_parameter = CameraParameter(name=camera_key)
            camera_para_dict = {
                'RT':
                ren_body_0418_cam_dict[camera_key]['RT'].reshape(1, 4, 4),
                'K': ren_body_0418_cam_dict[camera_key]['K'].reshape(1, 3, 3),
            }
            temp_camera_parameter.load_from_lightstage(camera_para_dict, 0)
            dist_array = ren_body_0418_cam_dict[camera_key]['D']
            dist_keys = [
                'k1',
                'k2',
                'p1',
                'p2',
                'k3',
            ]
            for dist_index, dist_key in enumerate(dist_keys):
                temp_camera_parameter.set_value(dist_key,
                                                float(dist_array[dist_index]))
            camera_para_list.append(temp_camera_parameter)
    else:
        raise KeyError
    return camera_para_list


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Triangulate multi-view keypoints2d to keypoints3d' +
        ' powered by aniposelib')
    # input args
    parser.add_argument(
        '--human_data_path',
        type=str,
        help='Path to keypoints2d human_data file.',
        default='')
    parser.add_argument(
        '--cam_parameters_path',
        type=str,
        help='Path to camera parameters.',
        default='')
    parser.add_argument(
        '--cam_parameters_type',
        type=str,
        help='Type of camera parameters.',
        choices=['smc', 'chessboard', 'dump', 'lightstage', 'ren_body_0418'],
        default='ren_body_0418')
    parser.add_argument(
        '--cam_parameters_keys',
        type=str,
        help='Keys of selected camera parameters file,' + 'split by \'_\' .',
        default='')
    # output args
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to the directory saving projected human data.',
        default='./default_output')
    parser.add_argument(
        '--visualize',
        action='store_true')
    parser.add_argument(
        '--vis_type',
        type=str,
        default='video')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    args.visualize = True
    args.vis_type = 'image'
    # main(args)
    # all_list_fn = '/nvme/all_400.txt'
    # with open(all_list_fn, 'r') as f:
    #     all_list = f.readlines()
    # all_200_list = all_list[:400]
    # valid_num = 0
    # from tqdm import tqdm
    # for seq_id in tqdm(all_200_list):
    #     seq_id = seq_id.strip('\n')
    #     # seq_id = '20220824/siqigong_m/siqigong_yf3_dz10'
    #     date = seq_id.split('/')[0]
    #     person_name = seq_id.split('/')[1]
    #     args.cam_parameters_path = f's3://transfer/RenBody_luohuiwen/chengwei_RenBody/{date}/{person_name}/annots.npy'
    #     args.human_data_path = f's3://transfer/RenBody_luohuiwen/yinwanqi_zoehuman_ren_body/{seq_id}/pose_3d/optim/human_data_tri.npz'
    #     args.output_dir = f'/nvme/lufan/ckpts/renbody/test/zoehuman_ren_body_full/{seq_id}/pose_2d/'
    #     args.human_image_path = f's3://transfer/RenBody_luohuiwen/chengwei_RenBody/{seq_id}/image'
    #     args.visualize = True
    #     args.vis_type = 'image'
    #     main(args)
    #     os.system(f"/nvme/lufan/local/bin/aws --endpoint-url=http://10.140.27.254:80 s3 cp {args.output_dir} {args.output_dir.replace('/nvme/lufan/ckpts/renbody/test/','s3://transfer/RenBody_lufan/')} --recursive")
    #     os.system(f"rm -rf {args.output_dir}")
    seq_id = '20221015/fengdeli_m/fengdeli_yf1_dz3'
    date = seq_id.split('/')[0]
    person_name = seq_id.split('/')[1]
    args.cam_parameters_path = f's3://transfer/RenBody_luohuiwen/chengwei_RenBody/{date}/{person_name}/annots.npy'
    args.human_data_path = f's3://renbody/data_part3/smpl_results/{seq_id}/smplify_final/pose_3d/optim/human_data_tri.npz'
    args.output_dir = f'/mnt/petrelfs/lufan/zoehuman/tmp_out/results/{seq_id}/pose_2d/'
    args.human_image_path = f's3://transfer/RenBody_luohuiwen/chengwei_RenBody/{seq_id}/image'
    args.visualize = True
    args.vis_type = 'image'
    main(args)
