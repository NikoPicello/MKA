import argparse
import json
import os
import io

import sys
# sys.path.append('/nvme/lufan/Projects/RenBody/Renbody_benchmark/keypoint_smpl/zoehuman')
sys.path.append('/mnt/petrelfs/lufan/zoehuman')

import numpy as np
from mocap.multi_view_3d_keypoint.triangulate_scene import TriangulateScene

from zoehuman.core.cameras.camera_parameters import CameraParameter
from zoehuman.core.visualization.visualize_keypoints3d import visualize_kp3d
# from zoehuman.data.data_structures import SMCReader
from smc_reader_4k4d import SMCReader
from zoehuman.data.data_structures.human_data import HumanData
from zoehuman.utils.path_utils import (  # prevent yapf
    Existence, check_path_existence, check_path_suffix,
)

from petrel_client.client import Client
conf_path = '~/petreloss.conf'
client = Client(conf_path)


def main(args):
    # check output path
    # exist_result = check_path_existence(args.output_dir, 'dir')
    # if exist_result == Existence.MissingParent:
    #     raise FileNotFoundError
    # elif exist_result == Existence.DirectoryNotExist:
    #     os.mkdir(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    # load HumanData
    human_data_list = []
    human_data_key_list = args.human_data_keys.split('_')
    for human_data_key in human_data_key_list:
        tmp_human_data_path = os.path.join(
            args.human_data_dir,
            f'{args.human_data_prefix}{human_data_key}.npz')
        if tmp_human_data_path.startswith('s3://'):
            tmp_human_data_path = io.BytesIO(client.get(tmp_human_data_path))

        # print(tmp_human_data_path)
        # data_ = np.load('/nvme/lufan/ckpts/renbody/test/zoehuman_ren_body_full_kps/20220420/wuyinghao_m/wuyinghao_yf1_dz2/smplx_xrmocap/pose_2d/human_data_00.npz')
        # print(data_['keypoints2d'].shape)
        # if check_path_existence(tmp_human_data_path, 'file') != \
        #         Existence.FileExist:
        #     raise FileNotFoundError(
        #         f'HumanData file not found: {tmp_human_data_path}')
        tmp_human_data = HumanData.fromfile(tmp_human_data_path)
        print(tmp_human_data['keypoints2d'].shape)
        human_data_list.append(tmp_human_data)
    assert len(human_data_list) >= 2, 'HumanData fewer than 2!'

    # load Camera Parameters
    # assert check_path_existence(args.cam_parameters_path, 'auto') == \
    #     Existence.FileExist
    cam_parameters_key_list = args.cam_parameters_keys.split('_')
    camera_parameter_list = load_camera_parameters(args.cam_parameters_type,
                                                   args.cam_parameters_path,
                                                   cam_parameters_key_list)
    assert len(camera_parameter_list) == len(human_data_list),\
        'numbers of cameras and HumanData do not match'
    if args.keypoints_thr != 'auto':
        args.keypoints_thr = float(args.keypoints_thr)
    scene = TriangulateScene(camera_parameter_list, args.keypoints_thr)

    # result dict
    result_dict = {'optim': None, 'no_optim': {}}
    # triangulate
    keypoints3d = scene.triangulate(human_data_list)
    result_dict['no_optim']['keypoints3d'] = keypoints3d
    if not args.disable_optim:
        # optim with bone constraints
        keypoints3d = scene.optim(
            human_data_list, keypoints3d=keypoints3d, constraints=None)
        result_dict['optim'] = {
            'keypoints3d': keypoints3d,
        }
    for key in result_dict.keys():
        if result_dict[key] is not None:
            keypoints3d = result_dict[key]['keypoints3d']
            human_data_3d = \
                TriangulateScene.convert_result_to_human_data(
                    keypoints3d, human_data_list[0]['keypoints2d_mask'])
            result_dict[key]['human_data'] = human_data_3d
            result_dict[key]['human_data_path'] = \
                os.path.join(
                    args.output_dir, key, 'human_data_tri.npz')
            if check_path_existence(
                    os.path.join(args.output_dir, key), 'dir') == \
                    Existence.DirectoryNotExist:
                os.mkdir(os.path.join(args.output_dir, key))
            human_data_3d.dump(result_dict[key]['human_data_path'])

    # project keypoints3d back to keypoints2d(HumanData)
    if args.project:
        for key in result_dict.keys():
            if result_dict[key] is not None:
                human_data_3d = result_dict[key]['human_data']
                projected_human_data_list = scene.project(human_data_3d)
                project_dir = os.path.join(args.output_dir, key, 'project')
                if check_path_existence(project_dir, 'dir') == \
                        Existence.DirectoryNotExist:
                    os.mkdir(project_dir)
                for index, human_data_key in enumerate(human_data_key_list):
                    tmp_human_data_path = os.path.join(
                        project_dir, f'human_data_{human_data_key}.npz')
                    np.savez_compressed(tmp_human_data_path,
                                        **(projected_human_data_list[index]))

    # visualize
    if args.visualize:
        for key in result_dict.keys():
            if result_dict[key] is not None:
                human_data_3d = result_dict[key]['human_data']
                vis3d_path = os.path.join(args.output_dir, key,
                                          'human_data_tri_vis.mp4')
                visualize_kp3d(
                    human_data_3d['keypoints3d'],
                    vis3d_path,
                    data_source='human_data',
                    mask=human_data_3d['keypoints3d_mask'])
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
            ren_body_0418_cam_dict = np.load(
                io.BytesIO(client.get(cam_parameters_path)),
                allow_pickle=True).item()['cams']
        else:
            ren_body_0418_cam_dict = np.load(
                cam_parameters_path, allow_pickle=True).item()['cams']
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
        '--human_data_dir',
        type=str,
        help='Path to human_data directory.',
        default='')
    parser.add_argument(
        '--human_data_prefix',
        type=str,
        help='Prefix string of human_data files.',
        default='human_data_')
    parser.add_argument(
        '--human_data_keys',
        type=str,
        help='Keys of selected human_data file,' + 'split by \'_\' .',
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
        choices=['smc', 'chessboard', 'dump', 'lightstage', 'ren_body_0418', 'smc_new'],
        default='ren_body_0418')
    parser.add_argument(
        '--cam_parameters_keys',
        type=str,
        help='Keys of selected camera parameters file,' + 'split by \'_\' .',
        default='')
    # processing args
    parser.add_argument(
        '--keypoints_thr',
        type=str,
        help='Threshold of keypoint scores.',
        default='auto')
    parser.add_argument(
        '--disable_optim',
        action='store_true',
        help='If checked, disable scene.optim().',
        default=False)
    # output args
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to the directory saving ' + 'all possible output files.',
        default='./default_output')
    parser.add_argument(
        '--project',
        action='store_true',
        help='If checked, visualize poses.',
        default=False)
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='If checked, visualize poses.',
        default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    main(args)
