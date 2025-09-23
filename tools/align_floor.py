import argparse
import json
import os

from zoehuman.core.cameras.camera_parameters import DepthCameraParameter
from zoehuman.core.visualization.visualize_keypoints3d import visualize_kp3d
from zoehuman.data.data_structures import SMCReader
from zoehuman.data.data_structures.human_data import HumanData
from zoehuman.utils.path_utils import (  # prevent yapf isort conflict
    Existence, check_path_existence, check_path_suffix,
)


def main(args):
    # check output path
    exist_result = check_path_existence(args.output_path, 'file')
    if exist_result == Existence.MissingParent:
        raise FileNotFoundError
    elif exist_result == Existence.FileExist and not args.replace:
        raise FileExistsError(f'File already exists at {args.output_path}.' +
                              'Set --repalce to overwrite.')
    # load HumanData
    if check_path_existence(args.input_path, 'file') != \
            Existence.FileExist:
        raise FileNotFoundError(f'HumanData file not found: {args.input_path}')
    human_data = HumanData.fromfile(args.input_path)

    # load Camera Parameters
    assert check_path_existence(args.cam_parameters_path, 'auto') == \
        Existence.FileExist
    depth_camera_parameter = load_camera_parameters(args.cam_parameters_type,
                                                    args.cam_parameters_path,
                                                    args.cam_parameters_key)
    depth_camera_parameter.setup_transform()

    # align
    src_kp3d = human_data.get_raw_value('keypoints3d')
    _, kpt_number, dim = src_kp3d.shape
    aligned_points = \
        depth_camera_parameter.transform_points_cam_to_floor(
            src_kp3d.reshape(-1, dim)
        )
    aligned_kp3d = aligned_points.reshape(-1, kpt_number, dim)
    # save
    human_data.set_raw_value('keypoints3d', aligned_kp3d)
    if not human_data.check_keypoints_compressed():
        try:
            human_data.compress_keypoints_by_mask()
        except KeyError:
            pass
    human_data.dump(args.output_path)

    # visualize
    if args.visualize:
        output_path = args.output_path.replace('\\', '/')
        output_dir = output_path.rsplit('/', 1)[0]
        vis3d_path = os.path.join(output_dir, 'align_floor_vis.mp4')
        visualize_kp3d(
            human_data['keypoints3d'].copy(),
            vis3d_path,
            data_source='human_data',
            mask=human_data['keypoints3d_mask'].copy())
    return 0


def load_camera_parameters(cam_parameters_type, cam_parameters_path,
                           cam_parameters_key):
    ret_dep_cam_parameter = DepthCameraParameter()
    if cam_parameters_type == 'smc':
        assert check_path_suffix(cam_parameters_path, ['.smc']) is True
        smc_reader = SMCReader(cam_parameters_path)
        ret_dep_cam_parameter.load_kinect_from_smc(smc_reader,
                                                   int(cam_parameters_key))
    elif cam_parameters_type == 'chessboard':
        assert check_path_suffix(cam_parameters_path, ['.json']) is True
        camera_para_json_dict = json.load(open(cam_parameters_path))
        ret_dep_cam_parameter.load_from_chessboard(
            camera_para_json_dict[cam_parameters_key], cam_parameters_key)
    elif cam_parameters_type == 'dump':
        assert check_path_suffix(cam_parameters_path, ['.json']) is True
        ret_dep_cam_parameter.load(cam_parameters_path)
    else:
        raise KeyError
    return ret_dep_cam_parameter


def setup_parser():
    parser = argparse.ArgumentParser(description='Align keypoints3d to floor' +
                                     ' recorded in calibration.')
    # input args
    parser.add_argument(
        '--input_path',
        type=str,
        help='Path to the input human_data file.',
        default='')
    parser.add_argument(
        '--cam_parameters_path',
        type=str,
        help='Path to parameters of a depth camera.',
        default='')
    parser.add_argument(
        '--cam_parameters_type',
        type=str,
        help='Type of camera parameters.',
        choices=['smc', 'chessboard', 'dump'],
        default='smc')
    parser.add_argument(
        '--cam_parameters_key',
        type=str,
        help='Key of the selected depth camera.' +
        'Ignored when cam_parameters_type == dump',
        default='')
    # output args
    parser.add_argument(
        '--output_path',
        type=str,
        help='Path to the output human_data file.',
        default='')
    parser.add_argument(
        '--replace',
        action='store_true',
        help='If checked, replace the file at output_path.',
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
