import argparse
import os

from mocap.multi_view_3d_keypoint.mmpose_helper.top_down_pose_det import (  # noqa:E501
    PoseDetector, convert_results_to_human_data,
)

from zoehuman.core.visualization.visualize_keypoints2d import visualize_kp2d
from zoehuman.utils import str_or_empty
from zoehuman.utils.path_utils import Existence, check_path_existence


def main(args):
    # check output path
    exist_result = check_path_existence(args.output_dir, 'dir')
    if exist_result == Existence.MissingParent:
        raise FileNotFoundError
    elif exist_result == Existence.DirectoryNotExist:
        os.mkdir(args.output_dir)
    # init models
    detector_kwargs, pose_model_kwargs = get_model_files(args)
    detector = PoseDetector(detector_kwargs, pose_model_kwargs)
    pose_data_source = detector.get_data_source_name()
    # list all views
    file_list = os.listdir(args.views_dir)
    file_list.sort()
    view_dict = {}
    for file_name in file_list:
        try:
            view_index = int(file_name)
            if view_index >= 0 and view_index <= 47:
                image_dir = os.path.join(args.views_dir, file_name)
                if check_path_existence(image_dir, 'dir') == \
                        Existence.DirectoryExistNotEmpty:
                    view_dict[file_name] = image_dir
        except ValueError:
            continue
    for view_str, image_dir in view_dict.items():
        # infer one view
        print(f'Inferring view {view_str}:')
        frame_list = []
        frame_dict = {}
        file_list = os.listdir(image_dir)
        file_list.sort()
        for file_name in file_list:
            if file_name.lower().endswith('.jpg') or \
                    file_name.lower().endswith('.png'):
                frame_path = os.path.join(image_dir, file_name)
                frame_list.append(frame_path)
                frame_dict[file_name] = frame_path
        result_dict = \
            detector.infer_frames(
                frame_dict, multi_person=False)

        # save HumanData
        human_data = convert_results_to_human_data(
            result_dict,
            bbox_threshold=0.0,
            data_source=pose_data_source,
            data_destination='human_data')
        human_data.compress_keypoints_by_mask()
        human_data.dump(
            os.path.join(args.output_dir, f'human_data_{view_str}.npz'))

        # visualize
        if args.visualize:
            if args.vis_type == 'image':
                vis_path = os.path.join(args.output_dir,
                                        f'keypoints2d_{view_str}_vis')
            elif args.vis_type == 'video':
                vis_path = os.path.join(args.output_dir,
                                        f'keypoints2d_{view_str}_vis.mp4')
            else:
                raise KeyError(f'Wrong visualization type: {args.vis_type}')
            # visualize 2D keypoints on source frame
            visualize_kp2d(
                kp2d=human_data['keypoints2d'],
                output_path=vis_path,
                data_source='human_data',
                frame_list=frame_list,
                mask=human_data['keypoints2d_mask'])
    return 0


def get_model_files(args):
    detector_kwargs = {
        'config': args.det_config,
        'checkpoint': args.det_checkpoint,
    }
    pose_model_kwargs = {
        'config': args.pose_config,
        'checkpoint': args.pose_checkpoint,
    }
    return detector_kwargs, pose_model_kwargs


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Top down detection and pose estimation' +
        ' powered by mmdet and mmpose. For lightstage data.')
    # model args
    parser.add_argument(
        'det_config', help='Config file for detection', type=str)
    parser.add_argument(
        'det_checkpoint', help='Checkpoint file for detection', type=str)
    parser.add_argument('pose_config', help='Config file for pose', type=str)
    parser.add_argument(
        'pose_checkpoint', help='Checkpoint file for pose', type=str)
    # input args
    parser.add_argument(
        '--views_dir',
        type=str_or_empty,
        help='Path to directory with 48 views.',
        default='')
    # output args
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to the directory saving ' + 'all possible output files.',
        default='./default_output')
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='If checked, visualize poses.',
        default=False)
    parser.add_argument(
        '--vis_type',
        type=str,
        help='Whether visualize pose into image or video.',
        choices=['image', 'video'],
        default='video')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    main(args)
