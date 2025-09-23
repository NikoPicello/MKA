import argparse
import os
import numpy as np
import cv2
import sys
sys.path.append('/nvme/lufan/Projects/RenBody/Renbody_benchmark/keypoint_smpl/zoehuman')

from mocap.multi_view_3d_keypoint.mmpose_helper.top_down_pose_det import (  # noqa:E501
    PoseDetector, convert_results_to_human_data,
)

from zoehuman.core.visualization.visualize_keypoints2d import visualize_kp2d
from zoehuman.utils import str_or_empty
from zoehuman.utils.path_utils import Existence, check_path_existence

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
    # init models
    detector_kwargs, pose_model_kwargs = get_model_files(args)
    detector = PoseDetector(detector_kwargs, pose_model_kwargs)
    pose_data_source = detector.get_data_source_name()
    # list all views
    print(args.views_dir)
    if args.views_dir.startswith('s3://'):
        file_list = list(client.list(args.views_dir))
    else:
        file_list = os.listdir(args.views_dir)
    print(file_list)
    file_list.sort()
    view_dict = {}
    # end_view = 999 if args.end_view < 0 else args.end_view
    for file_name in file_list:
        try:
            view_index = int(file_name.strip('/'))
            if view_index >= args.start_view:
                image_dir = os.path.join(args.views_dir, file_name)
                img_fns = list(client.list(image_dir))
                if check_path_existence(image_dir, 'dir') == \
                        Existence.DirectoryExistNotEmpty:
                    view_dict[file_name] = image_dir
                elif client.contains(os.path.join(image_dir, img_fns[0])):
                    view_dict[file_name] = image_dir
        except ValueError:
            continue
    for view_str, image_dir in view_dict.items():
        # infer one view
        view_str = view_str.strip('/')
        print(f'Inferring view {view_str}:')
        frame_list = []
        frame_dict = {}
        if image_dir.startswith('s3://'):
            file_list = sorted(list(client.list(image_dir)))
        else:
            file_list = sorted(os.listdir(image_dir))
        if args.drop_number > 0:
            if args.drop_location == 1:
                file_list = file_list[:-args.drop_number]
            elif args.drop_location == 0:
                file_list = file_list[args.drop_number:]
            else:
                raise ValueError
        for file_name in file_list:
            if file_name.lower().endswith('.jpg') or \
                    file_name.lower().endswith('.png'):
                frame_path = os.path.join(image_dir, file_name)
                frame_list.append(frame_path)
                if frame_path.startswith('s3://'):
                    img_bytes = client.get(frame_path)
                    img_mem_view = memoryview(img_bytes)
                    img_array = np.frombuffer(img_mem_view, np.uint8)
                    frame_path = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
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
        help='Path to directory with 60 views.',
        default='')
    parser.add_argument(
        '--start_view', type=int, help='From which view to start.', default=0)
    parser.add_argument(
        '--end_view', type=int, help='Till which view to stop.', default=-1)
    # drop frame args
    parser.add_argument(
        '--drop_location',
        type=int,
        help='0 for drop head, 1 for drop tail',
        default=1)
    parser.add_argument(
        '--drop_number',
        type=int,
        help='How many frames will be dropped.',
        default=0)
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
