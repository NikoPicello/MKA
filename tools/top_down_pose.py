import argparse
import glob
import os
import shutil

from mocap.multi_view_3d_keypoint.mmpose_helper.top_down_pose_det import (  # noqa:E501
    PoseDetector, convert_results_to_human_data,
)

from zoehuman.core.visualization.visualize_keypoints2d import (  # noqa:E501
    visualize_kp2d, visualize_kp2d_multiperson,
)
from zoehuman.utils import str_or_empty
from zoehuman.utils.ffmpeg_utils import video_to_images
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
    # infer
    frame_list = []
    if args.input_type == 'image' and \
            check_path_existence(args.image_dir, 'dir') == \
            Existence.FileExist:
        frame_dict = {}
        file_list = os.listdir(args.image_dir)
        file_list.sort()
        for file_name in file_list:
            if file_name.lower().endswith('.jpg') or \
                    file_name.lower().endswith('.png'):
                frame_path = os.path.join(args.image_dir, file_name)
                frame_list.append(frame_path)
                frame_dict[file_name] = frame_path
        result_dict = \
            detector.infer_frames(
                frame_dict, multi_person=args.det_multi_person)
    elif args.input_type == 'video' and \
            check_path_existence(args.video_path, 'file') == \
            Existence.FileExist:
        result_dict = \
            detector.infer_video(
                args.video_path, multi_person=args.det_multi_person)
    else:
        raise FileNotFoundError(
            'Please provide image_dir or video_path as input')

    # save HumanData
    if not args.det_multi_person:
        human_data = convert_results_to_human_data(
            result_dict,
            bbox_threshold=args.bbox_thr,
            data_source=pose_data_source)
        human_data.compress_keypoints_by_mask()
        human_data.dump(
            os.path.join(args.output_dir, 'keypoints2d_result.npz'))
    else:
        max_person_number = get_max_person_number(result_dict, args.bbox_thr)
        human_data_list = []
        for person_index in range(max_person_number):
            human_data = convert_results_to_human_data(
                result_dict,
                bbox_threshold=args.bbox_thr,
                person_index=person_index,
                data_source=pose_data_source)
            human_data_list.append(human_data)
            human_data.compress_keypoints_by_mask()
            human_data.dump(
                os.path.join(args.output_dir,
                             f'keypoints2d_result_{person_index:02d}.npz'))

    # visualize
    if args.visualize:
        temp_frame_dir = None
        if args.input_type == 'video' and len(args.video_path) > 0:
            temp_frame_dir = os.path.join(args.output_dir, 'src_frames')
            os.makedirs(temp_frame_dir, exist_ok=False)
            video_to_images(
                input_path=args.video_path,
                output_folder=temp_frame_dir,
                img_format='%06d.png')
            frame_list = glob.glob(os.path.join(temp_frame_dir, '*.png'))
            frame_list.sort()
        if args.vis_type == 'image':
            vis_path = os.path.join(args.output_dir, 'keypoints2d_result_vis')
        elif args.vis_type == 'video':
            vis_path = os.path.join(args.output_dir,
                                    'keypoints2d_result_vis.mp4')
        else:
            raise KeyError(f'Wrong visualization type: {args.vis_type}')
        # visualize 2D keypoints on source frame
        if not args.det_multi_person:
            visualize_kp2d(
                kp2d=human_data['keypoints2d'],
                output_path=vis_path,
                data_source='smplx',
                frame_list=frame_list,
                mask=human_data['keypoints2d_mask'])
        else:
            kp2d_list = []
            for human_data in human_data_list:
                kp2d_list.append(human_data['keypoints2d'])
                vis_mask = human_data['keypoints2d_mask']
            visualize_kp2d_multiperson(
                kp2d_list=kp2d_list,
                output_path=vis_path,
                frame_list=frame_list,
                data_source='smplx',
                mask=vis_mask)
        if temp_frame_dir is not None:
            shutil.rmtree(temp_frame_dir)
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


def get_max_person_number(result_dict, bbox_thr=0.0):
    max_person_number = 0
    for result_dict_list in result_dict.values():
        cur_person_number = 0
        for person_index, result_dict in enumerate(result_dict_list):
            bbox_score = result_dict_list[person_index]['bbox'][4]
            if bbox_score >= bbox_thr:
                cur_person_number += 1
        if cur_person_number > max_person_number:
            max_person_number = cur_person_number
    return max_person_number


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Top down detection and pose estimation' +
        ' powered by mmdet and mmpose')
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
        '--image_dir',
        type=str_or_empty,
        help='Path to input image directory.',
        default='')
    parser.add_argument(
        '--video_path',
        type=str_or_empty,
        help='Path to input video.',
        default='')
    parser.add_argument(
        '--input_type',
        type=str,
        help='Whether take input from image or video.',
        choices=['image', 'video'],
        default='image')
    # output args
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to the directory saving ' + 'all possible output files.',
        default='./default_output')
    parser.add_argument(
        '--bbox_thr',
        type=float,
        help='Threshold of bbox scores.',
        default=0.0)
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
    parser.add_argument(
        '--det_multi_person',
        action='store_true',
        help='If checked, save multi-person result.',
        default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    main(args)
