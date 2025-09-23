import argparse
import os

from zoehuman.core.visualization.visualize_keypoints2d import visualize_kp2d
from zoehuman.data.data_structures.human_data import HumanData
from zoehuman.utils import str_or_empty
from zoehuman.utils.path_utils import Existence, check_path_existence


def main(args):
    human_data_2d = HumanData.fromfile(args.human_data_path)
    frame_dir = args.background_dir
    if len(frame_dir) > 0 and \
            check_path_existence(frame_dir, 'dir') == \
            Existence.DirectoryExistNotEmpty:
        frame_list = os.listdir(frame_dir)
        frame_list.sort()
        abs_frame_list = [
            os.path.join(frame_dir, frame_name) for frame_name in frame_list
        ]
    else:
        abs_frame_list = None
    visualize_kp2d(
        human_data_2d['keypoints2d'].copy(),
        output_path=args.output_video_path,
        frame_list=abs_frame_list,
        data_source='human_data',
        mask=human_data_2d['keypoints2d_mask'].copy())


def setup_parser():
    parser = argparse.ArgumentParser(
        description='A tool for keypoints2d(HumanData) visualization.')
    # input args
    parser.add_argument(
        'human_data_path', type=str, help='Path to the input human_data file.')
    # output args
    parser.add_argument(
        'output_video_path',
        type=str,
        help='Path to the output video.',
        default='./default_output.mp4')
    # optional args
    parser.add_argument(
        '--background_dir',
        type=str_or_empty,
        help='Path to the directory with background images in it.',
        default='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    main(args)
