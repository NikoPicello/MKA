import argparse
import sys 
sys.path.append("/mnt/petrelfs/luohuiwen/renbody/zip/zoehuman")
from zoehuman.core.visualization.visualize_keypoints3d import visualize_kp3d
from zoehuman.data.data_structures.human_data import HumanData


def main(args):
    human_data_3d = HumanData.fromfile(args.human_data_path)
    print(human_data_3d['keypoints3d'].shape)
    visualize_kp3d(
        human_data_3d['keypoints3d'].copy(),
        output_path=args.output_video_path,
        data_source='smplx',
        mask=human_data_3d['keypoints3d_mask'].copy(),
        value_range=None)


def setup_parser():
    parser = argparse.ArgumentParser(description='')
    # input args
    parser.add_argument(
        '--human_data_path',
        type=str,
        help='Path to the input human_data file.',
        default='')
    # output args
    parser.add_argument(
        '--output_video_path',
        type=str,
        help='Path to the output video.',
        default='./default_output.mp4')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    main(args)
