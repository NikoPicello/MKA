import argparse
import io
import os
import os.path as osp

import h5py
from mocap.multi_view_3d_keypoint.mmpose_helper import TopDownProcessorSMC
from mocap.multi_view_3d_keypoint.triangulator import Triangulator

from zoehuman.utils.process_humandata_utils import H5Helper, Logger


class FileProcessor:

    def __init__(self,
                 file_list_txt: str,
                 out_dir: str,
                 vis_dir: str,
                 start_index: int = 0,
                 end_index: int = None,
                 ceph_sdk_conf_path: str = '~/petreloss.conf') -> None:
        """ Init a FileProcessor instance that process
            input and output paths
        Args:
            file_list_txt (str):
                Input txt file containing .7z file list
                could be a ceph path starting with 's3://'
            out_dir (str):
                Directory containing processed keypoints
            vis_dir (str):
                Directory containing visualization results
            start_index (int, optional):
                Index of the first file to be processed
            end_index (int, optional):
                Index of the last file to be processed
            ceph_sdk_conf_path (str, optional):
                path to petreloss sdk config file
        """
        self.file_list_txt = file_list_txt
        if not out_dir.startswith('s3://'):
            os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        os.makedirs(vis_dir, exist_ok=True)
        self.vis_dir = vis_dir

        self.client_init = False
        if file_list_txt.startswith('s3://') or out_dir.startswith('s3://'):
            from petrel_client.client import Client
            self.client = Client(ceph_sdk_conf_path)
            self.client_init = True

        if file_list_txt.startswith('s3://'):
            txt_bytes = self.client.get(file_list_txt)
            files = txt_bytes.decode('utf-8')
            files = files.split('\n')
        else:
            with open(file_list_txt) as f:
                files = f.read().split('\n')
        if files[-1] == '':
            files = files[0:-1]

        end_index = len(files) if not end_index else end_index
        files = [f for f in files if (f.endswith('.7z') or f.endswith('.smc'))]
        self.files = files[start_index:end_index]
        has_ceph_file = False
        for f in self.files:
            if 's3://' in f:
                has_ceph_file = True
                break
        if has_ceph_file and not self.client_init:
            from petrel_client.client import Client
            self.client = Client(ceph_sdk_conf_path)
            self.client_init = True

        self.length = len(self.files)
        print(f'{self.length} files to be processed')
        self.file_keys = [osp.splitext(osp.basename(f))[0] for f in self.files]

    def get_file_path(self, index: int) -> str:
        """ Get the smc buffer of the
            specified index
        Args:
            index (int):
                File index
        """
        return self.files[index]

    def get_basename(self, index: int):
        """Get the basename of input file."""
        return self.file_keys[index]

    def get_vis_dir(self, index: int, make_dir: bool = True):
        basename = self.get_basename(index)
        vis_path = osp.join(self.vis_dir, basename)
        if make_dir:
            os.makedirs(vis_path, exist_ok=True)
        return vis_path

    def save_smc(self, index: int, smc_buffer: io.BytesIO):
        file_name = self.get_basename(index) + '.smc'
        out_path = osp.join(self.out_dir, file_name)
        if out_path.startswith('s3://'):
            self.client.put(out_path, io.BytesIO(smc_buffer.getvalue()))
        else:
            with open(out_path, 'wb') as f:
                f.write(smc_buffer.getvalue())
        return None


class PipeLine:

    def __init__(self, args) -> None:
        """Create a PipeLine instance that processes data.

        Five members are heavily used here. FileProcessor helps to get input
        file and create output directory. Unzipper helps to unzip 7z into smc
        and clean smc when it's used. Logger helps to record information for
        each 7z file. TopDownProcessorSMC helps to get 2d key points from smc
        file. Triangulator helps to estimate 3d key points.
        """
        self.file_processor = FileProcessor(args.file_list_txt, args.out_dir,
                                            args.vis_dir, args.start_index,
                                            args.end_index,
                                            args.ceph_sdk_conf_path)
        self.top_down_processor_smc = TopDownProcessorSMC(
            args, 'cuda', use_batch=args.batch_detector)
        self.triangulator = Triangulator(
            args.init_cam_error_tolerance,
            args.max_cam_error_tolerance,
            args.error_tolerance_step,
            args.keep_best_n_cam,
            args.at_least_n_cam,
            args.scale_smooth,
        )

        visualize_list = args.visualize.split(',')
        self.need_vis_frame = 'frame' in visualize_list
        self.need_vis_kp2d = 'kp2d' in visualize_list
        self.need_vis_kp3d_tri = 'kp3d_tri' in visualize_list
        self.need_vis_kp3d_optim = 'kp3d_optim' in visualize_list
        self.need_vis_project = 'project' in visualize_list
        if len(visualize_list) > 1 and (not os.path.isdir(args.vis_dir)):
            os.makedirs(args.vis_dir)
        if self.need_vis_project:
            args.project = True
        self.args = args

    def process_all(self):
        file_num = self.file_processor.length
        for i in range(file_num):
            self.process_one_file(
                index=i, copy_input_to_output=self.args.copy_input_to_output)
        return None

    def process_one_file(self, index, copy_input_to_output=False):
        time_dict = {}
        info_dict = {}
        pipeline_results_dict = {}
        input_path, vis_dir, file_name = \
            self.preprocess(index)

        camera_param_list, human_data_list, \
            key_list, bgr_frame_list, iphone_cam, smc_reader = \
            self.process_2d(
                input_path,
                vis_dir,
                data_source=self.args.data_source,
                pipeline_results_dict=pipeline_results_dict,
                info_dict=info_dict,
                time_dict=time_dict)

        status = info_dict.get('status', None)
        if (status is None) or ('fail' not in status):
            self.process_3d(
                vis_dir,
                camera_param_list,
                human_data_list,
                key_list,
                bgr_frame_list,
                iphone_cam=iphone_cam,
                pipeline_results_dict=pipeline_results_dict,
                info_dict=info_dict,
                time_dict=time_dict,
                data_source=self.args.data_source)

        log_dict = {
            'index': index,
            'name': file_name,
            'status': info_dict.get('status', None) or 'success',
            'time': time_dict,
            'info': info_dict,
            'date': Logger.get_current_time()
        }

        log_str = Logger.get_json_str_from_dict(log_dict)
        pipeline_results_dict['Debug/Log'] = log_str

        if copy_input_to_output:
            iob = H5Helper.h5py_to_binary(smc_reader.smc)
        else:
            iob = io.BytesIO()
        with h5py.File(iob, 'a') as writable_smc:
            H5Helper.recursively_save_dict_contents_to_h5file(
                writable_smc, '/', pipeline_results_dict)
        self.file_processor.save_smc(index, iob)

        return None

    def preprocess(self, index):
        input_path = self.file_processor.get_file_path(index)
        file_name = self.file_processor.get_basename(index)
        vis_dir = self.file_processor.get_vis_dir(index)
        return input_path, vis_dir, file_name

    def process_2d(self,
                   smc_buffer,
                   vis_dir,
                   data_source='coco_wholebody',
                   pipeline_results_dict={},
                   info_dict={},
                   time_dict={}):
        camera_param_list, human_data_list, \
            key_list, bgr_frame_list, iphone_cam, smc_reader = \
            self.top_down_processor_smc.top_down_process(
                smc_buffer, vis_dir, data_source,
                need_vis_frame=self.need_vis_frame,
                need_vis_kp2d=self.need_vis_kp2d,
                det_multi_person=self.args.det_multi_person,
                bbox_thr=self.args.bbox_thr,
                pipeline_results_dict=pipeline_results_dict,
                info_dict=info_dict,
                time_dict=time_dict
            )
        return camera_param_list, human_data_list, \
            key_list, bgr_frame_list, iphone_cam, smc_reader

    def process_3d(
        self,
        vis_dir: str,
        camera_param_list: list,
        human_data_list: list,
        key_list: list,
        bgr_frame_list: list,
        iphone_cam=None,
        pipeline_results_dict={},
        info_dict={},
        time_dict={},
        data_source='coco_wholebody',
    ):
        self.triangulator.process_3d(
            vis_dir,
            camera_param_list,
            human_data_list,
            key_list,
            bgr_frame_list=bgr_frame_list,
            iphone_cam=iphone_cam,
            keypoints_thr=self.args.keypoints_thr,
            select_cam=self.args.select_cam,
            pipeline_results_dict=pipeline_results_dict,
            info_dict=info_dict,
            time_dict=time_dict,
            project=self.args.project,
            visualize_kp3d_tri=self.need_vis_kp3d_tri,
            visualize_kp3d_optim=self.need_vis_kp3d_optim,
            visualize_reprojection=self.need_vis_project,
            data_source=data_source,
        )
        return None


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Top down detection and pose estimation, ' +
        'and triangulate '
        ' powered by mmdet and mmpose')
    # model args
    parser.add_argument(
        'det_config', help='Config file for detection', type=str)
    parser.add_argument(
        'det_checkpoint', help='Checkpoint file for detection', type=str)
    parser.add_argument('pose_config', help='Config file for pose', type=str)
    parser.add_argument(
        'pose_checkpoint', help='Checkpoint file for pose', type=str)
    parser.add_argument(
        '--data_source',
        help='pose data source type',
        default='coco_wholebody',
        type=str)
    # directory args
    parser.add_argument(
        '--in_dir',
        type=str,
        help='Path to input directory containing 7z files.',
        default='')
    parser.add_argument(
        '--file_list_txt',
        type=str,
        help='Path to txt file containing 7z file list.',
        default='')
    parser.add_argument(
        '--ceph_sdk_conf_path',
        type=str,
        help='Path to petreloss sdk configure file',
        default='~/petreloss.conf')
    parser.add_argument(
        '--out_dir',
        type=str,
        help='Path to the directory saving processed output files \
            could be ceph path or local path',
        default='./keypoints')
    parser.add_argument(
        '--copy_input_to_output',
        action='store_true',
        help='if False, save only pipeline results \
            if True, save input info to out smc file')
    parser.add_argument(
        '--vis_dir',
        type=str,
        help='Path to the directory saving ' + 'output mp4 files.',
        default='./vis_dir')
    parser.add_argument(
        '--start_index',
        type=int,
        help='start index to the files in in_dir to be processed',
        default=0)
    parser.add_argument(
        '--end_index',
        type=int,
        help='end index to the files in in_dir to be processed',
        default=9999)
    parser.add_argument(
        '--bbox_thr',
        type=float,
        help='Threshold of bbox scores.',
        default=0.0)
    parser.add_argument(
        '--visualize',
        type=str,
        help='visualize the results with mp4, \
            frame: the original RGB frame, \
            kp2d: detected 2d key points, \
            kp3d_tri: 3d key points after triangulation, \
            kp3d_optim: 3d key points after optimization, \
            project: reprojected 2d key points',
        default='frame,kp2d,kp3d_tri,kp3d_optim,project')
    parser.add_argument(
        '--det_multi_person',
        action='store_true',
        help='If checked, save multi-person result.',
        default=False)
    parser.add_argument(
        '--save_cam_parameter',
        action='store_true',
        help='If checked, save multi-person result.',
        default=False)
    # triangulate args
    parser.add_argument(
        '--keypoints_thr',
        type=float,
        help='Threshold of keypoint scores.',
        default=0.0)
    parser.add_argument(
        '--scale_smooth',
        type=float,
        help='scale of smooth strength in optimization',
        default=4.0)
    parser.add_argument(
        '--disable_optim',
        action='store_true',
        help='If checked, disable scene.optim().',
        default=False)
    parser.add_argument(
        '--project',
        action='store_true',
        help='If checked, visualize poses.',
        default=False)
    parser.add_argument(
        '--select_cam',
        action='store_true',
        help='If checked, visualize poses.',
        default=False)
    parser.add_argument(
        '--init_cam_error_tolerance',
        type=float,
        help='initial cam error tolerance',
        default=100.00)
    parser.add_argument(
        '--max_cam_error_tolerance',
        type=float,
        help='cam error tolerance maximum, \
        if there are not enough cams with errors smaller than this value, \
        pipeline will stop for this piece of data',
        default=1000.00)
    parser.add_argument(
        '--error_tolerance_step',
        type=float,
        help='tolerance will increase with this value, \
        if there are not enough cams with errors below tolerance',
        default=50.00)
    parser.add_argument(
        '--keep_best_n_cam',
        type=int,
        help='select n cams with least errors',
        default=10)
    parser.add_argument(
        '--at_least_n_cam',
        type=int,
        help='cam num must equal to or be higher than n cam',
        default=3)
    parser.add_argument(
        '--batch_detector',
        action='store_true',
        help='If checked, visualize poses.',
        default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    pipe_line = PipeLine(args)
    pipe_line.process_all()
