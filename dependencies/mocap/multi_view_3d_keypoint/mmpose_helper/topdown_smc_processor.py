import io
import os.path as osp
import time
from typing import Union
import sys
sys.path.append('/nvme/lufan/Projects/RenBody/Renbody_benchmark/keypoint_smpl/zoehuman')
sys.path.append('/nvme/lufan/Projects/RenBody/Renbody_benchmark/keypoint_smpl/zoehuman/tools')
from mmhuman3d.core.visualization.visualize_keypoints2d import visualize_kp2d
from mmhuman3d.utils.ffmpeg_utils import array_to_video
from mocap.multi_view_3d_keypoint.mmpose_helper.top_down_pose_det import (  # noqa: E501
    PoseDetector, PoseDetectorBatch, convert_results_to_human_data,
)
# from tools.top_down_pose import get_max_person_number, get_model_files
from top_down_pose import get_max_person_number, get_model_files

from zoehuman.core.cameras.camera_parameters import DepthCameraParameter
from zoehuman.core.conventions.keypoints_mapping import convert_kps
from zoehuman.data.data_structures import SMCReader
from zoehuman.utils.process_humandata_utils import Logger, PipeLine2DHelper


class TopDownProcessorSMC:

    def __init__(self, args, detector_device='cuda', use_batch=False) -> None:
        """Create a TopDownProcessorSMC instance that will load frames from smc
        file, detect their poses and convert the poses into human_data
        format."""
        det_args, pose_args = get_model_files(args)
        det_args['device'] = detector_device
        pose_args['device'] = detector_device
        if use_batch:
            self.detector = PoseDetectorBatch(det_args, pose_args)
        else:
            self.detector = PoseDetector(det_args, pose_args)

    def get_iphone_cam_param(
        self,
        smc_reader,
    ):
        if smc_reader.iphone_exists:
            cam_param = DepthCameraParameter(name='0')
            cam_param.load_iphone_from_smc(smc_reader, 0)
            return cam_param
        else:
            return None

    def top_down_process(
        self,
        smc_file: Union[str, io.BytesIO],
        vis_dir,
        data_source='coco_wholebody',
        need_vis_frame=True,
        need_vis_kp2d=True,
        det_multi_person=False,
        bbox_thr=0.0,
        pipeline_results_dict={},
        info_dict={},
        time_dict={},
    ):
        """Process one smc file First, load the smc file with SMCReader;
        Secord, read the frames from each RGB cam; Third, detect the poses from
        frames; Forth, visualize and save.

        Args:
            smc_file (str or io.BytesIO):
                path to the input smc file
                or io.Bytes of smc file
            vis_dir (str):
                where to save the visualization in this process
            data_source (str):
                data_source type
            save_cam_parameter (bool, optional):
                save the cam parameter or not
            visualize (bool, optional):
                whether you will visualize the results
            det_multi_person (bool, optional):
                detect multi people or not
            bbox_thr (float, optional):
                bbox score above bbox_thr will be assumed
                containing a body
            pipeline_results_dict (dict, optional):
                store all the results in pipeline
            info_dict (dict, optional):
                store the infomation in info_dict
            time_dict (dict, optional):
                store the time consumption in time_dict
        Return:
            camera_param_list (list):
                list of camera parameters that
                will be used in TriangulateScene
            human_data_list (list):
                list of human data results got in this stage
            key_list (list):
                list of camera key list such as ['01', '02', '03']
        """
        try:
            smc_reader = SMCReader(smc_file)
        except OSError:
            info_dict['status'] = 'fail_smc_broken'
            return None, None, None, None
        num_of_kinect = smc_reader.get_num_kinect()
        time_dict['get_frames'] = 0.0
        time_dict['infer_pose'] = 0.0
        time_dict['visualize_2d'] = 0.0
        camera_param_list = []
        human_data_list = []
        key_list = []
        bgr_frame_list = []
        all_2d_results = {}
        pipeline_results_dict['Debug/Keypoints2D'] = all_2d_results
        all_2d_results['Kinect'] = {}

        for i in range(num_of_kinect):
            key = f'{i:02d}'
            key_list.append(key)
            pose_dict, bgr_frames, camera_param = \
                self.top_down_process_per_view(
                    kinect_index=i,
                    smc_reader=smc_reader,
                    det_multi_person=det_multi_person,
                    time_dict=time_dict,
                )
            camera_param_list.append(camera_param)
            bgr_frame_list.append(bgr_frames)
            resolution = smc_reader.get_kinect_color_resolution(i)
            if need_vis_frame:
                t0 = time.time()
                array_to_video(
                    bgr_frames,
                    output_path=osp.join(vis_dir, f'frame_cam_{key}.mp4'),
                    resolution=(resolution[1], resolution[0]),
                    disable_log=False)
                t1 = time.time()
                time_dict['visualize_2d'] += (t1 - t0)
            if need_vis_kp2d:
                pose_array, error_frame_key = \
                    PipeLine2DHelper.pose_dict_to_array(pose_dict)
                t0 = time.time()
                visualize_kp2d(
                    pose_array[..., 0:2].copy(),
                    image_array=bgr_frames,
                    data_source=data_source,
                    output_path=osp.join(vis_dir, f'kp2d_cam_{key}.mp4'),
                    resolution=(resolution[1], resolution[0]),
                    overwrite=True,
                    disable_tqdm=True,
                )
                t1 = time.time()
                time_dict['visualize_2d'] += (t1 - t0)
                if len(error_frame_key) > 0:
                    info_dict[f'cam_{key}_error_frame'] = error_frame_key
                    info_dict['status'] = 'frame_with_no_pose'
            if not det_multi_person:
                human_data = convert_results_to_human_data(
                    pose_dict,
                    bbox_threshold=bbox_thr,
                    data_source=data_source)
                human_data_list.append(human_data)
            else:
                max_person_number = get_max_person_number(pose_dict, bbox_thr)
                for person_index in range(max_person_number):
                    human_data = convert_results_to_human_data(
                        pose_dict,
                        bbox_threshold=bbox_thr,
                        person_index=person_index,
                        data_source=data_source)
                    human_data_list.append(human_data)

            kp_2d_i, mask_2d = convert_kps(
                human_data['keypoints2d'], src='human_data', dst=data_source)

            all_2d_results['Kinect'][i] = {}
            all_2d_results['Kinect'][i]['keypoints2d'] = kp_2d_i
            all_2d_results['Kinect'][i]['keypoints2d_mask'] = mask_2d

        time_dict['get_frames'] = Logger.format_time(time_dict['get_frames'])
        time_dict['infer_pose'] = Logger.format_time(time_dict['infer_pose'])
        time_dict['visualize_2d'] = Logger.format_time(
            time_dict['visualize_2d'])
        return camera_param_list, human_data_list, key_list, \
            bgr_frame_list, self.get_iphone_cam_param(smc_reader), smc_reader

    def top_down_process_per_view(
        self,
        kinect_index: int,
        smc_reader,
        det_multi_person=False,
        time_dict: dict = None,
    ):
        """ Get the color frames from kinect color camera,
            then detect their poses
        Args:
            kinect_index (int):
                index of view to be processed
            smc_reader (SMCReader):
                smc reader instance
            det_multi_person (bool, optional):
                detect multi people or not
            time_dict (dict, optional):
                store the time consumption in time_dict
        Return:
            pose_dict (dict):
                detected pose from mmpose detector
            bgr_frames (np.ndarray):
                frames of BGR channel
            camera_param (CameraParameter):
                camera parameter of this view
        """
        t0 = time.time()
        rgb_frames = smc_reader.get_kinect_color(kinect_index)
        t1 = time.time()
        time_dict['get_frames'] += (t1 - t0)
        bgr_frames = PipeLine2DHelper.switch_color_channel(
            rgb_frames, (0, 2), (2, 0))
        frame_dict = PipeLine2DHelper.frame_array_to_dict(bgr_frames)
        t0 = time.time()
        pose_dict = self.detector.infer_array(
            frame_dict,
            disable_tqdm=True,
            multi_person=det_multi_person,
        )
        t1 = time.time()
        time_dict['infer_pose'] += (t1 - t0)
        camera_param = DepthCameraParameter(f'{kinect_index:02d}')
        camera_param.load_kinect_from_smc(smc_reader, kinect_index)
        return pose_dict, bgr_frames, camera_param
