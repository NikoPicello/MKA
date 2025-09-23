import copy
import os.path as osp
import time

import numpy as np
from mmhuman3d.core.visualization.visualize_keypoints2d import visualize_kp2d
from mocap.multi_view_3d_keypoint.triangulate_scene import TriangulateScene

from zoehuman.core.cameras.camera_parameters import (  # noqa: E501
    CameraParameter, DepthCameraParameter,
)
from zoehuman.core.conventions.keypoints_mapping import convert_kps
from zoehuman.core.visualization.visualize_keypoints3d import visualize_kp3d
from zoehuman.utils.process_humandata_utils import Logger


class Triangulator:

    def __init__(
        self,
        init_cam_error_tolerance: float = 100.0,
        max_cam_error_tolerance: float = 1000.0,
        error_tolerance_step: float = 50.0,
        keep_best_n_cam: int = None,
        at_least_n_cam: int = 3,
        scale_smooth: float = 4.0,
    ) -> None:
        """Create a Triangulator instance that will build a TriangulateScene
        with camera parameters list, and triangulate human data.

        After this, you get 3d keypoints. 3d keypoints will be optimized later.
        """
        self.cam_select_strategy = {
            'keep_n_cam': keep_best_n_cam,
            'mean_error_thr': init_cam_error_tolerance
        }
        self.scale_smooth = scale_smooth
        self.max_tol = max(max_cam_error_tolerance, init_cam_error_tolerance)
        self.tol_step = error_tolerance_step
        self.at_least_n_cam = at_least_n_cam
        return

    def select_cameras(self,
                       cam_errors: np.ndarray,
                       cam_select_strategy: dict = None):
        """ Select cameras with smaller reprojection errors
        Args:
            cam_errors (np.ndarray):
                Errors in (number_of_cam, 1) shape
            cam_select_strategy (dict):
                A camera will be selected if it meets two requirements.
                First, its error is one of the n smallest,
                where n = cam_select_strategy['keep_n_cam'].
                If cam_select_strategy['keep_n_cam'] is None,
                this requirememt will be ignored.
                Second, its error is not larger than thr,
                where thr = cam_select_strategy['mean_error_thr'].
                If cam_select_strategy['mean_error_thr'] is None,
                this requirement will be ignored.
        Return:
            Indices of cameras that meet the requirements.
        """
        if cam_select_strategy is None:
            cam_select_strategy = self.cam_select_strategy
        if cam_select_strategy['keep_n_cam'] is not None:
            n_cam = cam_select_strategy['keep_n_cam']
            cam_indices_1 = np.argpartition(
                cam_errors, n_cam - 1, axis=0)[:n_cam]
        else:
            cam_indices_1 = np.arange(cam_errors.shape[0])
        if cam_select_strategy['mean_error_thr'] is not None:
            thr = cam_select_strategy['mean_error_thr']
            cam_indices_2 = np.where(cam_errors <= thr)[0]
        return np.intersect1d(cam_indices_1, cam_indices_2)

    @staticmethod
    def get_camera_error(scene,
                         human_data_list: list,
                         keypoints3d: np.ndarray,
                         constraints=None) -> np.ndarray:
        errors = scene.get_init_error(human_data_list, keypoints3d,
                                      constraints)
        errors = np.mean(np.abs(errors), axis=(1, 2))
        return errors

    def process_3d(self,
                   vis_dir: str,
                   camera_param_list: list,
                   human_data_list: list,
                   key_list: list,
                   bgr_frame_list: list,
                   iphone_cam: CameraParameter = None,
                   keypoints_thr: float = 0.3,
                   select_cam=True,
                   pipeline_results_dict: dict = {},
                   info_dict: dict = {},
                   time_dict: dict = {},
                   project: bool = True,
                   visualize_kp3d_tri: bool = True,
                   visualize_kp3d_optim: bool = True,
                   visualize_reprojection: bool = True,
                   data_source: str = 'coco_wholebody'):
        scene, keypoints3d_tri, \
            human_data_list_select, \
            camera_param_list_select, \
            key_list_select = \
            self.triangulate(
                camera_param_list,
                human_data_list,
                key_list,
                keypoints_thr,
                select_cam,
                info_dict,
                time_dict
                )
        status = info_dict.get('status', None)
        if (status is None) or ('fail' not in status):
            keypoints3d_optim = self.optim(scene, human_data_list_select,
                                           keypoints3d_tri.copy(), None, True,
                                           info_dict, time_dict)
            self.save_results(
                vis_dir,
                camera_param_list,
                human_data_list,
                key_list,
                bgr_frame_list,
                iphone_cam,
                keypoints3d_tri,
                keypoints3d_optim,
                project=project,
                visualize_kp3d_tri=visualize_kp3d_tri,
                visualize_kp3d_optim=visualize_kp3d_optim,
                visualize_reprojection=visualize_reprojection,
                pipeline_results_dict=pipeline_results_dict,
                time_dict=time_dict,
                data_source=data_source)

    def triangulate(
        self,
        camera_param_list: list,
        human_data_list: list,
        key_list: list,
        keypoints_thr: float = 0.3,
        select_cam=True,
        info_dict={},
        time_dict={},
    ):
        """Triangulate the 2d human data list and output 3d points.

        Cameras will be selected greedily according to cam_select_strategy.
            After selecting cameras, a new TriangulateScene instance
            will be created with the selected cameras. The selection will
            be looped until all cameras in the TriangulateScene instance
            are selected in the following selection.
        Args:
            camera_param_list (list):
                Original camera param list used to create
                TriangulateScene instance.
            human_data_list (list):
                Original human data list used to estimate the 3d keypoints
            key_list (list):
                Camera key list corresponding to the camera indices.
            keypoints_thr (float):
                Human data with confidence above keypoints_thr will be
                used to estimate the 3d keypoints. Others will not contribute
                to 3d estimation.
            select_cam (bool, optional):
                Select camera or use all camera.
            info_dict (dict, optional):
                Store the infomation in info_dict.
            time_dict (dict, optional):
                Store the time consumption in time_dict.
        Return:
            scene (TriangulateScene):
                The TriangulateScene instance.
            keypoints3d (np.ndarray):
                The estimated 3d key points from selected 2d key points.
            human_data_list (list):
                The selected human data list.
            camera_param_list (list):
                The selected camera parameter list.
            key_list (list):
                The selected key list corresponding to the selected
                camera indices.
        """
        t0 = time.time()
        try_times_outer = 0
        while True and try_times_outer < 3:
            scene = TriangulateScene(camera_param_list, keypoints_thr)
            keypoints3d = scene.triangulate(human_data_list)
            if select_cam:
                cam_errors = Triangulator.get_camera_error(
                    scene, human_data_list, keypoints3d)
                strategy = copy.deepcopy(self.cam_select_strategy)
                cam_indices = self.select_cameras(cam_errors, strategy)
                while (len(cam_indices) < self.at_least_n_cam
                       and strategy['mean_error_thr'] < self.max_tol):
                    strategy['mean_error_thr'] += self.tol_step
                    cam_indices = self.select_cameras(cam_errors, strategy)
                if len(cam_indices) == len(key_list) or len(cam_indices) < 3:
                    break
                else:
                    camera_param_list = [
                        camera_param_list[i] for i in cam_indices
                    ]
                    human_data_list = [human_data_list[i] for i in cam_indices]
                    key_list = [key_list[i] for i in cam_indices]
                    try_times_outer += 1
            else:
                cam_indices = list(range(0, len(camera_param_list)))
                break
        t1 = time.time()
        time_dict['triangulate'] = Logger.format_time(t1 - t0)
        failure = (
            len(cam_indices) < self.at_least_n_cam
            or len(cam_indices) != len(key_list))
        if try_times_outer == 0 and failure:
            info_dict['status'] = 'fail_cam_select'
        if select_cam:
            info_dict['selected_cam'] = key_list
            info_dict['cam_select_error_thr'] = strategy['mean_error_thr']
        return scene, keypoints3d, human_data_list, camera_param_list, key_list

    def optim(
        self,
        scene,
        human_data_list,
        keypoints3d,
        constraints=None,
        log_cam_errors=True,
        info_dict={},
        time_dict={},
    ):
        """Optimize the 3d key points."""
        t0 = time.time()
        keypoints3d = scene.optim(
            human_data_list,
            keypoints3d,
            constraints,
            scale_smooth=self.scale_smooth)
        t1 = time.time()
        time_dict['optimize'] = Logger.format_time(t1 - t0)
        if log_cam_errors:
            cam_errors = Triangulator.get_camera_error(scene, human_data_list,
                                                       keypoints3d)
            info_dict['final_cam_errors'] = cam_errors.tolist()
        return keypoints3d

    @staticmethod
    def transform_points_cam_to_floor(
        keypoints3d: np.ndarray,
        camera: DepthCameraParameter,
    ):
        camera.setup_transform()
        shape = keypoints3d.shape
        keypoints3d = camera.transform_points_cam_to_floor(
            keypoints3d.reshape(-1, shape[-1]))
        return keypoints3d.reshape(shape)

    def save_results(self,
                     vis_dir,
                     camera_param_list,
                     human_data_list,
                     key_list,
                     bgr_frame_list,
                     iphone_cam,
                     keypoints3d_tri,
                     keypoints3d_optim,
                     project=True,
                     visualize_kp3d_tri=True,
                     visualize_kp3d_optim=True,
                     visualize_reprojection=True,
                     pipeline_results_dict={},
                     time_dict={},
                     data_source='coco_wholebody'):
        scene = TriangulateScene(camera_param_list)

        human_data_3d_optim = TriangulateScene.convert_result_to_human_data(
            keypoints3d_optim, human_data_list[0]['keypoints2d_mask'])

        if project:
            pipeline_results_dict['Keypoints2D'] = {}
            pipeline_results_dict['Keypoints2D']['Kinect'] = {}
            kinect_reproj_results = pipeline_results_dict['Keypoints2D'][
                'Kinect']

            projected_human_data_list = scene.project(human_data_3d_optim)

            for index, key in enumerate(key_list):
                i = str(int(index))
                kp_2d_reproj, mask_2d = convert_kps(
                    projected_human_data_list[index]['keypoints2d'],
                    src='human_data',
                    dst=data_source)
                kinect_reproj_results[i] = {}
                kinect_reproj_results[i]['keypoints2d'] = kp_2d_reproj
                kinect_reproj_results[i]['keypoints2d_mask'] = mask_2d

        if project and (iphone_cam is not None):
            iphone_scene = TriangulateScene([iphone_cam])
            iphone_projected_human_data_list = \
                iphone_scene.project(human_data_3d_optim)
            pipeline_results_dict['Keypoints2D']['iPhone'] = {}
            iphone_reproj_res = pipeline_results_dict['Keypoints2D']['iPhone']

            # convert convention
            kp_2d_reproj, mask_2d = convert_kps(
                iphone_projected_human_data_list[0]['keypoints2d'],
                mask=iphone_projected_human_data_list[0]['keypoints2d_mask'],
                src='human_data',
                dst=data_source)

            iphone_reproj_res['0'] = {}
            iphone_reproj_res['0']['keypoints2d'] = kp_2d_reproj
            iphone_reproj_res['0']['keypoints2d_mask'] = mask_2d

            if visualize_reprojection:
                t0 = time.time()
                for index, key in enumerate(key_list):
                    vis_path = osp.join(vis_dir, f'kp2d_project_cam_{key}.mp4')
                    visualize_kp2d(
                        projected_human_data_list[index]['keypoints2d'],
                        output_path=vis_path,
                        image_array=bgr_frame_list[index],
                        data_source='human_data',
                        mask=projected_human_data_list[index]
                        ['keypoints2d_mask'],
                        resolution=bgr_frame_list[index][0].shape,
                        overwrite=True,
                        disable_tqdm=True,
                    )
                t1 = time.time()
                time_dict['visualize_reproj'] = Logger.format_time(t1 - t0)

        human_data_3d_tri = TriangulateScene.convert_result_to_human_data(
            keypoints3d_tri, human_data_list[0]['keypoints2d_mask'])
        human_data_3d_optim = TriangulateScene.convert_result_to_human_data(
            keypoints3d_optim, human_data_list[0]['keypoints2d_mask'])
        kp3d_tri = human_data_3d_tri['keypoints3d']
        kp3d_tri[np.isnan(kp3d_tri)] = 0.0

        kp_3d_optim, mask_3d = convert_kps(
            human_data_3d_optim['keypoints3d'],
            src='human_data',
            dst=data_source)

        pipeline_results_dict['Keypoints3D'] = {}
        pipeline_results_dict['Keypoints3D']['attrs'] = {}
        pipeline_results_dict['Keypoints3D']['attrs']['num_frame'] = \
            kp_3d_optim.shape[0]
        pipeline_results_dict['Keypoints3D']['attrs']['convention'] = \
            data_source
        pipeline_results_dict['Keypoints3D']['attrs']['created_time'] = \
            Logger.get_current_time()
        pipeline_results_dict['Keypoints3D']['keypoints3d'] = kp_3d_optim
        pipeline_results_dict['Keypoints3D']['keypoints3d_mask'] = mask_3d

        if visualize_kp3d_tri:
            t0 = time.time()
            vis3d_path = osp.join(vis_dir, 'human_data_3d_tri_vis.mp4')
            visualize_kp3d(
                human_data_3d_tri['keypoints3d'].copy(),
                vis3d_path,
                data_source='human_data',
                mask=human_data_3d_tri['keypoints3d_mask'].copy())
            t1 = time.time()
            time_dict['visualize_3d_tri'] = Logger.format_time(t1 - t0)
        if visualize_kp3d_optim:
            t0 = time.time()
            vis3d_path = osp.join(vis_dir, 'human_data_3d_optim_vis.mp4')
            visualize_kp3d(
                human_data_3d_optim['keypoints3d'].copy(),
                vis3d_path,
                data_source='human_data',
                mask=human_data_3d_optim['keypoints3d_mask'].copy())
            t1 = time.time()
            time_dict['visualize_3d_optim'] = Logger.format_time(t1 - t0)
        return 0
