from mocap.multi_view_3d_keypoint.mmpose_helper.top_down_pose_det import (  # noqa:E501
    PoseDetector, convert_results_to_human_data,
)
from mocap.multi_view_3d_keypoint.triangulate_scene import TriangulateScene

from zoehuman.core.cameras.camera_parameters import CameraParameter


class KeyPointExtractor:

    def __init__(self, smc, detector_config_path, detector_checkpoint_path,
                 pose_config_path, pose_checkpoint_path):
        self.smc = smc
        self.detector_kwargs = {
            'config': detector_config_path,
            'checkpoint': detector_checkpoint_path
        }
        self.pose_model_kwargs = {
            'config': pose_config_path,
            'checkpoint': pose_checkpoint_path
        }
        print(self.detector_kwargs)
        print(self.pose_model_kwargs)
        self.detector = PoseDetector(self.detector_kwargs,
                                     self.pose_model_kwargs)
        self.scene = self.__init_scene__()

    def extract_key_points(self):
        keypoints2d = []
        human_data = []
        frame_dict_list = self.__get_infer_inputs__()
        for frame_dict in frame_dict_list:
            frames_result_dict = self.detector.infer_array(frame_dict)
            tmp_human_data = convert_results_to_human_data(frames_result_dict)
            human_data.append(tmp_human_data)
            keypoints2d.append(tmp_human_data['keypoints2d'])
        keypoint3d = self.scene.triangulate(human_data)
        # optim with bone constraints
        keypoint3d = self.scene.optim(
            human_data, keypoints3d=keypoint3d, constraints=None)
        return keypoints2d, keypoint3d

    def __init_scene__(self):
        camera_para_list = []
        for camera_id in range(0, self.smc.get_num_kinect(), 1):
            camera_name = str(camera_id)
            temp_camera_parameter = CameraParameter(name=camera_name)
            temp_camera_parameter.load_kinect_from_smc(self.smc, camera_id)
            camera_para_list.append(temp_camera_parameter)
        scene = TriangulateScene(
            camera_parameters=camera_para_list, keypoint_threshold=0.6)
        return scene

    def __get_infer_inputs__(self, ):
        ret_list = []
        # camera.load_kinect_from_smc...
        for cam_id in range(0, self.smc.get_num_kinect(), 1):
            frames = self.smc.get_kinect_color(cam_id)
            infer_input_dict = {}
            for frame_id in range(0, frames.shape[0], 1):
                frame_name = 'frame_{}'.format(frame_id)
                infer_input_dict[frame_name] = frames[frame_id]
            ret_list.append(infer_input_dict)
        return ret_list
