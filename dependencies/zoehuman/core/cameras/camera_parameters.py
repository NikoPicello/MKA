import json
from typing import List

import numpy as np
from mmhuman3d.core.cameras.camera_parameters import \
    CameraParameter as CameraParameter_mm
from mmhuman3d.core.cameras.camera_parameters import __parse_chessboard_param__
from scipy.spatial.transform import Rotation as scipy_Rotation

from zoehuman.core.conventions.cameras.convert_convention import \
    convert_K_3x3_to_4x4  # prevent yapf isort conflict


class CameraParameter(CameraParameter_mm):

    AUGMENTED_SUPPORTED_KEYS = {
        'floor_normal': {
            'type': list,
            'len': 3,
        },
        'floor_center': {
            'type': list,
            'len': 3,
        }
    }

    SUPPORTED_KEYS = dict(CameraParameter_mm.SUPPORTED_KEYS,
                          **AUGMENTED_SUPPORTED_KEYS)

    def __init__(self,
                 name: str = 'default',
                 H: int = 1080,
                 W: int = 1920) -> None:
        """
        Args:
            name (str, optional):
                Name of this camera. Defaults to "default".
            H (int, optional):
                Height of a frame, in pixel. Defaults to 1080.
            W (int, optional):
                Width of a frame, in pixel. Defaults to 1920.
        """
        super().__init__(name=name, H=H, W=W)

    def get_KRT(self,
                k_dim: int = 3,
                inverse_extrinsic: bool = False) -> List[np.ndarray]:
        """Get intrinsic and extrinsic of a camera.

        Args:
            k_dim (int, optional):
                Dimension of the returned mat K.
                Defaults to 3.
            inverse_extrinsic (bool, optional):
                If true, R_mat and T_vec transform a point
                from view to world. Defaults to False.

        Raises:
            ValueError: k_dim is neither 3 nor 4.

        Returns:
            List[np.ndarray]:
                K_mat (np.ndarray):
                    In shape [3, 3].
                R_mat (np.ndarray):
                    Rotation from world to view in default.
                    In shape [3, 3].
                T_vec (np.ndarray):
                    Translation from world to view in default.
                    In shape [3,].
        """
        K_3x3 = self.get_mat_np('in_mat')
        R_mat = self.get_mat_np('rotation_mat')
        T_vec = np.asarray(self.get_value('translation'))
        if inverse_extrinsic:
            R_mat = np.linalg.inv(R_mat).reshape(3, 3)
            T_vec = -np.dot(R_mat, T_vec)
        if k_dim == 3:
            return [K_3x3, R_mat, T_vec]
        elif k_dim == 4:
            K_3x3 = np.expand_dims(K_3x3, 0)  # shape (1, 3, 3)
            K_4x4 = convert_K_3x3_to_4x4(
                K=K_3x3, is_perspective=True)  # shape (1, 4, 4)
            K_4x4 = K_4x4[0, :, :]
            return [K_4x4, R_mat, T_vec]
        else:
            raise ValueError(f'K mat cannot be converted to {k_dim}x{k_dim}')

    def inverse_extrinsics(self):
        """Inverse camera extrinsics.

        Call it when you get wrong results with current parameters.
        """
        r_mat_np = self.get_mat_np('rotation_mat')
        r_mat_inv_np = np.linalg.inv(r_mat_np).reshape(3, 3)
        t_vec_list = self.get_value('translation')
        t_vec_np = np.asarray(t_vec_list)
        t_vec_inv_np = -np.dot(r_mat_inv_np, t_vec_np)
        self.set_mat_np('rotation_mat', r_mat_inv_np)
        self.set_value('translation', t_vec_inv_np.tolist())

    def load_kinect_from_smc(self, smc_reader, kinect_id: int) -> None:
        """Load name and parameters of a kinect from an SmcReader instance.

        Args:
            smc_reader (mocap.data_collection.smc_reader.SMCReader):
                An SmcReader instance containing kinect camera parameters.
            kinect_id (int):
                Id of the target kinect.
        """
        name = kinect_id
        extrinsics_dict = \
            smc_reader.get_kinect_color_extrinsics(
                kinect_id, homogeneous=False
            )
        rot_np = extrinsics_dict['R']
        trans_np = extrinsics_dict['T']
        intrinsics_np = \
            smc_reader.get_kinect_color_intrinsics(
                kinect_id
            )
        resolution = \
            smc_reader.get_kinect_color_resolution(
                kinect_id
            )
        rmatrix = np.linalg.inv(rot_np).reshape(3, 3)
        tvec = -np.dot(rmatrix, trans_np)
        self.name = name
        self.set_mat_np('in_mat', intrinsics_np)
        self.set_mat_np('rotation_mat', rmatrix)
        self.set_value('translation', tvec.tolist())
        self.set_value('H', int(resolution[1]))
        self.set_value('W', int(resolution[0]))

    def load_iphone_from_smc(self,
                             smc_reader,
                             iphone_id: int = 0,
                             frame_id: int = 0) -> None:
        """Load name and parameters of an iPhone from an SmcReader instance.

        Args:
            smc_reader (mocap.data_collection.smc_reader.SMCReader):
                An SmcReader instance containing kinect camera parameters.
            iphone_id (int):
                Id of the target iphone.
                Defaults to 0.
            frame_id (int):
                Frame id of one selected frame.
                It only influences the intrinsics.
                Defaults to 0.
        """
        name = f'iPhone_{iphone_id}'
        extrinsics_mat = \
            smc_reader.get_iphone_extrinsics(
                iphone_id, frame_id, vertical=False
            )
        rot_np = extrinsics_mat['R']
        trans_np = extrinsics_mat['T']
        intrinsics_np = \
            smc_reader.get_iphone_intrinsics(
                iphone_id, frame_id, vertical=False
            )
        resolution = \
            smc_reader.get_iphone_color_resolution(
                iphone_id, vertical=False
            )
        rmatrix = np.linalg.inv(rot_np).reshape(3, 3)
        tvec = -np.dot(rmatrix, trans_np)
        self.name = name
        self.set_mat_np('in_mat', intrinsics_np)
        self.set_mat_np('rotation_mat', rmatrix)
        self.set_value('translation', tvec.tolist())
        self.set_value('H', int(resolution[1]))
        self.set_value('W', int(resolution[0]))

    def load_from_lightstage(self,
                             lightstage_dict: dict,
                             cam_id: int,
                             inverse: bool = True) -> None:
        """Load name and parameters from a dict calibrated in lightstage.

        Args:
            lightstage_dict (dict):
                A dict loaded by np.load().item()['cams']
            cam_id (int):
                ID of this camera.
                An int in [0, 48).
        """
        extrinsic = lightstage_dict['RT'][cam_id, :, :]  # 4x4 mat
        intrinsic = lightstage_dict['K'][cam_id, :, :]  # 3x3 mat
        r_mat_inv = extrinsic[:3, :3]
        r_mat = np.linalg.inv(r_mat_inv)
        t_vec = extrinsic[:3, 3:]
        t_vec = -np.dot(r_mat, t_vec).reshape((3))
        self.set_mat_np('rotation_mat', r_mat)
        self.set_mat_np('in_mat', intrinsic)
        self.set_value('translation', t_vec.tolist())

    def get_aist_dict(self) -> dict:
        """Get a dict of camera parameters, which contains all necessary args
        for aniposelib.cameras.Camera(). Use
        aniposelib.cameras.Camera(**return_dict) to construct a camera.

        Returns:
            dict:
                A dict of camera parameters: name, dist, size, matrix, etc.
        """
        ret_dict = {}
        ret_dict['name'] = self.name
        ret_dict['dist'] = [
            self.parameters_dict['k1'],
            self.parameters_dict['k2'],
            self.parameters_dict['p1'],
            self.parameters_dict['p2'],
            self.parameters_dict['k3'],
        ]
        ret_dict['size'] = (self.parameters_dict['H'],
                            self.parameters_dict['W'])
        ret_dict['matrix'] = np.array(self.parameters_dict['in_mat'])
        rotation_mat = np.array(self.parameters_dict['rotation_mat'])
        # convert rotation as axis angle(rotation vector)
        rotation_vec = scipy_Rotation.from_matrix(rotation_mat).as_rotvec()
        ret_dict['rvec'] = rotation_vec
        ret_dict['tvec'] = self.parameters_dict['translation']
        return ret_dict

    def setup_transform(self):
        """Setup transform between camera0 and self."""
        # rotation matrix from camera0 coordinates to self coordinates
        rot_mat = \
            np.asarray(self.get_value('rotation_mat')).reshape(3, 3)
        self.rotation = \
            scipy_Rotation.from_matrix(rot_mat)
        self.translation = np.asarray(self.get_value('translation'))
        # from self to camera0
        self.inv_rotation = self.rotation.inv()
        self.transform_ready = True

    def transform_points_cam_to_self(self, points3d: np.ndarray) -> np.ndarray:
        """Transform an array of 3d points in camera0 coordinates to self
        coordinates.

        Args:
            points3d (np.ndarray):
                An array of 3d points, in shape [point_number, 3]
                or [point_number, 4] with confidence.

        Returns:
            np.ndarray:
                An array of transformed 3d points, in the same
                shape of points3d. Only data points[:, :3] is
                different.
        """
        assert self.transform_ready is True, \
            'Transform not ready, call self.setup_transform() first.'
        assert points3d.ndim == 2 and \
            (points3d.shape[1] == 3 or points3d.shape[1] == 4), \
            'Input.shape has to be [point_number, 3] or [point_number, 4].'
        self_points3d = \
            self.rotation.apply(points3d[:, :3])
        translation_np = \
            self.translation[np.newaxis, :].repeat(
                self_points3d.shape[0], axis=0)
        self_points3d += translation_np

        output_points3d = points3d.copy()
        output_points3d[:, :3] = self_points3d
        return output_points3d

    def to_string(self) -> str:
        """Convert self.to_dict() to a string.

        Returns:
            str:
                A dict in json string format.
        """
        dump_dict = self.to_dict()
        ret_str = json.dumps(dump_dict, default=convert_np)
        return ret_str


class DepthCameraParameter(CameraParameter):

    def __init__(self,
                 name: str = 'default',
                 H: int = 1080,
                 W: int = 1920) -> None:
        """
        Args:
            name (str, optional):
                Name of this camera. Defaults to 'default'.
            H (int, optional):
                Height of a frame, in pixel. Defaults to 1080.
            W (int, optional):
                Width of a frame, in pixel. Defaults to 1920.
        """
        super().__init__(name=name, H=H, W=W)
        self.parameters_dict['floor_center'] = [0, 0, 0]
        self.parameters_dict['floor_normal'] = [0, 1, 0]
        self.transform_ready = False

    def load_kinect_from_smc(self, smc_reader, kinect_id: int) -> None:
        """Load name and parameters from an SmcReader instance.

        Args:
            smc_reader (mocap.data_collection.smc_reader.SMCReader):
                An SmcReader instance containing kinect camera parameters.
            kinect_id (int):
                Id of the target kinect.
        """
        super().load_kinect_from_smc(smc_reader, kinect_id)
        floor_dict = smc_reader.get_depth_floor(kinect_id)
        norm_vec = np.array(floor_dict['normal']).reshape((3)).tolist()
        self.set_value('floor_normal', norm_vec)
        center = np.array(floor_dict['center']).reshape((3)).tolist()
        self.set_value('floor_center', center)
        self.transform_ready = False

    def load_from_chessboard(self,
                             chessboard_dict: dict,
                             name: str,
                             inverse: bool = True) -> None:
        """Load name and parameters from a dict.

        Args:
            chessboard_dict (dict):
                A dict loaded from json.load(chessboard_file).
            name (str):
                Name of this camera. Must be a depth camera.
            inverse (bool, optional):
                Whether to inverse rotation and translation mat.
                Defaults to False.
        """
        camera_param_dict = \
            __parse_chessboard_param__(
                chessboard_dict,
                name,
                inverse=inverse)
        assert 'floor' in chessboard_dict
        camera_param_dict['floor_normal'] = \
            np.array(chessboard_dict['floor']['normal']).reshape((3)).tolist()
        camera_param_dict['floor_center'] = \
            np.array(chessboard_dict['floor']['center']).reshape((3)).tolist()
        self.load_from_dict(camera_param_dict)
        self.transform_ready = False

    def setup_transform(self):
        """Setup transform between camera0 and floor, camera0 and self."""
        super().setup_transform()
        # rotation matrix from camera0 coordinates to self coordinates
        rot_mat = \
            np.asarray(self.get_value('rotation_mat')).reshape(3, 3)
        self.rotation = \
            scipy_Rotation.from_matrix(rot_mat)
        # from self to camera0
        self.inv_rotation = self.rotation.inv()

        floor_normal = np.asarray(self.get_value('floor_normal'))
        self_normal = np.asarray([0, -1, 0])
        self.self_to_floor_rotation = scipy_Rotation.align_vectors(
            np.expand_dims(self_normal, 0), np.expand_dims(floor_normal, 0))[0]
        self.floor_to_self_rotation = self.self_to_floor_rotation.inv()
        self.transform_ready = True

    def transform_points_cam_to_floor(self,
                                      points3d: np.ndarray) -> np.ndarray:
        """Transform an array of 3d points in camera0 coordinates to floor
        coordinates.

        Args:
            points3d (np.ndarray):
                An array of 3d points, in shape [point_number, 3]
                or [point_number, 4] with confidence.

        Returns:
            np.ndarray:
                An array of transformed 3d points, in the same
                shape of points3d. Only data points[:, :3] is
                different.
        """
        assert self.transform_ready is True, \
            'Transform not ready, call self.setup_transform() first.'
        assert points3d.ndim == 2 and \
            (points3d.shape[1] == 3 or points3d.shape[1] == 4), \
            'Input.shape has to be [point_number, 3] or [point_number, 4].'
        self_points3d = \
            self.rotation.apply(points3d[:, :3])
        floor_points3d = \
            self.self_to_floor_rotation.apply(self_points3d[:, :3])
        output_points3d = points3d.copy()
        output_points3d[:, :3] = floor_points3d
        return output_points3d

    def transform_points_floor_to_cam(self,
                                      points3d: np.ndarray) -> np.ndarray:
        """Transform an array of 3d points in floor coordinates back to camera0
        coordinates. Inverse of self.transform_points_cam_to_floor()

        Args:
            points3d (np.ndarray):
                An array of 3d points, in shape [point_number, 3]
                or [point_number, 4] with confidence.

        Returns:
            np.ndarray:
                An array of transformed 3d points, in the same
                shape of points3d. Only data points[:, :3] is
                different.
        """
        assert self.transform_ready is True, \
            'Transform not ready, call self.setup_transform() first.'
        assert points3d.ndim == 2 and \
            (points3d.shape[1] == 3 or points3d.shape[1] == 4), \
            'Input.shape has to be [point_number, 3] or [point_number, 4].'
        self_points3d = \
            self.floor_to_self_rotation.apply(points3d[:, :3])
        cam_points3d = \
            self.inv_rotation.apply(self_points3d[:, :3])
        output_points3d = points3d.copy()
        output_points3d[:, :3] = cam_points3d
        return output_points3d


def convert_np(value):
    if isinstance(value, np.generic):
        return value.item()
    else:
        return value
