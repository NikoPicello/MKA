from typing import List, Union, Optional, Tuple

import aniposelib
import numpy as np
import prettytable as pt
from aniposelib.cameras import interpolate_data

from zoehuman.data.data_structures.human_data import HumanData
from zoehuman.utils.keypoint_utils import search_limbs

from mmhuman3d.core.conventions.keypoints_mapping.human_data import (
    HUMAN_DATA,
    HUMAN_DATA_LIMBS_INDEX,
    HUMAN_DATA_PALETTE,
)

# SMPLX_KEYPOINTS_25 = [
#     'pelvis',
#     'left_hip',
#     'right_hip',
#     'spine_1',
#     'left_knee',
#     'right_knee',
#     'spine_2',
#     'left_ankle',
#     'right_ankle',
#     'spine_3',
#     'left_foot',
#     'right_foot',
#     'neck',
#     'left_collar',
#     'right_collar',
#     'head',
#     'left_shoulder',
#     'right_shoulder',
#     'left_elbow',
#     'right_elbow',
#     'left_wrist',
#     'right_wrist',
#     'jaw',
#     'left_eyeball',
#     'right_eyeball'
# ]
# KEYPOINTS_FACTORY = {
#     'human_data': HUMAN_DATA,
#     'smplx': SMPLX_KEYPOINTS_25,
# }
# def search_limbs(
#         data_source: str,
#         mask: Optional[Union[np.ndarray, tuple, list]] = None,
#         keypoints_factory: dict = KEYPOINTS_FACTORY) -> Tuple[dict, dict]:
#     """Search the corresponding limbs following the basis human_data limbs. The
#     mask could mask out the incorrect keypoints.

#     Args:
#         data_source (str): data source type.
#         mask (Optional[Union[np.ndarray, tuple, list]], optional):
#             refer to keypoints_mapping. Defaults to None.
#         keypoints_factory (dict, optional): Dict of all the conventions.
#             Defaults to KEYPOINTS_FACTORY.
#     Returns:
#         Tuple[dict, dict]: (limbs_target, limbs_palette).
#     """
#     limbs_source = HUMAN_DATA_LIMBS_INDEX
#     limbs_palette = HUMAN_DATA_PALETTE
#     keypoints_source = keypoints_factory['human_data']
#     keypoints_target = keypoints_factory[data_source]
#     limbs_target = {}
#     for k, part_limbs in limbs_source.items():
#         limbs_target[k] = []
#         for limb in part_limbs:
#             flag = False
#             if (keypoints_source[limb[0]]
#                     in keypoints_target) and (keypoints_source[limb[1]]
#                                               in keypoints_target):
#                 if mask is not None:
#                     if mask[keypoints_target.index(keypoints_source[
#                             limb[0]])] != 0 and mask[keypoints_target.index(
#                                 keypoints_source[limb[1]])] != 0:
#                         flag = True
#                 else:
#                     flag = True
#                 if flag:
#                     limbs_target.setdefault(k, []).append([
#                         keypoints_target.index(keypoints_source[limb[0]]),
#                         keypoints_target.index(keypoints_source[limb[1]])
#                     ])
#         if k in limbs_target:
#             if k == 'body':
#                 np.random.seed(0)
#                 limbs_palette[k] = np.random.randint(
#                     0, high=255, size=(len(limbs_target[k]), 3))
#             else:
#                 limbs_palette[k] = np.array(limbs_palette[k])
#     return limbs_target, limbs_palette

class TriangulateScene:

    def __init__(self,
                 camera_parameters: List,
                 keypoint_threshold: Union[float, str] = 'auto') -> None:
        """A scene composed of several cameras, triangulating 2D locations of
        points to 3D locations.

        Args:
            camera_parameters (list):
                A list of camera_parameters. Each element is
                an instance of class
                zoehuman.core.cameras.camera_parameters.CameraParameter.
                The order of cameras in this list is important for triangulate.
            keypoint_threshold (Union[float, str], optional):
                The score threshold to filter bad points.
                When it's set to 'auto', the scene will try to find the highest
                keypoint_threshold without losing any pair.
                Defaults to 'auto'.
        """
        self.camera_parameters = camera_parameters
        aniposelib_camera_list = []
        for camera_parameter in camera_parameters:
            args_dict = camera_parameter.get_aist_dict()
            camera = aniposelib.cameras.Camera(**args_dict)
            aniposelib_camera_list.append(camera)
        self.camera_group = \
            aniposelib.cameras.CameraGroup(aniposelib_camera_list)
        self.keypoint_threshold = keypoint_threshold

    def triangulate(self, human_data_list, keep_cam_num=None, show_stats=True):
        """Triangulate multi-view keypoints 2d to keypoints 3d, ignore bad
        points filtered by self.keypoint_threshold.

        Args:
            human_data_list (list):
                A list of HumanData. It must be in the same order
                of camera_parameters.
            show_stats (bool):
                Whether to show statistics about ignored keypoints.
                Defaults to True.

        Returns:
            ndarray:
                Keypoints 3d in numpy array.
                The shape of it is [frame_number, keypoints_number, 3].
        """
        keypoints2d_np, mask = \
            TriangulateScene.__human_data_to_keypoints2d__(human_data_list)
        # print(keypoints2d_np.shape, mask.shape)
        view_number, frame_number, _, _ = keypoints2d_np.shape
        if isinstance(self.keypoint_threshold, float) and \
                self.keypoint_threshold >= 0 and \
                self.keypoint_threshold <= 1:
            # filter by keypoint score, set the low position to np.nan
            ignore_idxs = np.where(
                keypoints2d_np[:, :, :, 2] < self.keypoint_threshold)
        elif self.keypoint_threshold == 'auto':
            keypoint_threshold = \
                __try_keypoints_threshold__(
                    keypoints2d_np, mask
                )
            print(f'Auto keypoints threshold found: {keypoint_threshold}')
            ignore_idxs = np.where(
                keypoints2d_np[:, :, :, 2] < keypoint_threshold)
        else:
            raise ValueError(
                f'Wrong keypoint_threshold: {self.keypoint_threshold}')

        if keep_cam_num is not None:
            keypoints2d_scores = keypoints2d_np[:, :, :, 2]
            partition_indices = np.argpartition(
                keypoints2d_scores, view_number - keep_cam_num, axis=0)
            min_indies = partition_indices[:view_number - keep_cam_num, :, :]
            min_indies = np.expand_dims(min_indies, -1)
            np.put_along_axis(keypoints2d_np, min_indies, np.nan, axis=0)
        
        # print('2d', keypoints2d_np.shape)
        # print(keypoints2d_np[0, 0, 14])
        keypoints2d_np[ignore_idxs[0], ignore_idxs[1],
                       ignore_idxs[2], :] = np.nan
        mask_idxs = np.where(mask[:] == 0)
        keypoints2d_np[:, :, mask_idxs, :] = np.nan
        # print(keypoints2d_np[0, 0, 14])
        if show_stats:
            __show_nan_stats__(keypoints2d_np, mask_idxs)
        keypoints3d = self.camera_group.triangulate(
            keypoints2d_np[:, :, :, :2].reshape(view_number, -1, 2), progress=True).reshape(
                frame_number, -1, 3)
        return keypoints3d

    def get_init_error(
        self,
        human_data_list,
        keypoints3d,
        constraints=None,
    ):
        keypoints2d_np, mask = \
            TriangulateScene.__human_data_to_keypoints2d__(human_data_list)
        _, frame_number, keypoints_number, _ = keypoints2d_np.shape
        keypoints2d_np = keypoints2d_np[:, :, :, :2]
        if 'keypoints2d_convention' in human_data_list[0]:
            data_source = human_data_list[0]['keypoints2d_convention']
        else:
            data_source = 'human_data'
        if constraints is None:
            limbs_target_dict, _ = search_limbs(
                data_source=data_source, mask=mask)
            constraints = []
            for limb_name, limb_list in limbs_target_dict.items():
                for sub_index, limb in enumerate(limb_list):
                    assert len(limb) == 2, \
                        f'limb {limb_name}:{sub_index:02d} has' +\
                        f' {len(limb)} points!'
                    constraints.append(limb)
        constraints_np = np.asarray(constraints)
        assert np.min(constraints_np) >= 0, \
            'Keypoints index in constraints starts from 0'
        assert np.max(constraints_np) < keypoints_number, \
            'Keypoints index in constraints no more than %d' % keypoints_number

        n_cams, n_frames, n_joints, _ = keypoints2d_np.shape
        p3ds_intp = np.apply_along_axis(interpolate_data, 0, keypoints3d)
        constraints_weak = []
        x0 = self.camera_group._initialize_params_triangulation(
            p3ds_intp, constraints, constraints_weak)

        n_3d = n_frames * n_joints * 3
        p3ds = x0[:n_3d].reshape((n_frames, n_joints, 3))
        p3ds_flat = p3ds.reshape(-1, 3)
        p2ds_flat = keypoints2d_np.reshape((n_cams, -1, 2))
        errors = self.camera_group.reprojection_error(p3ds_flat, p2ds_flat)
        return errors

    def optim(self,
              human_data_list,
              keypoints3d,
              constraints=None,
              verbose=True,
              **kwargs):
        """Creates an optimized array of 3D points of shape NxJx3 like
        triangulate_optim().

        Args:
            human_data_list (list):
                A list of HumanData. It must be in the same order
                of camera_parameters, also it is the exact source input
                for keypoints3d generation.
            keypoints3d (ndarray):
                The result returned by self.triangulate(human_data_list).
            constraints (list):
                A list like [[0,1], [0, 2]] records
                connections between keypoints.
                When None, search limbs from human_data definition with mask.
                Defaults to None.
            verbose (bool):
                Show real-time infomation to stdout about optim.
                Defaults to True.

        Returns:
            ndarray:
                Optimized keypoints 3d in numpy array.
                The shape of it is [frame_number, keypoints_number, 3].
        """
        c = np.isfinite(keypoints3d[:, :, 0])
        if np.sum(c) < 20:
            print('Warning: not enough 3D points to run optimization')
            return keypoints3d
        keypoints2d_np, mask = \
            TriangulateScene.__human_data_to_keypoints2d__(human_data_list)
        _, frame_number, keypoints_number, _ = keypoints2d_np.shape
        keypoints2d_np = keypoints2d_np[:, :, :, :2]
        if 'keypoints2d_convention' in human_data_list[0]:
            data_source = human_data_list[0]['keypoints2d_convention']
        else:
            data_source = 'human_data'
        if constraints is None:
            limbs_target_dict, _ = search_limbs(
                data_source=data_source, mask=mask)
            constraints = []
            for limb_name, limb_list in limbs_target_dict.items():
                for sub_index, limb in enumerate(limb_list):
                    assert len(limb) == 2, \
                        f'limb {limb_name}:{sub_index:02d} has' +\
                        f' {len(limb)} points!'
                    constraints.append(limb)
        constraints_np = np.asarray(constraints)
        assert np.min(constraints_np) >= 0, \
            'Keypoints index in constraints starts from 0'
        assert np.max(constraints_np) < keypoints_number, \
            'Keypoints index in constraints no more than %d' % keypoints_number
        keypoints3d = \
            self.camera_group.optim_points(
                keypoints2d_np, keypoints3d, constraints=constraints,
                verbose=verbose, **kwargs
            ).reshape(frame_number, -1, 3)
        return keypoints3d

    def project(self, keypoints3d_human_data):
        """Project keypoints 3d back to multi-view keypoints 2d.

        Args:
            keypoints3d_human_data (HumanData):
                Keypoints 3d in HumanData form.
                Typically set this arg to the output of
                TriangulateScene.convert_result_to_human_data().

        Returns:
            list:
                A list of HumanData, like the input arg human_data_list
                in self.triangulate()human_data_list.
        """
        keypoints3d = keypoints3d_human_data['keypoints3d']
        frame_number, keypoints_number, _ = keypoints3d.shape

        # all in one
        keypoints3d_to_project = keypoints3d[:, :, :3].reshape(-1, 3)
        projected_keypoints2d = self.camera_group.project(
            keypoints3d_to_project)
        keypoints2d_np = projected_keypoints2d.reshape(
            len(self.camera_parameters), frame_number, keypoints_number, 2)

        if keypoints3d_human_data['keypoints3d_mask'] is not None:
            mask = keypoints3d_human_data['keypoints3d_mask']
        else:
            mask = np.ones(shape=[keypoints_number], dtype=np.int8)
        ret_list = []
        for view_index in range(keypoints2d_np.shape[0]):
            view_human_data = HumanData()
            keypoints2d_view = np.ones(
                shape=[frame_number, keypoints_number, 3])
            keypoints2d_view[:, :, :2] = keypoints2d_np[view_index, :, :, :]
            view_human_data['keypoints2d'] = keypoints2d_view
            view_human_data['keypoints2d_mask'] = mask
            ret_list.append(view_human_data)
        return ret_list

    @staticmethod
    def __human_data_to_keypoints2d__(human_data_list):
        view_number = len(human_data_list)
        frame_number, keypoints_number, dim_number = \
            human_data_list[0]['keypoints2d'].shape
        keypoints2d_np = np.zeros(
            shape=(view_number, frame_number, keypoints_number, 3))
        view_count = 0
        for human_data in human_data_list:
            keypoints2d_view = human_data['keypoints2d']
            assert keypoints2d_view.shape[0] == frame_number, \
                'frame number difference between view 0 and view %d' \
                % view_count
            keypoints2d_np[view_count, :, :, :dim_number] = keypoints2d_view
            view_count += 1
        # if there's no keypoints score in HumanData, set them to 1.0
        if dim_number == 2:
            keypoints2d_np[:, :, :, 2] = 1.0
        if human_data_list[0]['keypoints2d_mask'] is not None:
            mask = human_data_list[0]['keypoints2d_mask']
        else:
            mask = np.ones(shape=[keypoints_number], dtype=np.int8)
        
        return keypoints2d_np, mask

    @staticmethod
    def unmask_human_data(human_data_list):
        """Un-mask the mask in HumanData and convert them into a keypoints2d np
        array. Not in use.

        Args:
            human_data_list (list):
                A list of HumanData. It must be in the same order
                of camera_parameters, also it is the exact source input
                for keypoints3d generation.

        Returns:
            ndarray:
                Keypoints 2d in numpy array.
                The shape of it is
                [view_number, frame_number, keypoints_number, 3].
            dict:
                Mapping keypoint index in returned ndarray to
                HumanData Keypoints.
            dict:
                Mapping keypoint index in HumanData Keypoints
                back to returned ndarray. Some are absent in
                this_dict.keys().
        """
        view_number = len(human_data_list)
        frame_number, _, dim_number = \
            human_data_list[0]['keypoints2d'].shape
        mask = human_data_list[0]['keypoints2d_mask']
        keypoints_number = np.sum(mask)
        unmask_mapping = {}
        mask_mapping = {}
        unmasked_index = 0
        for masked_index, mask_value in enumerate(mask):
            if mask_value == 1:
                unmask_mapping[unmasked_index] = masked_index
                mask_mapping[masked_index] = unmasked_index
                unmasked_index += 1
        keypoints2d_np = np.zeros(
            shape=(view_number, frame_number, keypoints_number, 3))
        view_count = 0
        for human_data in human_data_list:
            keypoints2d_view = human_data['keypoints2d']
            assert keypoints2d_view.shape[0] == frame_number, \
                'frame number difference between view 0 and view %d' \
                % view_count
            for unmasked_keypoint_index in range(keypoints_number):
                masked_index = unmask_mapping[unmasked_keypoint_index]
                keypoints2d_np[
                    view_count, :, unmasked_keypoint_index, :dim_number] = \
                    keypoints2d_view[:, masked_index, :]
            view_count += 1
        # if there's no keypoints score in HumanData, set them to 1.0
        if dim_number == 2:
            keypoints2d_np[:, :, :, 2] = 1.0
        return keypoints2d_np, unmask_mapping, mask_mapping

    @staticmethod
    def convert_result_to_human_data(keypoints3d, mask, src_human_data=None):
        """Convert value returned by TriangulateScene.triangulate() or
        TriangulateScene.optim() to HumanData dict.

        Args:
            keypoints3d (ndarray):
                The result returned by self.triangulate()
                or self.optim().
            mask (np.ndarray):
                Mask for keypoints.
                0 means that the joint in this position
                cannot be found in original dataset.
            src_human_data (HumanData, optional):
                When src_human_data is set,
                the human_data to return will inherit values
                from src_human_data,
                except keypoints3d and mask.

        Returns:
            dict:
                HumanData containing keypoints3d, as well as values
                inherit from src_human_data.
        """
        frame_number, keypoints_number, dim_number = keypoints3d.shape
        ret_human_data = HumanData()
        if src_human_data is not None:
            ret_human_data.update(src_human_data)
        keypoints3d_with_score = np.zeros(
            shape=(frame_number, keypoints_number, 4))
        keypoints3d_with_score[:, :, :dim_number] = keypoints3d
        keypoints3d_with_score[:, :, 3] = 1
        ret_human_data['keypoints3d'] = keypoints3d_with_score
        ret_human_data['keypoints3d_mask'] = mask
        return ret_human_data


def __get_valid_stats_dict__():
    valid_stats_dict = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    return valid_stats_dict


def __show_nan_stats__(keypoints2d_np, mask_idxs):
    """Ignoring masked keypoints, define a pair containing all views about a
    single keypoint, in one fram. Count how many valid views in each pair, and
    print percentage of critical pairs in a table.

    Args:
        keypoints2d_np (np.ndarray):
            The first output of
            TriangulateScene.__human_data_to_keypoints2d__().
        mask_idxs (tuple):
            Output of np.where()
    """
    _, frame_number, keypoints_number, _ = keypoints2d_np.shape
    mask_idxs_list = mask_idxs[0].tolist()
    total_pairs = frame_number * (keypoints_number - len(mask_idxs_list))
    # init valid count
    valid_stats_dict = __get_valid_stats_dict__()
    for keypoint_index in range(keypoints_number):
        # if ignored by mask, skip
        if keypoint_index in mask_idxs_list:
            continue
        for frame_index in range(frame_number):
            # check how many valid views in one data pair
            pair_data = \
                keypoints2d_np[:, frame_index, keypoint_index, 0]
            valid_data = pair_data[~np.isnan(pair_data)]
            valid_number = len(valid_data)
            # if critical, count it
            if valid_number in valid_stats_dict.keys():
                valid_stats_dict[valid_number] += 1
    # ratio
    if total_pairs > 0:
        for key in valid_stats_dict.keys():
            valid_stats_dict[key] = \
                valid_stats_dict[key] / float(total_pairs)
    table = pt.PrettyTable()
    table.field_names = ['Valid Views', 'Pairs']
    for key, item in valid_stats_dict.items():
        table.add_row([key, item])
    print(table)
    return table


def __try_keypoints_threshold__(keypoints2d_np: np.ndarray,
                                keypoints2d_mask: np.ndarray,
                                start: float = 0.95,
                                stride: float = -0.05,
                                lower_bound: float = 0.0):
    """Try the largest keypoints_threshold by loop, which can be represented as
    start+n*stride and makes number of valid views >= 2.

    Args:
        keypoints2d_np (np.ndarray):
            In shape [view_number, frame_number, keypoints_number, 3].
        keypoints2d_mask (np.ndarray):
            In shape [keypoints_number, ].
        start (float, optional):
            Init threshold, should be in (0, 1].
            Defaults to 0.95.
        stride (float, optional):
            Step of one loop, should be in (-start, 0).
            Defaults to -0.05.
        lower_bound (float, optional):
            Lower bound of threshold, should be in [0.0, start).
            Defaults to 0.0.

    Returns:
        float: The best keypoints_threshold.
    """
    assert start > 0 and start <= 1
    assert stride > (0 - start) and stride < 0
    keypoints2d_init = keypoints2d_np.copy()
    keypoints2d_init[:, :, keypoints2d_mask == 0, :] = 1
    keypoints_thr = start
    while True:
        pair_fail = False
        if keypoints_thr < lower_bound:
            keypoints_thr = lower_bound
            break
        valid_mask = keypoints2d_init[..., -1] >= keypoints_thr
        valid_num_array = np.sum(valid_mask, axis=0)
        if np.any(valid_num_array < 2):
            pair_fail = True
        if pair_fail:
            keypoints_thr += stride
        else:
            break
    return keypoints_thr
