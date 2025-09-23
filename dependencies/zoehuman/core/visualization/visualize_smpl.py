from typing import Tuple

import cv2
import numpy as np
from mmhuman3d.core.visualization.visualize_smpl import \
    visualize_smpl_calibration  # prevent yapf isort conflict

from zoehuman.core.conventions.cameras.convert_convention import \
    convert_K_4x4_to_3x3  # prevent yapf isort conflict


def visualize_smpl_distortion(
    K,
    R,
    T,
    resolution,
    image_array,
    dist_coeffs,
    **kwargs,
) -> None:
    """Visualize a smpl mesh which has opencv calibration matrix defined in
    distorted screen. In this method we assume that all images in image_array
    share the same intrinsic.

    Only image_array background accepted for now.
    """
    assert K is not None, '`K` is required.'
    assert resolution is not None, '`resolution`(h, w) is required.'
    intrinsic = K if len(K.shape) == 2 else K[0]
    if len(intrinsic) == 4:
        intrinsic = convert_K_4x4_to_3x3(K=intrinsic)
    corrected_image_array, corrected_intrinsic = __undistort_images__(
        intrinsic=intrinsic,
        width=resolution[1],
        height=resolution[0],
        dist_coeffs=dist_coeffs,
        image_array=image_array)
    ret_val = visualize_smpl_calibration(
        K=corrected_intrinsic,
        R=R,
        T=T,
        resolution=resolution,
        image_array=corrected_image_array,
        **kwargs)
    return ret_val


def __undistort_images__(intrinsic: np.ndarray, width: int, height: int,
                         dist_coeffs: dict, image_array: np.ndarray) -> Tuple:
    dist_coeff_list = [
        dist_coeffs.get('k1', 0.0),
        dist_coeffs.get('k2', 0.0),
        dist_coeffs.get('p1', 0.0),
        dist_coeffs.get('p2', 0.0),
        dist_coeffs.get('k3', 0.0),
        dist_coeffs.get('k4', 0.0),
        dist_coeffs.get('k5', 0.0),
        dist_coeffs.get('k6', 0.0),
    ]
    corrected_image_array = np.ones_like(image_array)
    corrected_intrinsic = np.zeros(shape=(3, 3))
    resolution_wh = np.array([width, height])
    corrected_intrinsic, _ = cv2.getOptimalNewCameraMatrix(
        intrinsic, np.array(dist_coeff_list), resolution_wh, 0, resolution_wh)
    for image_index, image_np in enumerate(image_array):
        corrected_image_array[image_index] = cv2.undistort(
            image_np,
            intrinsic,
            np.array(dist_coeff_list),
            newCameraMatrix=corrected_intrinsic)
    return corrected_image_array, corrected_intrinsic
