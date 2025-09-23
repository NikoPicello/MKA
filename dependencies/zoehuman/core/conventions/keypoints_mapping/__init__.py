from typing import Optional, Tuple, Union

import numpy as np
import torch
from mmhuman3d.core.conventions.keypoints_mapping import \
    KEYPOINTS_FACTORY as KEYPOINTS_FACTORY_mm  # noqa: F401, E501
from mmhuman3d.core.conventions.keypoints_mapping import \
    convert_kps as convert_kps_mm  # noqa: F401, E501

from . import (  # noqa: F401, E501
    gta, k4abt, label_convention, openpose, sense_omni_v1, sense_whole_body,
)

# k4abt: Azure Kinect Body Tracking SDK
_KEYPOINTS_FACTORY = {
    'gta': gta.GTA_JOINTS,
    'k4abt': k4abt.K4ABT_JOINTS,
    'label_superset': label_convention.LABELME_SUPERSET,
    'label_superset_dance': label_convention.LABELME_SUPERSET_DANCE,
    'sense_whole_body': sense_whole_body.SENSE_WHOLE_BODY_KEYPOINTS,
    'sense_omni_v1': sense_omni_v1.SENSE_OMNI_V1_KEYPOINTS,
    'openpose_118': openpose.OPENPOSE_118_KEYPOINTS
}
KEYPOINTS_FACTORY = KEYPOINTS_FACTORY_mm.copy()
KEYPOINTS_FACTORY.update(_KEYPOINTS_FACTORY)


def convert_kps(
    keypoints: Union[np.ndarray, torch.Tensor],
    src: str,
    dst: str,
    approximate: bool = False,
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    keypoints_factory: dict = KEYPOINTS_FACTORY,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """Convert keypoints following the mapping correspondence between src and
    dst keypoints definition.

    Args:
        keypoints (np.ndarray): input keypoints array, could be
            (f * n * J * 3/2) or (f * J * 3/2). You can set keypoints as
            np.zeros((1, J, 2)) if you only need mask.
        src (str): source data type from keypoints_factory.
        dst (str): destination data type from keypoints_factory.
        approximate (bool): control whether approximate mapping is allowed.
        mask (Optional[Union[np.ndarray, torch.Tensor]], optional):
            The original mask to mark the existence of the keypoints.
            None represents all ones mask.
            Defaults to None.
        keypoints_factory (dict, optional): A class to store the attributes.
            Defaults to keypoints_factory.
    Returns:
        Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]
            : tuple of (out_keypoints, mask). out_keypoints and mask will be of
            the same type.
    """
    return convert_kps_mm(
        keypoints=keypoints,
        src=src,
        dst=dst,
        approximate=approximate,
        mask=mask,
        keypoints_factory=keypoints_factory)
