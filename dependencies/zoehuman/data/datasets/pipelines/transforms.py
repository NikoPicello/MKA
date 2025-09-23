from typing import List

import numpy as np
import torch
from mmhuman3d.data.datasets.pipelines.transforms import \
    get_affine_transform  # noqa: F401


def crop_resize_points(points2d: torch.Tensor, bbox_xyxy: torch.Tensor,
                       dst_resolution: torch.Tensor) -> torch.Tensor:
    """Transform points2d to destination image space cropped and resized.

    Args:
        points2d (torch.Tensor):
            2D points in shape (b_size, points_number, 2).
        bbox_xyxy (torch.Tensor):
            Bbox of the cropped area.
            In shape (b_size, 4).
        dst_resolution (torch.Tensor):
            Desination resolution of the cropped area.
            In shape (b_size, 2).

    Returns:
        torch.Tensor:
            Points2d in destination image.
    """
    x_tensor = torch.cat((bbox_xyxy[:, 0:1], bbox_xyxy[:, 2:3]), dim=1)
    upper_left_x, _ = torch.min(x_tensor, dim=1)
    y_tensor = torch.cat((bbox_xyxy[:, 1:2], bbox_xyxy[:, 3:4]), dim=1)
    upper_left_y, _ = torch.min(y_tensor, dim=1)
    bbox_h = torch.abs(y_tensor[:, 0] - y_tensor[:, 1])
    bbox_w = torch.abs(x_tensor[:, 0] - x_tensor[:, 1])
    points2d[:, :, 0] = (points2d[:, :, 0] - upper_left_x)\
        / bbox_w * dst_resolution[:, 0]
    points2d[:, :, 1] = (points2d[:, :, 1] - upper_left_y)\
        / bbox_h * dst_resolution[:, 1]
    return points2d


def get_bbox_xyxy_from_mask(mask: np.ndarray) -> List:
    """Get bbox_xyxy from a one-channel mask.

    Args:
        mask (np.ndarray):
            A cv2 image in shape (height, width),
            filled with 0 and 1.
    Returns:
        List:
            Four intergers, left, top, right, bottom.
    """
    # mask in shape (h, w)
    valid_location = np.where(mask != 0)
    h, w = list(mask.shape[:2])
    top, left, bottom, right = \
        np.min(valid_location[0]), np.min(valid_location[1]),\
        np.max(valid_location[0]), np.max(valid_location[1])
    bbox_h, bbox_w = bottom - top, right - left
    bbox_c = ((top + bottom) / 2, (left + right) / 2)
    # try to expand bbox 1.2 times
    # vertical upper bound
    half_bbox_h_bound = min(bbox_c[1], w - bbox_c[1])
    half_bbox_h = bbox_h * (0.5 + 0.1)
    # horizontal upper bound
    half_bbox_w_bound = min(bbox_c[0], h - bbox_c[0])
    half_bbox_w = bbox_w * (0.5 + 0.1)
    # the longest edge below upper bound
    half_bbox_edge = min(
        max(half_bbox_h, half_bbox_w), half_bbox_h_bound, half_bbox_w_bound)
    bottom = int(bbox_c[0] + half_bbox_edge)
    top = int(bbox_c[0] - half_bbox_edge)
    right = int(bbox_c[1] + half_bbox_edge)
    left = int(bbox_c[1] - half_bbox_edge)
    return left, top, right, bottom
