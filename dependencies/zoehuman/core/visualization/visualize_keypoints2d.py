import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from mmhuman3d.core.visualization.visualize_keypoints2d import (  # noqa:F401
    _CavasProducer, _check_temp_path, _plot_kp2d_frame, _prepare_limb_palette,
    _prepare_output_path, visualize_kp2d,
)
from tqdm import tqdm

from zoehuman.utils.ffmpeg_utils import images_to_video, pad_for_libx264
from zoehuman.utils.path_utils import check_path_suffix


def visualize_kp2d_multiperson(
    kp2d_list: list,
    output_path: str,
    frame_list: Optional[List[str]] = None,
    limbs: Optional[Union[np.ndarray, List[int]]] = None,
    data_source: str = 'coco_wholebody',
    mask: Optional[Union[list, np.ndarray]] = None,
    overwrite: bool = False,
    with_file_name: bool = True,
    resolution: Optional[Union[Tuple[int, int], list]] = None,
    fps: Union[float, int] = 30,
    draw_bbox: bool = False,
    with_number: bool = False,
    disable_tqdm: bool = False,
) -> None:
    """Visualize 2d keypoints into a video or into a folder of frames. No color
    difference between people.

    Args:
        kp2d_list (list):
            List of keypoint2d numpy array. An element should be
            an array of shape (f * J * 2) or (f * n * J * 2)]
        output_path (str): output video path or image folder.
        frame_list (List[str]): list of frame paths, if None, would be
                initialized as white background.
        limbs (Optional[Union[np.ndarray, List[int]]], optional):
                if not specified, the limbs will be searched by search_limbs,
                this option is for free skeletons like BVH file.
                Defaults to None.
        data_source (str, optional): data source type.
            Defaults to 'coco_wholebody'.
        mask (Optional[Union[list, np.ndarray]], optional):
                mask to mask out the incorrect points. Defaults to None.
        overwrite (bool, optional): whether replace the origin frames.
                Defaults to False.
        with_file_name (bool, optional): whether write origin frame name on
                the images. Defaults to True.
        resolution (Optional[Union[Tuple[int, int], list]], optional):
                (width, height) of the output video
                will be the same size as the original images if not specified.
                Defaults to None.
        fps (Union[float, int], optional): fps. Defaults to 30.
        draw_bbox (bool, optional): whether need to draw bounding boxes.
                Defaults to False.
        with_number (bool, optional): whether draw index number.
                Defaults to False.
        disable_tqdm (bool, optional):
            Whether to disable the entire progressbar wrapper.
            Defaults to False.

    Raises:
        FileNotFoundError: check input frame paths.

    Returns:
        NoReturn.
    """
    assert len(kp2d_list) > 0
    # check output path
    temp_folder = _prepare_output_path(output_path, overwrite)

    # check whether temp_folder will overwrite frame_list by accident
    _check_temp_path(temp_folder, frame_list, overwrite)

    if check_path_suffix(output_path, ['']):
        output_type = 'frames'
    else:
        output_type = 'video'

    # search the limb connections and palettes from superset smplx
    limbs_target, limbs_palette = _prepare_limb_palette(
        limbs, None, [], data_source, mask)

    kp2d_sample = kp2d_list[0].copy()
    # check the input array shape, reshape to (num_frame, num_person, J, 2)
    kp2d_sample = kp2d_sample[..., :2]
    if kp2d_sample.ndim == 3:
        kp2d_sample = kp2d_sample[:, np.newaxis]
    assert kp2d_sample.ndim == 4

    # setup canvas producer from frame files or np.ones()
    canvas_producer = _CavasProducer(frame_list, resolution, kp2d_sample)

    # assign keypoints from kp2d_list to kp2d_np
    kp2d_np = np.zeros(shape=[
        len(kp2d_list), kp2d_sample.shape[0], kp2d_sample.shape[1],
        kp2d_sample.shape[2], kp2d_sample.shape[3]
    ])
    for person_index, kp2d in enumerate(kp2d_list):
        kp2d_sample = kp2d.copy()
        kp2d_sample = kp2d_sample[..., :2]
        if kp2d_sample.ndim == 3:
            kp2d_sample = kp2d_sample[:, np.newaxis]
        kp2d_np[person_index] = kp2d_sample

    # start plotting by frame
    for frame_index in tqdm(range(kp2d_sample.shape[0]), disable=disable_tqdm):
        image_array, _ = canvas_producer[frame_index]
        image_array = pad_for_libx264(image_array)
        # start plotting by person
        for person_index in range(kp2d_np.shape[0]):
            image_array = _plot_kp2d_frame(
                kp2d_np[person_index, frame_index, :, :],
                image_array,
                limbs_target,
                limbs_palette,  # people_palette[person_index],
                draw_bbox=draw_bbox,
                with_number=with_number,
                font_size=0.5)
        # write the frame with opencv
        if with_file_name:
            h, w, _ = image_array.shape
            if frame_index < len(frame_list):
                text_to_add = str(Path(frame_list[frame_index]).name)
            else:
                text_to_add = ''
            cv2.putText(image_array, text_to_add, (w // 2, h // 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * h / 500,
                        np.array([255, 255, 255]).astype(np.int32).tolist(), 2)
        # write the frame with opencv
        if output_type == 'video':
            frame_path = \
                os.path.join(temp_folder, f'{frame_index:06d}.png')
        else:
            frame_path = \
                os.path.join(temp_folder, Path(frame_list[frame_index]).name)
        cv2.imwrite(frame_path, image_array)
    # convert frames to video
    if output_type == 'video':
        images_to_video(
            input_folder=temp_folder,
            output_path=output_path,
            remove_raw_file=True,
            fps=fps)
