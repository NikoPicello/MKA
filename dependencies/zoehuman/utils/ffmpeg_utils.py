from typing import Optional, Tuple, Union

import numpy as np
from mmhuman3d.utils.ffmpeg_utils import array_to_images as array_to_images_mm
from mmhuman3d.utils.ffmpeg_utils import (  # noqa: F401
    array_to_video, compress_video, gif_to_images, gif_to_video,
    images_to_array, images_to_gif, images_to_video, pad_for_libx264,
    spatial_concat_video, temporal_concat_video, video_to_array, video_to_gif,
    video_to_images,
)


def array_to_images(
    image_array: np.ndarray,
    output_folder: str,
    img_format: str = 'frame_%06d.jpg',
    resolution: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None,
    disable_log: bool = False,
) -> None:
    """Convert an array to images directly.

    Args:
        image_array (np.ndarray): shape should be (f * h * w * 3).
        output_folder (str): output folder for the images.
        img_format (str, optional): format of the images.
                Defaults to '%06d.png'.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
                optional): resolution(width, height) of output.
                Defaults to None.

    Raises:
        FileNotFoundError: check output folder.
        TypeError: check input array.

    Returns:
        NoReturn
    """
    array_to_images_mm(image_array, output_folder, img_format, resolution,
                       disable_log)
