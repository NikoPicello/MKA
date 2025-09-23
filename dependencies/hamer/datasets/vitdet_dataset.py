from typing import Dict

import cv2
import numpy as np
from skimage.filters import gaussian
from yacs.config import CfgNode
import torch

from .utils import (convert_cvimg_to_tensor,
                    expand_to_aspect_ratio,
                    generate_image_patch_cv2)

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])

class ViTDetDataset(torch.utils.data.Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 img_cv2: np.array,
                 boxes: np.array,
                 right: np.array,
                 rescale_factor=2.5,
                 train: bool = False,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.img_cv2 = img_cv2
        # self.boxes = boxes

        assert train == False, "ViTDetDataset is only for inference"
        self.train = train
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        # Preprocess annotations
        boxes = boxes.astype(np.float32)
        self.center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        self.scale = rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        self.personid = np.arange(len(boxes), dtype=np.int32)
        self.right = right.astype(np.float32)

    def __len__(self) -> int:
        return len(self.personid)

    def __getitem__(self, idx: int) -> Dict[str, np.array]:

        center = self.center[idx].copy()
        center_x = center[0]
        center_y = center[1]

        scale = self.scale[idx]
        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

        patch_width = patch_height = self.img_size

        right = self.right[idx].copy()
        flip = right == 0

        # 3. generate image patch
        # if use_skimage_antialias:
        cvimg = self.img_cv2.copy()
        if True:
            # Blur image to avoid aliasing artifacts
            downsampling_factor = ((bbox_size*1.0) / patch_width)
            # print(f'{downsampling_factor=}')
            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                cvimg  = gaussian(cvimg, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True)


        img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                                                    center_x, center_y,
                                                    bbox_size, bbox_size,
                                                    patch_width, patch_height,
                                                    flip, 1.0, 0,
                                                    border_mode=cv2.BORDER_CONSTANT)
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv)

        # apply normalization
        for n_c in range(min(self.img_cv2.shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        item = {
            'img': img_patch,
            'personid': int(self.personid[idx]),
        }
        item['box_center'] = self.center[idx].copy()
        item['box_size'] = bbox_size
        item['img_size'] = 1.0 * np.array([cvimg.shape[1], cvimg.shape[0]])
        item['right'] = self.right[idx].copy()
        return item

class ViTDetDataset_batch(torch.utils.data.Dataset):
    def __init__(self,
                 cfg: CfgNode,
                 img_cv2_list: list,   # List of images (each a NumPy array, shape: (H, W, 3))
                 boxes_list: list,     # List of boxes arrays; each element is (N, 4) with N >= 1
                 right_list: list,     # List of corresponding right flags; each element is (N,) or similar
                 rescale_factor=2.5,
                 train: bool = False,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        # This dataset is only intended for inference.
        assert train == False, "ViTDetDataset is only for inference"
        self.train = train

        # Build a flattened list of samples.
        # Each sample corresponds to one bounding box from one image.
        self.samples = []
        for i, (img, boxes, rights) in enumerate(zip(img_cv2_list, boxes_list, right_list)):
            boxes = boxes.astype(np.float32)
            rights = np.array(rights).astype(np.float32)
            # Each image may have one or more bounding boxes.
            for j in range(boxes.shape[0]):
                center = (boxes[j, 2:4] + boxes[j, 0:2]) / 2.0
                scale = rescale_factor * (boxes[j, 2:4] - boxes[j, 0:2]) / 200.0
                self.samples.append({
                    'img': img,  # We'll make a copy later in __getitem__
                    'center': center,
                    'scale': scale,
                    'personid': (i, j),  # Composite id: (image index, box index)
                    'right': rights[j],
                    'img_shape': (img.shape[1], img.shape[0])  # (width, height)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        # Copy the image to avoid modifying the original
        cvimg = sample['img'].copy()
        center = sample['center'].copy()
        scale = sample['scale']
        right = sample['right']  # Now a scalar float

        center_x, center_y = center[0], center[1]
        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
        # Compute bbox_size from scale; expand_to_aspect_ratio is assumed available.
        bbox_size = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=BBOX_SHAPE).max()

        patch_width = patch_height = self.img_size

        # Compute the flip flag as a boolean scalar
        flip = (right == 0)

        # Optionally blur image to avoid aliasing artifacts.
        downsampling_factor = (bbox_size / patch_width) / 2.0
        if downsampling_factor > 1.1:
            cvimg = gaussian(cvimg, sigma=(downsampling_factor - 1) / 2, channel_axis=2, preserve_range=True)

        # Generate the image patch using cv2-based function.
        img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                                                       center_x, center_y,
                                                       bbox_size, bbox_size,
                                                       patch_width, patch_height,
                                                       flip, 1.0, 0,
                                                       border_mode=cv2.BORDER_CONSTANT)
        # Convert from BGR to RGB.
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv)

        # Normalize each channel.
        for n_c in range(min(cvimg.shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        item = {
            'img': img_patch,                         # Tensor of shape (C, H, W)
            'personid': sample['personid'],           # Tuple (image index, box index)
            'box_center': center,                     # Center of the bounding box
            'box_size': bbox_size,                    # Scalar size of the box (square)
            'img_size': np.array(sample['img_shape']), # Array [width, height]
            'right': right,                           # Scalar flag
        }
        return item
