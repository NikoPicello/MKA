from pathlib import Path
import torch
import argparse
import os
import os.path as osp 
import cv2
import smplx
from smplx.joint_names import JOINT_NAMES
MANO_RIGHT_KEYPOINTS = [
    'right_wrist', 'right_index1', 'right_index2', 'right_index3',
    'right_middle1', 'right_middle2', 'right_middle3', 'right_pinky1',
    'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2',
    'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3',
    'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky'
]
MANO_LEFT_KEYPOINTS = [
    x.replace('right_', 'left_') for x in MANO_RIGHT_KEYPOINTS
]

import sys 
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path, "dependencies"))

import numpy as np
from tqdm import tqdm, trange
from itertools import combinations
import datetime
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.models.mano_wrapper import smpl_x
from hamer.utils import recursive_to
from hamer.utils.geometry import (
    matrix_to_axis_angle,
    matrix_to_euler_angles,
    axis_angle_to_matrix,
)
from hamer.datasets.vitdet_dataset import (
    ViTDetDataset,
    ViTDetDataset_batch,
    DEFAULT_MEAN,
    DEFAULT_STD,
)
from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.utils.geometry import perspective_projection, aa_to_rotmat, matrix_to_axis_angle
from hamer.mv.k4d_camera import *
from pycocotools.coco import COCO
from collections import defaultdict
from hamer.mv.joint_vis import (
    get_rasterizer,
    select_best_view,
    run_smplx_with_mano,
)
from hamer.mv.joint_vis_orig import rigid_align_batched

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

from models.vitpose_model import ViTPoseModel

import json
import copy
from typing import Dict, Optional
from scipy.spatial.transform import Rotation as R


def save_verts(verts, file_name='a.obj'):
    with open(file_name, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / cam_coord[:, 2] * f[0] + c[0]
    y = cam_coord[:, 1] / cam_coord[:, 2] * f[1] + c[1]
    z = cam_coord[:, 2]
    return np.stack((x, y, z), 1)

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)
    return cam_coord

def pre_detection(original_imgs, detector, cpm):
    rkeyp = []
    lkeyp = []
    all_boxs = []
    all_right = []
    confidences = np.zeros((4, 42))
    for view, img_cv2 in enumerate(original_imgs):
        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()
        if pred_bboxes.shape[0] > 1:
            box_id = np.argmax(
                (pred_bboxes[:, 2] - pred_bboxes[:, 0])
                * (pred_bboxes[:, 3] - pred_bboxes[:, 1])
            )
            pred_bboxes = pred_bboxes[box_id : box_id + 1]
            pred_scores = pred_scores[box_id : box_id + 1]

        # Detect human keypoints for each person
        vitposes_out, layer_outputs = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        hand_keyp = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes["keypoints"][-42:-21]
            right_hand_keyp = vitposes["keypoints"][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.7
            if sum(valid) > 3:
                bbox = [
                    keyp[valid, 0].min(),
                    keyp[valid, 1].min(),
                    keyp[valid, 0].max(),
                    keyp[valid, 1].max(),
                ]
                if (view == 0 and bbox[3] > 300) or (view != 0):
                    bboxes.append(bbox)
                    hand_keyp.append(vitposes["keypoints"][-42:-21])
                    confidences[view, :21] = keyp[:, 2]
                    is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.7
            if sum(valid) > 3:
                bbox = [
                    keyp[valid, 0].min(),
                    keyp[valid, 1].min(),
                    keyp[valid, 0].max(),
                    keyp[valid, 1].max(),
                ]
                if (view == 0 and bbox[3] > 300) or (view != 0):
                    bboxes.append(bbox)
                    hand_keyp.append(vitposes["keypoints"][-21:])
                    confidences[view, 21:] = keyp[:, 2]
                    is_right.append(1)

        if len(bboxes) == 0:
            all_boxs.append(np.array([]))
            all_right.append(np.array([]))
            continue
        boxes = np.stack(bboxes)
        hand_keyp = np.stack(hand_keyp)
        right = np.stack(is_right)

        if view == 0 and len(bboxes) > 2:
            e_bbox = list(enumerate(bboxes))
            e_bbox = sorted(e_bbox, key=lambda x: x[1][1], reverse=True)
            idx1, idx2 = e_bbox[0][0], e_bbox[1][0]
            boxes = boxes[[idx1, idx2], :]
            hand_keyp = hand_keyp[[idx1, idx2], ...]
            right = right[[idx1, idx2]]

        if hand_keyp[right == 0].shape[0] != 0:
            lkeyp.append(hand_keyp[right == 0][0])
        if hand_keyp[right == 1].shape[0] != 0:
            rkeyp.append(hand_keyp[right == 1][0])
        all_boxs.append(boxes)
        all_right.append(right)
    
    return rkeyp, lkeyp, all_boxs, all_right, confidences

def run_hamer(dataloader, original_imgs, model_cfg, model, renderer):
    device = "cuda"
    batch = next(iter(dataloader))
    batch = recursive_to(batch, device)
    
    with torch.no_grad():
        out = model(batch)
    multiplier = 2 * batch["right"] - 1
    pred_cam = out["pred_cam"]
    pred_cam[:, 1] = multiplier * pred_cam[:, 1]
    box_center = batch["box_center"].float()
    box_size = batch["box_size"].float()
    img_size = batch["img_size"].float()
    multiplier = 2 * batch["right"] - 1
    scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
    pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

    # for n in range(8):
    #     regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
    #         out['pred_cam_t'][n].detach().cpu().numpy(),
    #         batch['img'][n],
    #         mesh_base_color=LIGHT_BLUE,
    #         scene_bg_color=(1, 1, 1),
    #     )
    #     print("regression", type(regression_img), flush=True)
    #     cv2.imwrite(f"./check_hawor_{n}.jpg", regression_img[..., ::-1] * 255)
    # exit(0)
    
    cam_param = [
        torch.tensor([[scaled_focal_length, scaled_focal_length]]),
        torch.tensor([[img_size[0][0] / 2.0, img_size[0][1] / 2.0]]),
    ]
    out["pred_keypoints_3d"][..., 0] = (
        multiplier[..., None] * out["pred_keypoints_3d"][..., 0]
    )
    keypoints2d = perspective_projection(
        out["pred_keypoints_3d"],
        torch.tensor(pred_cam_t_full).to(device),
        cam_param[0],
        cam_param[1]
    ).reshape(-1, 2).detach().cpu().numpy()

    keypoints2d = torch.tensor(keypoints2d).cuda()[None, ...]
    keypoints2d = keypoints2d.reshape(batch["right"].shape[0], -1, 2)
    rhand2d = keypoints2d[batch["right"] == 1]
    lhand2d = keypoints2d[batch["right"] == 0]
    rhand3d = out["pred_keypoints_3d"][batch["right"] == 1]
    lhand3d = out["pred_keypoints_3d"][batch["right"] == 0]
    lview = torch.unique(batch["personid"][0][batch["right"] == 0])
    rview = torch.unique(batch["personid"][0][batch["right"] == 1])
    lshape = out["shape"][batch["right"] == 0]
    rshape = out["shape"][batch["right"] == 1]
    # for i in range(4):
    #     img_cv = original_imgs[i]

    #     for kpt2d in lhand2d[i].cpu().numpy():
    #         x = int(kpt2d[0])
    #         y = int(kpt2d[1])
    #         cv2.circle(img_cv, (x, y), 3, (0, 255, 0), -1)

    #     for kpt2d in rhand2d[i].cpu().numpy():
    #         x = int(kpt2d[0])
    #         y = int(kpt2d[1])
    #         cv2.circle(img_cv, (x, y), 3, (0, 0, 255), -1)
        
    #     cv2.imwrite(f"./check_kpt2d_{i:02d}.jpg", img_cv)
    # # exit(0)

    l_init_root_pose = matrix_to_axis_angle(
        out["root_pose"][batch["right"] == 0]
    ).flatten()[None, ...]
    l_init_hand_pose = matrix_to_axis_angle(
        out["hand_pose"][batch["right"] == 0]
    ).flatten()[None, ...]
    r_init_root_pose = matrix_to_axis_angle(
        out["root_pose"][batch["right"] == 1]
    ).flatten()[None, ...]
    r_init_hand_pose = matrix_to_axis_angle(
        out["hand_pose"][batch["right"] == 1]
    ).flatten()[None, ...]
    l_cam_t = torch.tensor(pred_cam_t_full, device=device)[
        batch["right"] == 0, ...
    ]
    r_cam_t = torch.tensor(pred_cam_t_full, device=device)[
        batch["right"] == 1, ...
    ]
    lhand_res = {
        "lhand2d": lhand2d,
        "lhand3d": lhand3d,
        "lview": lview,
        "lshape": lshape,
        "l_init_root_pose": l_init_root_pose,
        "l_init_hand_pose": l_init_hand_pose,
        "l_cam_t": l_cam_t,
        "hand_pose": out["hand_pose"],
        "root_pose": out["root_pose"],
        "is_right": batch["right"],
        "personid": batch["personid"],
    }
    rhand_res = {
        "rhand2d": rhand2d,
        "rhand3d": rhand3d,
        "rview": rview,
        "rshape": rshape,
        "r_init_root_pose": r_init_root_pose,
        "r_init_hand_pose": r_init_hand_pose,
        "r_cam_t": r_cam_t,
    }

    return lhand_res, rhand_res

def get_pose_c2w(pose_cam, c2w):
    batch = pose_cam.size(0)
    device = pose_cam.device 
    pose_mat_cam = aa_to_rotmat(pose_cam.reshape(-1, 3)).reshape(batch, -1, 3, 3)
    pose_mat_world = torch.matmul(c2w.to(device), pose_mat_cam)
    pose_world = matrix_to_axis_angle(pose_mat_world).reshape(batch, 3)
    return pose_world

def eval(gt_kpt3d, res_kpt3d):
    tgt_kpt3d = gt_kpt3d.copy()
    tgt_kpt3d_rel = tgt_kpt3d - tgt_kpt3d[0:1]
    
    src_kpt3d = res_kpt3d.copy()
    src_kpt3d_rel = src_kpt3d - src_kpt3d[0:1]

    src_aligned_rel = rigid_align_batched(src_kpt3d_rel[None,], tgt_kpt3d_rel[None,])

    mpjpe = np.sqrt(np.sum((src_kpt3d_rel - tgt_kpt3d_rel) ** 2, -1)).mean()
    pampjpe = np.sqrt(np.sum((src_aligned_rel - tgt_kpt3d_rel) ** 2, -1)).mean()
    tmpjpe = np.sqrt(np.sum((src_kpt3d - tgt_kpt3d) ** 2, -1)).mean()
            
    return mpjpe, pampjpe, tmpjpe  

def get_mano_idx():
    mano_idx_arr = []
    for name in MANO_LEFT_KEYPOINTS:
        idx = JOINT_NAMES.index(name)
        mano_idx_arr.append(idx)

    for name in MANO_RIGHT_KEYPOINTS:
        idx = JOINT_NAMES.index(name)
        mano_idx_arr.append(idx)
    return mano_idx_arr


def parse_args():
    parser = argparse.ArgumentParser(description="HaMeR demo code")
    parser.add_argument('--checkpoint', type=str, default=f'{root_path}/pretrained_models/hamer_ckpts/checkpoints/hamer.ckpt', help='Path to pretrained model checkpoint')
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    os.makedirs(args.out_dir , exist_ok=True)

    model, model_cfg = load_hamer(args.checkpoint)
    # Setup HaMeR model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

    if args.body_detector == "vitdet":
        from detectron2.config import LazyConfig
        import hamer

        cfg_path = (
            Path(hamer.__file__).parent
            / "configs"
            / "cascade_mask_rcnn_vitdet_h_75ep.py"
        )
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == "regnety":
        from detectron2 import model_zoo
        from detectron2.config import get_cfg

        detectron2_cfg = model_zoo.get_config(
            "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
        )
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    curr_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    # human_model_dir = os.path.join(curr_dir, "h3wb", "human_models")
    # smplx_model = smplx.create(
    #     human_model_dir,
    #     model_type="smplx",
    #     gender="neutral",
    #     ext="npz",
    #     flat_hand_mean=True,
    #     use_pca=False,
    #     use_face_contour=True,
    # )
    # mano_model = smplx.create(
    #     human_model_dir, "mano", gender="NEUTRAL", use_pca=False, flat_hand_mean=True
    # )
    # smplx_model.to(device)
    # mano_idx_arr = get_mano_idx()
    
    video_filenames = sorted([x for x in os.listdir(args.video_dir) if x.endswith(".mp4") or x.endswith(".mkv")])
    print("=== len", len(video_filenames), flush=True)
    
    for video_fn in video_filenames:
        video_name = video_fn.split(".")[0]
        out_npy_path = os.path.join(args.out_dir, f"{video_name}_res.npy")
        video_path = os.path.join(args.video_dir, video_fn) 
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        out_results = []
        for fidx in trange(total_frames):
            ret, frame = cap.read()
            if not ret:
                break 

            bgr_img = frame.copy()
            original_imgs = [bgr_img]

            rkeyp, lkeyp, all_boxs, all_right, confidences = pre_detection(original_imgs, detector, cpm)
        
            # Run reconstruction on all detected hands
            dataset = ViTDetDataset_batch(
                model_cfg,
                original_imgs,
                all_boxs,
                all_right,
                rescale_factor=args.rescale_factor,
            )
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=8, shuffle=False, num_workers=0
            )
            if len(dataloader) < 1:
                continue 
                
            lhand_res, rhand_res = run_hamer(dataloader, original_imgs, model_cfg, model, renderer)
            lhand_res['lkeyp'] = lkeyp 
            rhand_res['rkeyp'] = rkeyp

            lhand_res = {k: v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v for k, v in lhand_res.items()}
            rhand_res = {k: v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v for k, v in rhand_res.items()}
            out_results.append( {
                'left': lhand_res,
                'right': rhand_res,
                'fidx': fidx,
            })
        
        np.save(out_npy_path, out_results)
    
if __name__ == '__main__':
    main()
    print("=== done", flush=True)