import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'


import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import cv2
import datetime
from tqdm import tqdm, trange
from pathlib import Path
from human_models.human_models import SMPLX
from ultralytics import YOLO
from utils.base import Tester
from utils.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
from utils.visualization_utils import render_mesh, render_mesh_pt3d, get_rasterizer, check_visibility_pt3d
from utils.inference_utils import non_max_suppression
from utils.transforms import world2cam, cam2pixel, rigid_align
from pycocotools.coco import COCO
import copy
from time import time
import json
import imageio


joint_set = {
    'joint_num': 17,
    'joints_name': ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Head', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist'),
    'flip_pairs': ( (1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13) ),
    'eval_joint': (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16),
    # 'regressor': np.load(osp.join(data_dir, 'Human36M', 'J_regressor_h36m_smplx.npy'))
}
joint_set['root_joint_idx'] = joint_set['joints_name'].index('Pelvis')


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--skip", action='store_true')
    args = parser.parse_args()
    return args


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args = options()
    in_root = args.video_dir
    out_root = args.out_dir
    os.makedirs(out_root , exist_ok=True)

    # init config
    cudnn.benchmark = True
    ckpt_name = "smplest_x_h"

    config_path = osp.join('./configs', f'config_{ckpt_name}.py')
    cfg = Config.load_config(config_path)
    checkpoint_path = osp.join('./pretrained_models', ckpt_name, f'{ckpt_name}.pth.tar')

    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'inference_human36m_{ckpt_name}_{time_str}'

    new_config = {
        "model": {
            "pretrained_model_path": checkpoint_path,
            # "focal": [1145.5114, 1144.7739],
            # "princpt": [514.9682, 501.88202]
        },
        "log":{
            'exp_name':  exp_name,
            'log_dir': osp.join(out_root, 'outputs', exp_name, 'log'),
            }
    }
    cfg.update_config(new_config)
    cfg.prepare_log()

    # init human models
    smpl_x = SMPLX(cfg.model.human_model_path)
    faces_tensor = torch.from_numpy(smpl_x.face.astype(np.int32)).unsqueeze(0).to(device)

    # init tester
    demoer = Tester(cfg)
    demoer.logger.info(f"Using 1 GPU.")
    demoer.logger.info(f'Inference with [{cfg.model.pretrained_model_path}].')
    demoer._make_model()

    # init detector
    bbox_model = getattr(cfg.inference.detection, "model_path",
                        './pretrained_models/yolov8x.pt')
    detector = YOLO(bbox_model)

    #### run smplestx
    video_filenames = sorted([x for x in os.listdir(in_root) if x.endswith(".mp4") or x.endswith(".mkv")])
    for video_fn in video_filenames:
        video_name = video_fn.split(".")[0]
        out_npy_path = os.path.join(out_root, f"{video_name}_res.npy")
        if args.skip and os.path.exists(out_npy_path):
            print(f"=== skip {out_npy_path}", flush=True)
            continue

        video_path = os.path.join(in_root, video_fn)
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rasterizer = get_rasterizer(frame_height, frame_width)

        out_video_path = os.path.join(out_root, f"{video_name}_render.mp4")
        writer = imageio.get_writer(
            out_video_path,
            fps=fps, mode='I', format='FFMPEG', macro_block_size=1
        )

        out_results = []
        for fidx in trange(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            out_frame_dict = {
                "fidx": fidx,
            }
            transform = transforms.ToTensor()
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            original_img = img_rgb.copy().astype(np.float32)
            vis_img = original_img.copy()
            original_img_height, original_img_width = original_img.shape[:2]
            # detection, xyxy
            yolo_bbox = detector.predict(original_img,
                                    device='cuda',
                                    classes=00,
                                    conf=cfg.inference.detection.conf,
                                    save=cfg.inference.detection.save,
                                    verbose=cfg.inference.detection.verbose
                                        )[0].boxes.xyxy.detach().cpu().numpy()

            if len(yolo_bbox)<1:
                # save original image if no bbox
                num_bbox = 0
                writer.append_data(vis_img.astype(np.uint8))

                out_results.append(out_frame_dict)
                print("=== failed", fidx, flush=True)
                continue

            #### only focus on max bbox
            num_bbox = len(yolo_bbox)
            if num_bbox > 1:
                bbox_id = np.argmax(abs(yolo_bbox[:, 2] - yolo_bbox[:, 0]) * abs(yolo_bbox[:, 3] - yolo_bbox[:, 1]))
            else:
                bbox_id = 0

            yolo_bbox_xywh = np.zeros((4))
            yolo_bbox_xywh[0] = yolo_bbox[bbox_id][0]
            yolo_bbox_xywh[1] = yolo_bbox[bbox_id][1]
            yolo_bbox_xywh[2] = abs(yolo_bbox[bbox_id][2] - yolo_bbox[bbox_id][0])
            yolo_bbox_xywh[3] = abs(yolo_bbox[bbox_id][3] - yolo_bbox[bbox_id][1])

            # xywh
            bbox = process_bbox(bbox=yolo_bbox_xywh,
                                img_width=original_img_width,
                                img_height=original_img_height,
                                input_img_shape=cfg.model.input_img_shape,
                                ratio=getattr(cfg.data, "bbox_ratio", 1.25))

            focal = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2],
                    cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
            princpt = [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0],
                    cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]

            out_frame_dict["focal"] = focal
            out_frame_dict["princpt"] = princpt

            img, _, _ = generate_patch_image(cvimg=original_img,
                                                bbox=bbox,
                                                scale=1.0,
                                                rot=0.0,
                                                do_flip=False,
                                                out_shape=cfg.model.input_img_shape)

            img = transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')

            smplx_output = out["smplx_output"]
            mesh_cam = out["smplx_mesh_cam"]
            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

            # generate confidence based on visibility
            points_visibility = check_visibility_pt3d(rasterizer, vis_img, mesh_cam, faces_tensor, {'focal': focal, 'princpt': princpt})
            new_joints_img = demoer.model.module.get_joints_visibility(smplx_output, faces_tensor, points_visibility)
            new_joints_img[:, 0] = new_joints_img[:, 0] * bbox[2] / cfg.model.output_hm_shape[2] + bbox[0]
            new_joints_img[:, 1] = new_joints_img[:, 1] * bbox[3] / cfg.model.output_hm_shape[1] + bbox[1]

            out_frame_dict["kpt2d"] = new_joints_img
            out_frame_dict["betas"] = smplx_output.betas[0].cpu().detach().float().numpy()
            out_frame_dict["expression"] = smplx_output.expression[0].cpu().detach().float().numpy()
            out_frame_dict["full_pose"] = smplx_output.full_pose[0].cpu().detach().float().numpy()
            out_frame_dict["transl"] = smplx_output.transl[0].cpu().detach().float().numpy()
            out_results.append(out_frame_dict)

            vis_img = render_mesh_pt3d(vis_img, mesh_cam, faces_tensor, {'focal': focal, 'princpt': princpt}, rasterizer)
            writer.append_data(vis_img.astype(np.uint8))

        cap.release()
        writer.close()

        np.save(out_npy_path, out_results)


if __name__ == '__main__':
    main()
    print("=== done", flush=True)
