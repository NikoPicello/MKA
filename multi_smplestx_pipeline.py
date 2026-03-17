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
import cv2 as cv
import datetime
import glob
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
from pathlib import Path
import json
import imageio
import time

joint_set = {
  'joint_num': 17,
  'joints_name': ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Head', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist'),
  'flip_pairs': ( (1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13) ),
  'eval_joint': (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16),
  # 'regressor': np.load(osp.join(data_dir, 'Human36M', 'J_regressor_h36m_smplx.npy'))
}
joint_set['root_joint_idx'] = joint_set['joints_name'].index('Pelvis')

save_render = False

SPATIAL_REGIONS = {
  'GB' : {0 : [0., 0.5, 0., 1.], 1 : [0.5, 1., 0., 1.]},
  'GF' : {0 : [0.5, 1., 0., 1.], 1 : [0., 0.5, 0., 1.]},
  'FC1' : {0 : [0.25, 0.75, 0., 1.]},
  'FC2' : {1 : [0.25, 0.75, 0., 1.]},
  'HA1' : {0 : [0.25, 0.75, 0., 1.]},
  'HA2' : {1 : [0.25, 0.75, 0., 1.]}
}

activities = ['animals', 'gaze', 'ghost', 'lego', 'talk']

# ============================================================================
# CONVERT BBOXES TO NORMALIZED COORDINATES
# ============================================================================

def xyxy_to_normalized(bbox_xyxy, img_width, img_height):
  """
  Convert bbox from pixel coordinates to normalized [0, 1].

  Args:
      bbox_xyxy: [x1, y1, x2, y2] in pixels
      img_width, img_height: image dimensions

  Returns:
      [x_min, x_max, y_min, y_max] normalized to [0, 1]
  """
  x1, y1, x2, y2 = bbox_xyxy

  x_min = x1 / img_width
  x_max = x2 / img_width
  y_min = y1 / img_height
  y_max = y2 / img_height

  return np.array([x_min, x_max, y_min, y_max])


def xyxy_to_xywh(bbox_xyxy):
  """
  Convert from [x1, y1, x2, y2] to [x, y, w, h].

  Useful for cropping or processing with your SMPL-X code.
  """
  x1, y1, x2, y2 = bbox_xyxy
  return np.array([x1, y1, x2 - x1, y2 - y1])


def bbox_centroid(bbox_xyxy):
  """Get center point of bbox."""
  x1, y1, x2, y2 = bbox_xyxy
  cx = (x1 + x2) / 2
  cy = (y1 + y2) / 2
  return cx, cy


# ============================================================================
# ASSIGN BBOX TO PERSON BASED ON SPATIAL REGION
# ============================================================================

def assign_bbox_to_person(bbox_normalized, cam_id):
  """
  Assign a bbox to a person using spatial regions.

  Args:
      bbox_normalized: [x_min, x_max, y_min, y_max] in [0, 1]
      cam_id: camera name (e.g., "GB", "FC1")

  Returns:
      person_id: 0 or 1, or None if bbox doesn't match any region
  """

  if cam_id not in SPATIAL_REGIONS:
    print(f"Unknown camera: {cam_id}")
    return None

  # Get bbox center
  x_center = (bbox_normalized[0] + bbox_normalized[1]) / 2
  y_center = (bbox_normalized[2] + bbox_normalized[3]) / 2

  # Check which person regions match this bbox
  cam_regions = SPATIAL_REGIONS[cam_id]

  for person_id, region in cam_regions.items():
    x_min, x_max, y_min, y_max = region

    # Is centroid inside this region?
    if x_min <= x_center <= x_max and y_min <= y_center <= y_max:
      return person_id

  # Bbox doesn't match any region (probably background noise or occlusion)
  return None

def filter_and_assign_bboxes(boxes_xyxy, confidences, img_width, img_height,
                           cam_id, conf_threshold=0.5):
  """
  Filter bboxes by confidence and assign to persons.

  Args:
      boxes_xyxy: (N, 4) array of detections
      confidences: (N,) array of scores
      img_width, img_height: image dimensions
      cam_id: camera name
      conf_threshold: minimum confidence (0-1)

  Returns:
      assignments: dict {person_id: [x1, y1, x2, y2]} or empty if no valid detections
  """

  assignments = {}

  # Filter by confidence
  valid_idx = np.where(confidences >= conf_threshold)[0]

  if len(valid_idx) == 0:
    return assignments  # No detections

  valid_boxes = boxes_xyxy[valid_idx]
  valid_confs = confidences[valid_idx]

  # For each valid detection, try to assign to a person
  for bbox_xyxy, conf in zip(valid_boxes, valid_confs):
    # Convert to normalized coordinates
    bbox_norm = xyxy_to_normalized(bbox_xyxy, img_width, img_height)

    # Assign to person based on spatial region
    person_id = assign_bbox_to_person(bbox_norm, cam_id)

    if person_id is not None:
      # If multiple detections for same person, keep the highest confidence
      if person_id not in assignments or conf > assignments[person_id][4]:
        assignments[person_id] = np.append(bbox_xyxy, conf)

  return assignments

def main():
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  main_path = '/'.join(sys.path[0].split('/')[:-2]) + '/'
  resources_path = os.path.join(main_path, 'resources')
  calibs_path   = os.path.join(resources_path, 'calibs')
  sessions_path = os.path.join(resources_path, 'sessions')
  out_path = os.path.join(resources_path, 'smplestx_results')
  sid_paths = sorted(glob.glob(sessions_path + '/*'))

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
      'log_dir': osp.join(out_path, 'outputs', exp_name, 'log'),
      }
  }
  cfg.update_config(new_config)
  cfg.prepare_log()

  smpl_x = SMPLX(cfg.model.human_model_path)
  faces_tensor = torch.from_numpy(smpl_x.face.astype(np.int32)).unsqueeze(0).to(device)

  demoer = Tester(cfg)
  demoer.logger.info(f"Using 1 GPU.")
  demoer.logger.info(f'Inference with [{cfg.model.pretrained_model_path}].')
  demoer._make_model()

  # init detector
  bbox_model = getattr(cfg.inference.detection, "model_path",
                      './pretrained_models/yolov8x.pt')
  detector = YOLO(bbox_model)

  for sid_path in sid_paths:
    session_id = Path(sid_path).stem
    for activity in activities:
      print(f'Processing {activity} for session {session_id}')
      vid_paths = glob.glob(os.path.join(sid_path, activity) + '/*')
      vid_paths = [v for v in vid_paths if not ('E1.mp4' in v or 'E2.mp4' in v)]
      for vid_path in vid_paths:
        video_name = Path(vid_path).stem
        cap = cv.VideoCapture(vid_path)
        fps = int(cap.get(cv.CAP_PROP_FPS))
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        curr_out_path = os.path.join(out_path, f"{session_id}/{activity}")
        os.makedirs(curr_out_path, exist_ok=True)
        out_npy_path = os.path.join(curr_out_path, f"{video_name}_res.npy")

        if save_render:
          rasterizer = get_rasterizer(frame_height, frame_width)
          out_vid_path = os.path.join(curr_out_path, f"{video_name}_render.mp4")
          writer = imageio.get_writer(
              out_vid_path,
              fps=fps, mode='I', format='FFMPEG', macro_block_size=1
          )

        out_results = []
        for fidx in trange(total_frames):
          ret, frame = cap.read()
          if not ret: break

          out_frame_dict = {
            "fidx": fidx,
          }
          transform = transforms.ToTensor()
          img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
          original_img = img_rgb.copy().astype(np.float32)
          vis_img = original_img.copy()

          results = detector.predict(original_img,
                                      device='cuda:3',
                                      classes=00,
                                      conf=cfg.inference.detection.conf,
                                      save=cfg.inference.detection.save,
                                      verbose=cfg.inference.detection.verbose
          )
          boxes_xyxy = results[0].boxes.xyxy.detach().cpu().numpy()
          confidences = results[0].boxes.conf.detach().cpu().numpy()

          assignments = filter_and_assign_bboxes(
              boxes_xyxy, confidences, frame_width, frame_height, video_name, conf_threshold=0.5
          )

          for person_id, bbox_with_conf in assignments.items():
            curr_frame_dict = {}
            bbox_xyxy = bbox_with_conf[:4]
            conf = bbox_with_conf[4]
            bbox_xywh = xyxy_to_xywh(bbox_xyxy)
            # xywh
            bbox = process_bbox(bbox=bbox_xywh,
                                img_width=frame_width,
                                img_height=frame_height,
                                input_img_shape=cfg.model.input_img_shape,
                                ratio=getattr(cfg.data, "bbox_ratio", 1.25))

            focal = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2],
                    cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
            princpt = [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0],
                    cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]

            curr_frame_dict["focal"] = focal
            curr_frame_dict["princpt"] = princpt

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

            with torch.no_grad():
              out, smplx_output = demoer.model(inputs, targets, meta_info, 'test')

            mesh_cam = out["smplx_mesh_cam"]
            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

            # generate confidence based on visibility
            new_joints_img = demoer.model.module.not_get_joints_visibility(smplx_output)
            new_joints_img[:, 0] = new_joints_img[:, 0] * bbox[2] / cfg.model.output_hm_shape[2] + bbox[0]
            new_joints_img[:, 1] = new_joints_img[:, 1] * bbox[3] / cfg.model.output_hm_shape[1] + bbox[1]

            curr_frame_dict["kpt2d"] = new_joints_img
            curr_frame_dict["betas"] = smplx_output['betas'][0].cpu().detach().float().numpy()
            curr_frame_dict["expression"] = smplx_output['expression'][0].cpu().detach().float().numpy()
            curr_frame_dict["full_pose"] = smplx_output['full_pose'][0].cpu().detach().float().numpy()
            curr_frame_dict["transl"] = smplx_output['transl'][0].cpu().detach().float().numpy()
            out_frame_dict[person_id] = curr_frame_dict
            if save_render:
              vis_img = render_mesh_pt3d(vis_img, mesh_cam, faces_tensor, {'focal': focal, 'princpt': princpt}, rasterizer)

          out_results.append(out_frame_dict)
          if save_render:
            vis_image = cv.cvtColor(vis_img, cv.COLOR_BGR2RGB)
            writer.append_data(vis_img.astype(np.uint8))


        if save_render:
          writer.close()
        cap.release()
        np.save(out_npy_path, out_results)

if __name__ == '__main__':
  main()

