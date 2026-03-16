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

cam_map = {
  'GC' : 'GB',
  'HC' : 'GF',
  'Z1' : 'FC1',
  'Z2' : 'FC2',
  'N1' : 'HA1',
  'N2' : 'HA2'
}

activities = ['animals', 'gaze', 'ghost', 'lego', 'talk']
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
    with open(os.path.join(sid_path, 'session_data.txt')) as f:
      lines = f.readlines()
      calib_date = lines[1][11:].strip()
    curr_calib_path = os.path.join(calibs_path, calib_date)
    cam_calibs = glob.glob(curr_calib_path + '/*')
    cam_dict = {}
    for cam_calib in cam_calibs:
      cam_name = Path(cam_calib).stem
      fs = cv.FileStorage(os.path.join(calibs_path, f"{calib_date}/{cam_name}.yml"), cv.FILE_STORAGE_READ)
      K = fs.getNode('K').mat()
      D = fs.getNode('D').mat()
      R = fs.getNode('R').mat()
      T = fs.getNode('T').mat()
      fs.release()
      cam_dict[cam_map[cam_name]] = {'K' : K, 'D' : D, 'R' : R, 'T' : T}

    for activity in activities:
      vid_paths = glob.glob(os.path.join(sid_path, activity) + '/*')
      vid_paths = [v for v in vid_paths if not ('E1.mp4' in v or 'E2.mp4' in v)]
      for vid_path in vid_paths:
        video_name = Path(vid_path).stem
        K = cam_dict[video_name]['K']
        # cfg.model.focal = [K[0, 0], K[1, 1]]
        # cfg.model.princpt = [K[0, 2], K[1, 2]]

        cap = cv.VideoCapture(vid_path)
        fps = int(cap.get(cv.CAP_PROP_FPS))
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        rasterizer = get_rasterizer(frame_height, frame_width)

        out_vid_path = os.path.join(out_path, f"{video_name}_render.mp4")
        out_npy_path = os.path.join(out_path, f"{video_name}_res.npy")
        # writer = imageio.get_writer(
        #     out_vid_path,
        #     fps=fps, mode='I', format='FFMPEG', macro_block_size=1
        # )

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
          original_img_height, original_img_width = original_img.shape[:2]
          # detection, xyxy
          # start = time.time()
          yolo_bbox = detector.predict(original_img,
                                      device='cuda:3',
                                      classes=00,
                                      conf=cfg.inference.detection.conf,
                                      save=cfg.inference.detection.save,
                                      verbose=cfg.inference.detection.verbose
          )[0].boxes.xyxy.detach().cpu().numpy()

          if len(yolo_bbox)<1:
            # save original image if no bbox
            num_bbox = 0
            # writer.append_data(vis_img.astype(np.uint8))

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
          # print('bbox processing')
          # print(time.time() - start)

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
          # print('mesh recovery')
          # start = time.time()
          with torch.no_grad():
            out, smplx_output = demoer.model(inputs, targets, meta_info, 'test')
          # print(time.time() - start)

          # print('out 0')
          # print(out[0])
          # print('out 1')
          # print(out[1])

          # smplx_output = out["smplx_output"]
          mesh_cam = out["smplx_mesh_cam"]
          mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

          # generate confidence based on visibility
          # points_visibility = check_visibility_pt3d(rasterizer, vis_img, mesh_cam, faces_tensor, {'focal': focal, 'princpt': princpt})
          # new_joints_img = demoer.model.module.get_joints_visibility(smplx_output, faces_tensor, points_visibility)
          new_joints_img = demoer.model.module.not_get_joints_visibility(smplx_output)
          new_joints_img[:, 0] = new_joints_img[:, 0] * bbox[2] / cfg.model.output_hm_shape[2] + bbox[0]
          new_joints_img[:, 1] = new_joints_img[:, 1] * bbox[3] / cfg.model.output_hm_shape[1] + bbox[1]

          out_frame_dict["kpt2d"] = new_joints_img
          # out_frame_dict["betas"] = smplx_output.betas[0].cpu().detach().float().numpy()
          # out_frame_dict["expression"] = smplx_output.expression[0].cpu().detach().float().numpy()

          # out_frame_dict["full_pose"] = smplx_output.full_pose[0].cpu().detach().float().numpy()
          # out_frame_dict["transl"] = smplx_output.transl[0].cpu().detach().float().numpy()
          out_frame_dict["betas"] = smplx_output['betas'][0].cpu().detach().float().numpy()
          out_frame_dict["expression"] = smplx_output['expression'][0].cpu().detach().float().numpy()

          out_frame_dict["full_pose"] = smplx_output['full_pose'][0].cpu().detach().float().numpy()
          out_frame_dict["transl"] = smplx_output['transl'][0].cpu().detach().float().numpy()


          out_results.append(out_frame_dict)

          # vis_img = render_mesh_pt3d(vis_img, mesh_cam, faces_tensor, {'focal': focal, 'princpt': princpt}, rasterizer)
          # writer.append_data(vis_img.astype(np.uint8))


        cap.release()
        # writer.close()

        np.save(out_npy_path, out_results)

if __name__ == '__main__':
  main()

