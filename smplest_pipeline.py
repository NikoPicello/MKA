import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import cv2 as cv
import datetime
import glob
from tqdm import trange
from pathlib import Path
from human_models.human_models import SMPLX
from ultralytics import YOLO
from utils.base import Tester
from utils.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
from utils.visualization_utils import render_mesh, get_rasterizer, check_visibility_pt3d_cached, save_obj
from utils.inference_utils import non_max_suppression
from utils.transforms import world2cam, cam2pixel, rigid_align
from time import time
from pathlib import Path
import imageio

cudnn.benchmark = False
cudnn.deterministic = True

undistort = False
vis = True

joint_set = {
  'joint_num': 17,
  'joints_name': ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Head', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist'),
  'flip_pairs': ( (1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13) ),
  'eval_joint': (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16),
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

SPATIAL_REGIONS = {
  'GF' : {0 : [0., 0.5, 0., 1.], 1 : [0.5, 1., 0., 1.]},
  'GB' : {1 : [0., 0.5, 0., 1.], 0 : [0.5, 1., 0., 1.]},
  'FC1' : {0 : [0.25, 0.75, 0., 1.]},
  'FC2' : {1 : [0.25, 0.75, 0., 1.]},
  'HA1' : {0 : [0.25, 0.75, 0., 1.]},
  'HA2' : {1 : [0.25, 0.75, 0., 1.]}
}

activities = ['animals', 'gaze', 'ghost', 'lego', 'talk']

def render_mesh_simple(img, mesh_vertices, mesh_faces, camera_dict, person_id, face_transform=None):
  mesh_vertices = np.asarray(mesh_vertices, dtype=np.float32)
  mesh_faces = np.asarray(mesh_faces, dtype=np.int32)
  focal = np.asarray(camera_dict['focal'], dtype=np.float32)
  princpt = np.asarray(camera_dict['princpt'], dtype=np.float32)
  K = np.array([[focal[0], 0, princpt[0]],
                [0, focal[1], princpt[1]],
                [0, 0, 1]], dtype=np.float32)
  proj_homo = mesh_vertices @ K.T
  proj = proj_homo[:, :2] / proj_homo[:, 2:3]
  if face_transform is not None:
    pts_hom = np.concatenate([proj, np.ones((len(proj), 1))], axis=1).astype(np.float32)
    proj = (face_transform @ pts_hom.T).T
  img_out = (img * 255).astype(np.uint8).copy() if img.max() <= 1 else img.copy().astype(np.uint8)
  color = (0, 0, 255) if person_id == 1 else (0, 255, 0)
  for face in mesh_faces:
    pts = proj[face].astype(np.int32)
    for i in range(3):
      p1 = tuple(pts[i])
      p2 = tuple(pts[(i+1) % 3])
      if 0 <= p1[0] < img_out.shape[1] and 0 <= p1[1] < img_out.shape[0]:
        if 0 <= p2[0] < img_out.shape[1] and 0 <= p2[1] < img_out.shape[0]:
          cv.line(img_out, p1, p2, color, 1)
  return img_out.astype(np.float32) / 255.0

def xyxy_to_normalized(bbox_xyxy, img_width, img_height):
  x1, y1, x2, y2 = bbox_xyxy
  return np.array([x1 / img_width, x2 / img_width, y1 / img_height, y2 / img_height])

def xyxy_to_xywh(bbox_xyxy):
  x1, y1, x2, y2 = bbox_xyxy
  return np.array([x1, y1, x2 - x1, y2 - y1])

def bbox_centroid(bbox_xyxy):
  x1, y1, x2, y2 = bbox_xyxy
  return (x1 + x2) / 2, (y1 + y2) / 2

def back_project_keypoints(keypoints, inv_trans):
  kps_xy = keypoints[:, :2]
  ones = np.ones((kps_xy.shape[0], 1), dtype=np.float32)
  kps_hom = np.concatenate([kps_xy, ones], axis=1)
  kps_orig = kps_hom @ inv_trans.T
  if keypoints.shape[1] == 3:
    return np.concatenate([kps_orig, keypoints[:, 2:3]], axis=1)
  return kps_orig

def assign_bbox_to_person(bbox_normalized, cam_id):
  if cam_id not in SPATIAL_REGIONS:
    print(f"Unknown camera: {cam_id}")
    return None
  x_center = (bbox_normalized[0] + bbox_normalized[1]) / 2
  y_center = (bbox_normalized[2] + bbox_normalized[3]) / 2
  for person_id, region in SPATIAL_REGIONS[cam_id].items():
    x_min, x_max, y_min, y_max = region
    if x_min <= x_center <= x_max and y_min <= y_center <= y_max:
      return person_id
  return None

def filter_and_assign_bboxes(boxes_xyxy, confidences, img_width, img_height,
                             cam_id, conf_threshold=0.5):
  assignments = {}
  valid_idx = np.where(confidences >= conf_threshold)[0]
  if len(valid_idx) == 0:
    return assignments
  valid_boxes = boxes_xyxy[valid_idx]
  valid_confs = confidences[valid_idx]
  for bbox_xyxy, conf in zip(valid_boxes, valid_confs):
    bbox_norm = xyxy_to_normalized(bbox_xyxy, img_width, img_height)
    person_id = assign_bbox_to_person(bbox_norm, cam_id)
    if person_id is not None:
      if person_id not in assignments or conf > assignments[person_id][4]:
        assignments[person_id] = np.append(bbox_xyxy, conf)
  return assignments

def select_free_gpu():
  if not torch.cuda.is_available():
    return torch.device('cpu')
  free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
  return torch.device(f'cuda:{free.index(max(free))}')

def main():
  device = select_free_gpu()

  main_path = '/'.join(sys.path[0].split('/')[:-2]) + '/'
  resources_path = os.path.join(main_path, 'resources')
  calibs_path   = os.path.join(resources_path, 'calibs')
  sessions_path = os.path.join(resources_path, 'sessions')
  out_path      = os.path.join(resources_path, 'smplest_results')
  sid_paths = sorted(glob.glob(sessions_path + '/*'))

  cudnn.benchmark = True
  ckpt_name = "smplest_x_h"

  config_path = osp.join('./configs', f'config_{ckpt_name}.py')
  cfg = Config.load_config(config_path)
  checkpoint_path = osp.join('./pretrained_models', ckpt_name, f'{ckpt_name}.pth.tar')

  time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  exp_name = f'inference_{ckpt_name}_{time_str}'

  new_config = {
    "model": {
      "pretrained_model_path": checkpoint_path,
    },
    "log": {
      'exp_name':  exp_name,
      'log_dir': osp.join(out_path, 'outputs', exp_name, 'log'),
    }
  }
  cfg.update_config(new_config)
  cfg.prepare_log()

  torch.cuda.set_device(device)

  smpl_x = SMPLX(cfg.model.human_model_path)
  faces_tensor = torch.from_numpy(smpl_x.face.astype(np.int32)).unsqueeze(0).to(device)
  np.save(os.path.join(out_path, 'smplx_faces.npy'), smpl_x.face)

  demoer = Tester(cfg)
  demoer.logger.info(f"Using device {device}.")
  demoer.logger.info(f'Inference with [{cfg.model.pretrained_model_path}].')
  demoer._make_model()
  demoer.model.eval()

  bbox_model = getattr(cfg.inference.detection, "model_path", './pretrained_models/yolov8x.pt')
  detector = YOLO(bbox_model)

  visibility_cache = {}
  for sid_path in sid_paths:
    session_id = Path(sid_path).stem

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
      cam_dict[cam_map[cam_name]] = {'K': K, 'D': D, 'R': R, 'T': T}

    for activity in activities:
      print(f'Processing {activity} in session {session_id}')
      vid_paths = glob.glob(os.path.join(sid_path, activity) + '/*')
      vid_paths = [v for v in vid_paths if not ('E1.mp4' in v or 'E2.mp4' in v)]
      for vid_path in vid_paths:
        video_name = Path(vid_path).stem

        visibility_cache.clear()

        K = cam_dict[video_name]['K']
        D = cam_dict[video_name]['D']

        cap = cv.VideoCapture(vid_path)
        fps = int(cap.get(cv.CAP_PROP_FPS))
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        frame_width  = 1280
        frame_height = 720

        curr_out_path = os.path.join(out_path, f"{session_id}/{activity}")
        save_dir_smplx = os.path.join(curr_out_path, f"{video_name}_smplx")
        os.makedirs(curr_out_path, exist_ok=True)
        os.makedirs(save_dir_smplx, exist_ok=True)

        out_npy_path = os.path.join(curr_out_path, f"{video_name}_smplx.npy")

        if vis:
          if undistort:
            out_vid_path = os.path.join(curr_out_path, f"{video_name}_render_und.mp4")
          else:
            out_vid_path = os.path.join(curr_out_path, f"{video_name}_render.mp4")
          if vis:
            rasterizer = get_rasterizer(frame_height, frame_width)
          writer = imageio.get_writer(
              out_vid_path,
              fps=fps, mode='I', format='FFMPEG', macro_block_size=1
          )

        new_K, _ = cv.getOptimalNewCameraMatrix(K, D, (1280, 720), 1)

        out_results = []
        for fidx in trange(total_frames):
          ret, frame = cap.read()
          if not ret: break
          frame = cv.resize(frame, (1280, 720))
          if undistort:
            frame = cv.undistort(frame, K, D, None, new_K)

          out_frame_dict = {"fidx": fidx}
          transform = transforms.ToTensor()
          img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
          original_img = img_rgb.copy().astype(np.float32)
          vis_img = original_img.copy()
          original_img_height, original_img_width = original_img.shape[:2]

          results = detector.predict(original_img,
                                     device=str(device),
                                     classes=00,
                                     conf=cfg.inference.detection.conf,
                                     save=cfg.inference.detection.save,
                                     verbose=cfg.inference.detection.verbose)
          boxes_xyxy  = results[0].boxes.xyxy.detach().cpu().numpy()
          confidences = results[0].boxes.conf.detach().cpu().numpy()

          if len(results) < 1:
            if vis:
              writer.append_data(vis_img.astype(np.uint8))
            out_results.append(out_frame_dict)
            print("=== failed", fidx, flush=True)
            continue

          assignments = filter_and_assign_bboxes(
              boxes_xyxy, confidences, frame_width, frame_height, video_name, conf_threshold=0.5
          )

          for person_id, bbox_with_conf in assignments.items():
            curr_frame_dict = {}
            bbox_xyxy = bbox_with_conf[:4]
            bbox_xywh = xyxy_to_xywh(bbox_xyxy)

            bbox = process_bbox(bbox=bbox_xywh,
                                img_width=frame_width,
                                img_height=frame_height,
                                input_img_shape=cfg.model.input_img_shape,
                                ratio=getattr(cfg.data, "bbox_ratio", 1.25))

            focal  = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2],
                      cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
            princpt = [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0],
                       cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]

            img, trans, inv_trans = generate_patch_image(cvimg=original_img,
                                                         bbox=bbox,
                                                         scale=1.0,
                                                         rot=0.0,
                                                         do_flip=False,
                                                         out_shape=cfg.model.input_img_shape)
            img = transform(img.astype(np.float32)) / 255
            img = img.to(device)[None, :, :, :]
            inputs   = {'img': img}
            targets  = {}
            meta_info = {}

            with torch.no_grad():
              out, smplx_output = demoer.model(inputs, targets, meta_info, 'test')

            mesh = smplx_output['vertices'][0].detach().cpu().numpy()
            save_obj(mesh, smpl_x.face, f"{save_dir_smplx}/f{fidx}_p{person_id}.obj")

            cam_param_dict = {'focal': focal, 'princpt': princpt}
            pelvis_position = smplx_output['joints'][0, 55, :].cpu().numpy()
            verts = smplx_output['vertices']
            points_visibility = check_visibility_pt3d_cached(
                rasterizer, img_rgb, verts, faces_tensor,
                cam_param_dict, visibility_cache,
                video_name, fidx, person_id,
                pelvis_position, motion_threshold=0.05
            )
            new_joints_img = demoer.model.module.get_joints_visibility(smplx_output, faces_tensor, points_visibility)
            new_joints_img[:, 0] = new_joints_img[:, 0] * bbox[2] / cfg.model.output_hm_shape[2] + bbox[0]
            new_joints_img[:, 1] = new_joints_img[:, 1] * bbox[3] / cfg.model.output_hm_shape[1] + bbox[1]

            curr_frame_dict['joint_cam']       = smplx_output['joints'][0].detach().cpu().numpy()
            curr_frame_dict['kpt2d']           = new_joints_img
            curr_frame_dict['global_orient']   = smplx_output['global_orient'][0].reshape(-1, 3).detach().cpu().numpy()
            curr_frame_dict['body_pose']       = smplx_output['body_pose'][0].reshape(-1, 3).detach().cpu().numpy()
            curr_frame_dict['left_hand_pose']  = smplx_output['left_hand_pose'][0].reshape(-1, 3).detach().cpu().numpy()
            curr_frame_dict['right_hand_pose'] = smplx_output['right_hand_pose'][0].reshape(-1, 3).detach().cpu().numpy()
            curr_frame_dict['jaw_pose']        = smplx_output['jaw_pose'][0].reshape(-1, 3).detach().cpu().numpy()
            curr_frame_dict['leye_pose']       = np.zeros((1, 3))
            curr_frame_dict['reye_pose']       = np.zeros((1, 3))
            curr_frame_dict['betas']           = smplx_output['betas'][0].reshape(-1, 10).detach().cpu().numpy()
            curr_frame_dict['expression']      = smplx_output['expression'][0].reshape(-1, 10).detach().cpu().numpy()
            curr_frame_dict['transl']          = smplx_output['transl'][0].reshape(-1, 3).detach().cpu().numpy()
            curr_frame_dict['vertices']        = mesh
            curr_frame_dict['focal']           = np.array(focal,   dtype=np.float32)
            curr_frame_dict['princpt']         = np.array(princpt, dtype=np.float32)
            out_frame_dict[person_id] = curr_frame_dict

          if vis:
            for person_id in assignments:
              pdata = out_frame_dict.get(person_id, {})
              if 'vertices' not in pdata:
                continue
              vis_img = render_mesh_simple(vis_img, pdata['vertices'], smpl_x.face,
                                           {'focal': pdata['focal'], 'princpt': pdata['princpt']},
                                           person_id)
              vis_img_u8 = (vis_img * 255).astype(np.uint8) if vis_img.max() <= 1 else vis_img.astype(np.uint8)
              vis_img = vis_img_u8.astype(np.float32) / 255.0

          out_results.append(out_frame_dict)
          if vis:
            writer.append_data((vis_img * 255).astype(np.uint8))

        cap.release()
        if vis:
          writer.close()

        np.save(out_npy_path, out_results)

if __name__ == '__main__':
  main()
