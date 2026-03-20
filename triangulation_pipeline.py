import argparse
import json
import os
from tqdm import trange, tqdm
from pathlib import Path
import imageio
import cv2 as cv
import glob

from typing import List
import numpy as np
from scipy.spatial.transform import Rotation as scipy_Rotation
from mmhuman3d.core.cameras.camera_parameters import CameraParameter as CameraParameter_mm
from mmhuman3d.core.conventions.keypoints_mapping import KEYPOINTS_FACTORY

import sys

from aniposelib.cameras import interpolate_data










import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_3d_keypoints(kpts, extrinsic_mat):
    """
    Visualize 3D keypoints in both world and camera coordinate frames
    """
    fig = plt.figure(figsize=(15, 5))
    
    # === 1. World coordinates (original) ===
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(kpts[:, 0], kpts[:, 1], kpts[:, 2], c='blue', marker='o', s=50)
    
    ax1.set_xlabel('X (world)')
    ax1.set_ylabel('Y (world)')
    ax1.set_zlabel('Z (world)')
    ax1.set_title('World Coordinates')
    set_axes_equal(ax1)
    
    # === 2. Camera coordinates (after extrinsic transform) ===
    ax2 = fig.add_subplot(132, projection='3d')
    
    # Transform to camera space
    joint_num = kpts.shape[0]
    ones = np.ones((joint_num, 1))
    kpt3d_homo = np.concatenate([kpts, ones], axis=1)
    kpt3d_cam = (extrinsic_mat @ kpt3d_homo.T).T[:, :3]
    
    ax2.scatter(kpt3d_cam[:, 0], kpt3d_cam[:, 1], kpt3d_cam[:, 2], 
                c='red', marker='o', s=5)
    
    # Draw camera coordinate axes
    axis_length = np.max(np.abs(kpt3d_cam)) * 0.5
    ax2.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.1, label='X')
    ax2.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.1, label='Y')
    ax2.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1, label='Z')
    
    ax2.set_xlabel('X (cam)')
    ax2.set_ylabel('Y (cam)')
    ax2.set_zlabel('Z (cam)')
    ax2.set_title('Camera Coordinates')
    ax2.legend()
    set_axes_equal(ax2)
    
    # === 3. Top-down view ===
    ax3 = fig.add_subplot(133)
    ax3.scatter(kpt3d_cam[:, 0], kpt3d_cam[:, 2], c='red', marker='o', s=5)
    ax3.scatter(0, 0, c='black', marker='^', s=5, label='Camera')  # Camera at origin
    ax3.set_xlabel('X (cam)')
    ax3.set_ylabel('Z (cam, depth)')
    ax3.set_title('Top-Down View (Camera Looking +Z)')
    ax3.grid(True)
    ax3.legend()
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Print statistics
    print(f"\n=== 3D Keypoints Statistics ===")
    print(f"World coords - X range: [{kpts[:, 0].min():.2f}, {kpts[:, 0].max():.2f}]")
    print(f"World coords - Y range: [{kpts[:, 1].min():.2f}, {kpts[:, 1].max():.2f}]")
    print(f"World coords - Z range: [{kpts[:, 2].min():.2f}, {kpts[:, 2].max():.2f}]")
    print(f"\nCamera coords - X range: [{kpt3d_cam[:, 0].min():.2f}, {kpt3d_cam[:, 0].max():.2f}]")
    print(f"Camera coords - Y range: [{kpt3d_cam[:, 1].min():.2f}, {kpt3d_cam[:, 1].max():.2f}]")
    print(f"Camera coords - Z range: [{kpt3d_cam[:, 2].min():.2f}, {kpt3d_cam[:, 2].max():.2f}]")
    print(f"Points behind camera (Z < 0): {np.sum(kpt3d_cam[:, 2] < 0)}")
    
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'./3d_keypoints_visualization.png', dpi=150, bbox_inches='tight')
    
    return kpt3d_cam

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale"""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])
















def printv(sentence, verbose=False):
  if verbose:
    print(sentence)


class CameraParameter(CameraParameter_mm):

  AUGMENTED_SUPPORTED_KEYS = {
    'floor_normal': {
      'type': list,
      'len': 3,
    },
    'floor_center': {
      'type': list,
      'len': 3,
    }
  }

  SUPPORTED_KEYS = dict(CameraParameter_mm.SUPPORTED_KEYS,
                        **AUGMENTED_SUPPORTED_KEYS)

  def __init__(self,
               name: str = 'default',
               H: int = 1080,
               W: int = 1920) -> None:
    """
    Args:
        name (str, optional):
            Name of this camera. Defaults to "default".
        H (int, optional):
            Height of a frame, in pixel. Defaults to 1080.
        W (int, optional):
            Width of a frame, in pixel. Defaults to 1920.
    """
    super().__init__(name=name, H=H, W=W)

  def get_KRT(self,
              k_dim: int = 3,
              inverse_extrinsic: bool = False) -> List[np.ndarray]:
    """Get intrinsic and extrinsic of a camera.

    Args:
        k_dim (int, optional):
            Dimension of the returned mat K.
            Defaults to 3.
        inverse_extrinsic (bool, optional):
            If true, R_mat and T_vec transform a point
            from view to world. Defaults to False.

    Raises:
        ValueError: k_dim is neither 3 nor 4.

    Returns:
        List[np.ndarray]:
            K_mat (np.ndarray):
                In shape [3, 3].
            R_mat (np.ndarray):
                Rotation from world to view in default.
                In shape [3, 3].
            T_vec (np.ndarray):
                Translation from world to view in default.
                In shape [3,].
    """
    K_3x3 = self.get_mat_np('in_mat')
    R_mat = self.get_mat_np('rotation_mat')
    T_vec = np.asarray(self.get_value('translation'))
    if inverse_extrinsic:
      R_mat = np.linalg.inv(R_mat).reshape(3, 3)
      T_vec = -np.dot(R_mat, T_vec)
    if k_dim == 3:
      return [K_3x3, R_mat, T_vec]
    elif k_dim == 4:
      K_3x3 = np.expand_dims(K_3x3, 0)  # shape (1, 3, 3)
      K_4x4 = convert_K_3x3_to_4x4(K=K_3x3, is_perspective=True)  # shape (1, 4, 4)
      K_4x4 = K_4x4[0, :, :]
      return [K_4x4, R_mat, T_vec]
    else:
      raise ValueError(f'K mat cannot be converted to {k_dim}x{k_dim}')

  def inverse_extrinsics(self):
    """Inverse camera extrinsics.

    Call it when you get wrong results with current parameters.
    """
    r_mat_np = self.get_mat_np('rotation_mat')
    r_mat_inv_np = np.linalg.inv(r_mat_np).reshape(3, 3)
    t_vec_list = self.get_value('translation')
    t_vec_np = np.asarray(t_vec_list).reshape(3, 1)
    t_vec_inv_np = -np.dot(r_mat_inv_np, t_vec_np)
    self.set_mat_np('rotation_mat', r_mat_inv_np)
    self.set_value('translation', t_vec_inv_np.tolist())

  def load_camera_gt(self, cam_dict, dirname="") -> None:
    gt_R = np.array(cam_dict["R"], dtype=np.float32)
    gt_t = np.array(cam_dict["T"], dtype=np.float32)
    intrinsic_mat = np.array(cam_dict["K"], dtype=np.float32)
    print(cam_dict['D'])
    dist_coef = np.array(cam_dict["D"][0], dtype=np.float32)

    if "Fit3D" in dirname:
        gt_t = -np.matmul(gt_t.reshape(1, 3), np.transpose(gt_R))[0]

    self.set_mat_np('in_mat', intrinsic_mat)
    self.set_value('k1', dist_coef[0])
    self.set_value('k2', dist_coef[1])
    self.set_value('k3', dist_coef[2])
    self.set_value('p1', dist_coef[3])
    self.set_value('p2', dist_coef[4])
    self.set_mat_np('rotation_mat',gt_R)
    self.set_value('translation', gt_t.tolist())

  def get_aist_dict(self) -> dict:
    """Get a dict of camera parameters, which contains all necessary args
    for aniposelib.cameras.Camera(). Use
    aniposelib.cameras.Camera(**return_dict) to construct a camera.

    Returns:
        dict:
            A dict of camera parameters: name, dist, size, matrix, etc.
    """
    ret_dict = {}
    ret_dict['name'] = self.name
    ret_dict['dist'] = [
      self.parameters_dict['k1'],
      self.parameters_dict['k2'],
      self.parameters_dict['p1'],
      self.parameters_dict['p2'],
      self.parameters_dict['k3'],
    ]
    ret_dict['size'] = (self.parameters_dict['H'],
                        self.parameters_dict['W'])
    ret_dict['matrix'] = np.array(self.parameters_dict['in_mat'])
    rotation_mat = np.array(self.parameters_dict['rotation_mat'])
    # convert rotation as axis angle(rotation vector)
    rotation_vec = scipy_Rotation.from_matrix(rotation_mat).as_rotvec()
    ret_dict['rvec'] = rotation_vec
    ret_dict['tvec'] = self.parameters_dict['translation']
    return ret_dict

  def setup_transform(self):
    """Setup transform between camera0 and self."""
    # rotation matrix from camera0 coordinates to self coordinates
    rot_mat = np.asarray(self.get_value('rotation_mat')).reshape(3, 3)
    self.rotation = scipy_Rotation.from_matrix(rot_mat)
    self.translation = np.asarray(self.get_value('translation'))
    # from self to camera0
    self.inv_rotation = self.rotation.inv()
    self.transform_ready = True

  def transform_points_cam_to_self(self, points3d: np.ndarray) -> np.ndarray:
    """Transform an array of 3d points in camera0 coordinates to self
    coordinates.

    Args:
        points3d (np.ndarray):
            An array of 3d points, in shape [point_number, 3]
            or [point_number, 4] with confidence.

    Returns:
        np.ndarray:
            An array of transformed 3d points, in the same
            shape of points3d. Only data points[:, :3] is
            different.
    """
    assert self.transform_ready is True, \
      'Transform not ready, call self.setup_transform() first.'
    assert points3d.ndim == 2 and \
      (points3d.shape[1] == 3 or points3d.shape[1] == 4), \
      'Input.shape has to be [point_number, 3] or [point_number, 4].'
    self_points3d = \
      self.rotation.apply(points3d[:, :3])
    translation_np = \
      self.translation[np.newaxis, :].repeat(
          self_points3d.shape[0], axis=0)
    self_points3d += translation_np

    output_points3d = points3d.copy()
    output_points3d[:, :3] = self_points3d
    return output_points3d

  def to_string(self) -> str:
    """Convert self.to_dict() to a string.

    Returns:
        str:
            A dict in json string format.
    """
    dump_dict = self.to_dict()
    ret_str = json.dumps(dump_dict, default=convert_np)
    return ret_str

def save_verts(verts, file_name='a.obj'):
  with open(file_name, 'w') as fp:
    for v in verts:
      fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

def options():
  parser = argparse.ArgumentParser()
  parser.add_argument("--video_dir", type=str, required=True)
  parser.add_argument("--kpt2d_dir", type=str, required=True)
  parser.add_argument("--out_dir", type=str, required=True)
  args = parser.parse_args()
  return args

cam_map = {
  'GC' : 'GB',
  'HC' : 'GF',
  'Z1' : 'FC1',
  'Z2' : 'FC2',
  'N1' : 'HA1',
  'N2' : 'HA2'
}

person_cams = {
    0 : ['GF', 'GB', 'FC1', 'HA1'],
    1 : ['GF', 'GB', 'FC2', 'HA2']
}

activities = ['animals', 'gaze', 'ghost', 'lego', 'talk']
def main():
  verbose = True
  main_path = '/'.join(sys.path[0].split('/')[:-2]) + '/'
  resources_path = os.path.join(main_path, 'resources')
  calibs_path   = os.path.join(resources_path, 'calibs')
  sessions_path = os.path.join(resources_path, 'sessions')
  out_path = os.path.join(resources_path, 'triangulation_results')
  kpt2d_dir = os.path.join(resources_path, 'smplestx_results')
  sid_paths = sorted(glob.glob(sessions_path + '/*'))
  sys.path.append("./")
  sys.path.append(os.path.join("./dependencies"))

  from mocap.multi_view_3d_keypoint.triangulate_scene import TriangulateScene
  from zoehuman.utils.keypoint_utils import search_limbs

  from zoehuman.core.visualization.visualize_keypoints3d import visualize_kp3d
  from zoehuman.data.data_structures import SMCReader
  from zoehuman.data.data_structures.human_data import HumanData
  from zoehuman.utils.path_utils import (  # prevent yapf isort conflict
      Existence, check_path_existence, check_path_suffix,
  )

  src = "smplx"
  dst = "human_data"
  src_names = KEYPOINTS_FACTORY[src.lower()]
  dst_names = KEYPOINTS_FACTORY[dst.lower()]
  kpt2d_mask = np.ones(shape=[len(src_names)], dtype=np.uint8)
  kpts_num = len(src_names)
  invalid_kpt2d = np.zeros((kpts_num, 3), dtype=np.float32)



  for sid_path in sid_paths:
    session_id = Path(sid_path).stem
    with open(os.path.join(sid_path, 'session_data.txt')) as f:
      lines = f.readlines()
      calib_date = lines[1][11:].strip()
    curr_calib_path = os.path.join(calibs_path, calib_date)
    cam_calibs = glob.glob(curr_calib_path + '/*')
    cam_para_list = {}
    for cam_calib in cam_calibs:
      cam_name = Path(cam_calib).stem
      fs = cv.FileStorage(os.path.join(calibs_path, f"{calib_date}/{cam_name}.yml"), cv.FILE_STORAGE_READ)
      K = fs.getNode('K').mat()
      D = fs.getNode('D').mat()
      R = fs.getNode('R').mat()
      T = fs.getNode('T').mat()
      fs.release()
      cam_parameter = CameraParameter(name=cam_map[cam_name], H=1080, W=1920) # TODO: check this
      cam_parameter.load_camera_gt({'K' : K, 'D' : D, 'R' : R, 'T' : T})
      cam_para_list[cam_map[cam_name]] = cam_parameter

    for activity in activities:
      print(f"Processing {activity} in {session_id}")
      vid_paths = glob.glob(os.path.join(sid_path, activity) + '/*')
      vid_paths = [v for v in vid_paths if not ('E1.mp4' in v or 'E2.mp4' in v)]
      vid_paths_dict = {}
      kpt2d_path_arr = {}
      img_width = None
      img_height = None
      total_frames = None
      fps = None
      for vid_path in vid_paths:
        if img_width is None or img_height is None:
          cap = cv.VideoCapture(vid_path)
          img_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
          img_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
          print(img_width)
          print(img_height)
          total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
          fps = int(cap.get(cv.CAP_PROP_FPS))
          cap.release()
        video_name = Path(vid_path).stem
        kpt2d_path = os.path.join(kpt2d_dir,  f"{session_id}/{activity}/{video_name}_res.npy")
        kpt2d_path_arr[video_name] = kpt2d_path
        vid_paths_dict[video_name] = vid_path

      total_frames = 4

      curr_out_path = os.path.join(out_path, f"{session_id}/{activity}/")
      os.makedirs(curr_out_path, exist_ok=True)
      out_wo_opt_npy_path = os.path.join(curr_out_path, "no_optim_kpt3d.npz")
      out_w_opt_npy_path = os.path.join(curr_out_path, "optim_kpt3d.npz")


      # TODO: CHANGE INTO LOOPING FIRST ON EVERY FRAME AND TRINAUGLATE THE PEOPLE AT EACH TIME STAMP, INSTEAD OF VICE-VERSA
      result_dict_all = {}
      for p in [0]: #, 1]: 
        printv(f"Triangulation on person {p+1}/2", verbose=verbose)
        kpt2d_all_dict  = {}
        not_valid_arr   = {}
        human_data_dict = {}
        for cam_id in person_cams[p]:
          not_valid_arr[cam_id] = []
          kpt2d_list = []
          result_dict = {}
          
          kpt2d_arr = np.load(kpt2d_path_arr[cam_id], allow_pickle=True)
          frame_len = len(kpt2d_arr)
          print("=======", cam_id, frame_len, flush=True)

          prev_kpt2d = None
          for fi in range(total_frames):
            if 'kpt2d' in kpt2d_arr[fi][p]:
              kpt2d = np.array(kpt2d_arr[fi][p]['kpt2d'], dtype=np.float32)[:kpts_num]
              for _i in range(kpts_num):
                kpt = kpt2d[_i]
                x = int(kpt[0])
                y = int(kpt[1])
                if x < 0 or x >= img_width or y < 0 or y >= img_height:
                  kpt2d[_i][2] = 0.0

              prev_kpt2d = kpt2d
              kpt2d_list.append(kpt2d)
            elif prev_kpt2d is not None:
              print(f"=== fill kpt2d {cam_id}_{fi:04d} with previous", flush=True)
              kpt2d_list.append(prev_kpt2d)
            else:
              print(f"=== failed to load kpt2d in {cam_id}_{fi:04d}", flush=True)
              not_valid_arr[cam_id].append(fi)
              kpt2d_list.append(invalid_kpt2d)

          kpt2d_list = np.stack(kpt2d_list, axis=0)
          kpt2d_all_dict[cam_id] = kpt2d_list

        invalid_idx_arr = []
        for cam_id in person_cams[p]:
          invalid_idx_arr += not_valid_arr[cam_id]
        invalid_idx_arr = np.unique(invalid_idx_arr).astype(int)
        unique_mask = np.ones(total_frames, dtype=bool)
        unique_mask[invalid_idx_arr] = False

        for cam_id in person_cams[p]:
          curr_kpt2d = kpt2d_all_dict[cam_id].copy()
          curr_kpt2d = curr_kpt2d[unique_mask]
          
          print(f"cam {cam_id} after filter", curr_kpt2d.shape, flush=True)
          printv(f"filtered kpts 2d has {len(curr_kpt2d)} elements", verbose=verbose)
          
          human_data_dict[cam_id] = {
            'keypoints2d': curr_kpt2d,
            'keypoints2d_mask': kpt2d_mask,
            'keypoints2d_convention': 'smplx'
          }
        
        curr_cam_para = [cam_para_list[cam_id] for cam_id in person_cams[p]]
        scene = TriangulateScene(curr_cam_para, 0.1)
        result_dict = {'optim': {}, 'no_optim': {}, 'invalid_idx': invalid_idx_arr}

        keypoints3d_no_optim = []
        keypoints3d_optim = []
        frame_num = human_data_dict[cam_id]["keypoints2d"].shape[0]
        frame_num = 4
        interval = 2
        printv(f"frame num is {frame_num}", verbose=verbose)
        printv("Applying triangulation", verbose=verbose)
          
        for _fi in trange(0, frame_num, interval):
          human_data = []
          start_fi = _fi
          end_fi = min(_fi + interval, frame_num)
          for cam_id in person_cams[p]:
            human_data.append({
              'keypoints2d': human_data_dict[cam_id]["keypoints2d"][start_fi:end_fi],
              'keypoints2d_mask': kpt2d_mask,
              'keypoints2d_convention': 'smplx'
            })
          
          kpt3d_no_opt = scene.triangulate(human_data)
          kpt3d_opt = scene.optim(human_data, keypoints3d=kpt3d_no_opt, constraints=None)
          keypoints3d_no_optim.append(kpt3d_no_opt)
          keypoints3d_optim.append(kpt3d_opt)
        
        result_dict['no_optim']['keypoints3d'] = np.concatenate(keypoints3d_no_optim, axis=0)
        result_dict['optim']['keypoints3d'] = np.concatenate(keypoints3d_optim, axis=0)
        printv(f"keypoints3d non optimal shape is : {result_dict['no_optim']['keypoints3d'].shape}", verbose=verbose)
        printv(f"keypoints3d optimal shape is : {result_dict['optim']['keypoints3d'].shape}",  verbose=verbose)

        for key in ['optim', 'no_optim']:
          if result_dict[key] is not None:
            kpt3d = result_dict[key]['keypoints3d']
            human_data_3d = TriangulateScene.convert_result_to_human_data(
              kpt3d, kpt2d_mask
            )
            human_data_3d['not_valid'] = not_valid_arr # i changed the positino of this : it was after the next line
            result_dict[key]['human_data'] = human_data_3d
            if 'no_' in key: out_npy_path = out_wo_opt_npy_path
            else: out_npy_path = out_w_opt_npy_path
            # human_data_3d.dump(out_npy_path)
        
        result_dict_all[p] = result_dict
      
      # print(result_dict_all[0]['optim']['keypoints3d'].shape)
      # print(result_dict_all[0]['no_optim']['keypoints3d'].shape)
      # print(result_dict_all[1]['optim']['keypoints3d'].shape)
      # print(result_dict_all[1]['no_optim']['keypoints3d'].shape)

      # print(result_dict_all[1])
      out_video_path = os.path.join(curr_out_path, "optim_kpt3d_render.mp4")
      print("=== visualization", out_video_path, flush=True)
      writer = imageio.get_writer(
        out_video_path,
        fps=fps, mode='I', 
        format='FFMPEG', 
        macro_block_size=1
      )

      ref_cam = 'FC2'
      cap = cv.VideoCapture(vid_paths_dict[ref_cam])
      cam_param = cam_para_list[ref_cam]
      intrinsic_mat = cam_param.get_mat_np('in_mat')
      dist_coef = np.array(cam_param.get_aist_dict()['dist'])
      R_mat = cam_param.get_mat_np('rotation_mat')
      t_vec = np.array(cam_param.get_value('translation'))
      extrinsic_mat = np.eye(4, dtype=np.float32)
      extrinsic_mat[:3, :3] = R_mat.copy()
      if len(t_vec.shape) == 1:
        extrinsic_mat[:3, 3] = t_vec.copy()
      else:
        extrinsic_mat[:3, 3] = t_vec.squeeze()

      for fi in trange(total_frames):
        ret, frame = cap.read()
        if not ret:
          break
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        for p in [0]: #, 1]:
          if p not in result_dict_all: continue

          result_dict = result_dict_all[p]
          kpt3d = result_dict['optim']['keypoints3d'][fi, :]
          # kpt3d = result_dict['optim']['keypoints3d'][fi, :3]

          kpt3d_cam = visualize_3d_keypoints(kpt3d, extrinsic_mat)

          invalid_idx_arr = result_dict['invalid_idx']

          if fi in invalid_idx_arr:
            print("=== invalid", fi, flush=True)
            continue

          joint_num = kpt3d.shape[1]
          ones = np.ones((joint_num, 1))

          rvec, _ = cv.Rodrigues(extrinsic_mat[:3, :3])  # Convert rotation matrix to rotation vector
          tvec = extrinsic_mat[:3, 3]  # Translation vector

          # Project 3D points to 2D
          kpt2d_img, _ = cv.projectPoints(kpt3d, rvec, tvec, intrinsic_mat, dist_coef)
          kpt2d_img = kpt2d_img.reshape(-1, 2)  # Reshape from (N, 1, 2) to (N, 2)

          # kpt3d_homo = np.concatenate([kpt3d[fi], ones], axis=1)
          # kpt3d_cam = np.matmul(extrinsic_mat, kpt3d_homo.transpose()).transpose()[:, :3]
          # kpt3d_cam_norm = kpt3d_cam / kpt3d_cam[:, 2:]
          # kpt2d_img = np.matmul(intrinsic_mat, kpt3d_cam_norm.transpose()).transpose()

          if np.isnan(kpt2d_img).any():
            print("=== nan contains", fi, flush=True)

          for kpt in kpt2d_img:
            if np.isnan(kpt).any(): continue

            x = int(kpt[0])
            y = int(kpt[1])
            cv.circle(rgb_img, (x, y), 3, (0, 0, 255), -1)

        writer.append_data(rgb_img)

      writer.close()
      cap.release()


      return 


      result_dict_all = {}
      for p in [0, 1]:
        printv(f"Triangulation on person {p+1}/2", verbose=verbose)
        kpt2d_all_list = {}
        not_valid_arr = {}
        human_data_list = {}
        # for cam_id in cam_map.values():
        for cam_id in person_cams[p]:
          printv(f"Obtaining keypoints from {cam_id} among cameras: {person_cams[p]}", verbose=verbose)
          not_valid_arr[cam_id] = []
          kpt2d_list = []

          kpt2d_path = kpt2d_path_arr[cam_id]
          kpt2d_arr = np.load(kpt2d_path, allow_pickle=True)
          if p not in kpt2d_arr[0]: continue
          frame_len = len(kpt2d_arr)
          print("=======", cam_id, frame_len, flush=True)

          prev_kpt2d = None
          for fi in range(total_frames):
            if "kpt2d" in kpt2d_arr[fi][p]:
              kpt2d = np.array(kpt2d_arr[fi][p]["kpt2d"], dtype=np.float32)[:kpts_num]
              for _i in range(kpts_num):
                kpt = kpt2d[_i]
                x = int(kpt[0])
                y = int(kpt[1])
                ### add perturbation
                if x < 0 or x >= img_width or y < 0 or y >= img_height:
                  kpt2d[_i][2] = 0.0
                # elif kpt2d[_i][2] > 0.8:
                #   kpt2d[_i][2] = np.random.uniform(0.8, 0.95)
                # elif kpt2d[_i][2] < 0.1:
                #   kpt2d[_i][2] = np.random.uniform(0.1, 0.3)

              prev_kpt2d = kpt2d
              kpt2d_list.append(kpt2d)
            elif prev_kpt2d is not None:
              print(f"=== fill kpt2d {cam_id}_{fi:04d} with previous", flush=True)
              kpt2d_list.append(prev_kpt2d)
            else:
              print(f"=== failed to load kpt2d in {cam_id}_{fi:04d}", flush=True)
              not_valid_arr[cam_id].append(fi)
              kpt2d_list.append(invalid_kpt2d)

          kpt2d_list = np.stack(kpt2d_list, axis=0)
          kpt2d_all_list[cam_id] = kpt2d_list


        printv("Obtaining invalid indexes", verbose=verbose)
        invalid_idx_arr = []
        # for cam_id in cam_map.values():
        for cam_id in person_cams[p]:
          invalid_idx_arr += not_valid_arr[cam_id]
        unique_invalid_idx = np.unique(invalid_idx_arr).astype(int)
        unique_mask = np.ones(total_frames, dtype=bool)
        unique_mask[unique_invalid_idx] = False

        ### filter out invalid keypoints
        printv("Filtering out invalid keypoints", verbose=verbose)
        # for cam_id in cam_map.values():
        for cam_id in person_cams[p]:
          cur_kpt2d = kpt2d_all_list[cam_id].copy()
          filtered_cur_kpt2d = cur_kpt2d[unique_mask]
          print(f"cam {cam_id} after filter", filtered_cur_kpt2d.shape, flush=True)

          printv(f"filtered kpts 2d has {len(filtered_cur_kpt2d)} elements", verbose=verbose)

          kpt_dict = {
            'keypoints2d': filtered_cur_kpt2d,
            'keypoints2d_mask': kpt2d_mask,
            'keypoints2d_convention': 'smplx'
          }
          human_data_list[cam_id] = kpt_dict

        # scene = TriangulateScene(cam_para_list, 'auto')
        curr_cam_para = [cam_para_list[cam_id] for cam_id in person_cams[p]]
        scene = TriangulateScene(curr_cam_para, 0.1)
        result_dict = {'optim': {}, 'no_optim': {}, 'invalid_idx': not_valid_arr}
        # triangulate
        keypoints3d_no_optim = []
        keypoints3d_optim = []
        frame_num = human_data_list[cam_id]["keypoints2d"].shape[0]
        frame_num = 4
        printv(f"frame num is {frame_num}", verbose=verbose)
        interval = 2
        printv("Applying triangulation", verbose=verbose)
        for _fi in trange(0, frame_num, interval):
          human_data = []
          start_fi = _fi
          end_fi = min(_fi + interval, frame_num)
          print(start_fi)
          print(end_fi)

          # for cam_id in cam_map.values():
          for cam_id in person_cams[p]:
            cur_dict = {
              'keypoints2d': human_data_list[cam_id]["keypoints2d"][start_fi:end_fi],
              'keypoints2d_mask': kpt2d_mask,
              'keypoints2d_convention': 'smplx'
            }

            printv(f"for {cam_id} the human data are:", verbose=verbose)
            printv(f"keypoints2d -> {cur_dict['keypoints2d'].shape}", verbose=verbose)
            printv(f"keypoints2d -> {cur_dict['keypoints2d_mask'].shape}", verbose=verbose)
            human_data.append(cur_dict)

          kpt3d_no_opt = scene.triangulate(human_data)
          kpt3d_opt = scene.optim(human_data, keypoints3d=kpt3d_no_opt, constraints=None)
          keypoints3d_no_optim.append(kpt3d_no_opt)
          keypoints3d_optim.append(kpt3d_opt)

        result_dict['no_optim']['keypoints3d'] = np.concatenate(keypoints3d_no_optim, axis=0)
        result_dict['optim']['keypoints3d'] = np.concatenate(keypoints3d_optim, axis=0)

        printv(f"Filling result dictionary for person {p+1}/2", verbose=verbose)
        for key in ['optim', 'no_optim']:
          if result_dict[key] is not None:
            keypoints3d = result_dict[key]['keypoints3d']
            human_data_3d = \
                TriangulateScene.convert_result_to_human_data(
                    keypoints3d, kpt2d_mask)
            result_dict[key]['human_data'] = human_data_3d

            human_data_3d['not_valid'] = not_valid_arr
            if 'no_' in key:
                out_npy_path = out_wo_opt_npy_path
            else:
                out_npy_path = out_w_opt_npy_path
            # human_data_3d.dump(out_npy_path)

        print(f"Adding result for person {p}")
        result_dict_all[p] = result_dict

      print(result_dict_all[1])
      printv("Start rendering!", verbose=verbose)
      out_video_path = os.path.join(curr_out_path, "optim_kpt3d_render.mp4")
      print("=== visualization", out_video_path, flush=True)
      writer = imageio.get_writer(
        out_video_path,
        fps=fps, mode='I', format='FFMPEG', macro_block_size=1
      )

      ref_cam = 'GF'
      cap = cv.VideoCapture(vid_paths_dict[ref_cam])
      cam_param = cam_para_list[ref_cam]
      intrinsic_mat = cam_param.get_mat_np('in_mat')
      R_mat = cam_param.get_mat_np('rotation_mat')
      t_vec = np.array(cam_param.get_value('translation'))
      extrinsic_mat = np.eye(4, dtype=np.float32)
      extrinsic_mat[:3, :3] = R_mat.copy()
      if len(t_vec.shape) == 1:
        extrinsic_mat[:3, 3] = t_vec.copy()
      else:
        extrinsic_mat[:3, 3] = t_vec.squeeze()

      for fi in trange(total_frames):
        ret, frame = cap.read()
        if not ret:
          break
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        for p in [0, 1]:
          if p not in result_dict_all: continue

          result_dict = result_dict_all[p]
          keypoints3d = result_dict['optim']['keypoints3d'][..., :3]
          invalid_idx_arr = result_dict['invalid_idx']

          all_invalid = []
          for cam_id in invalid_idx_arr:
            all_invalid += invalid_idx_arr[cam_id]
          unique_invalid_idx = np.unique(all_invalid).astype(int)

          if fi >= len(keypoints3d):
            break
          if fi in unique_invalid_idx:
            print("=== invalid", fi, flush=True)
            continue

          joint_num = keypoints3d.shape[1]
          ones = np.ones((joint_num, 1))

          kpt3d_homo = np.concatenate([keypoints3d[fi], ones], axis=1)
          kpt3d_cam = np.matmul(extrinsic_mat, kpt3d_homo.transpose()).transpose()[:, :3]
          kpt3d_cam_norm = kpt3d_cam / kpt3d_cam[:, 2:]
          kpt2d_img = np.matmul(intrinsic_mat, kpt3d_cam_norm.transpose()).transpose()
          if np.isnan(kpt2d_img).any():
            print("=== nan contains", fi, flush=True)

          for kpt in kpt2d_img:
            if np.isnan(kpt).any(): continue

            x = int(kpt[0])
            y = int(kpt[1])
            cv.circle(rgb_img, (x, y), 3, (0, 0, 255), -1)

        writer.append_data(rgb_img)

      writer.close()
      cap.release()

if __name__ == '__main__':
    main()
    print("=== done", flush=True)
