import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm, trange
from pathlib import Path
import glob
import gc
import json
import argparse
import cv2 as cv
import imageio
import sys
from ultralytics import YOLO

# select the device for computation
if torch.cuda.is_available():
  device = torch.device("cuda")
elif torch.backends.mps.is_available():
  device = torch.device("mps")
else:
  device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
  # use bfloat16 for the entire notebook
  torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
  # # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
  # if torch.cuda.get_device_properties(0).major >= 8:
  #     torch.backends.cuda.matmul.allow_tf32 = True
  #     torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
  print(
    "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
    "give numerically different outputs and sometimes degraded performance on MPS. "
    "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
  )

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
  root_path = os.path.join('/', sys.path[0])
  sys.path.append(os.path.join(root_path, "dependencies"))
  from sam2.build_sam import build_sam2_video_predictor

  main_path = '/'.join(sys.path[0].split('/')[:-2]) + '/'
  resources_path = os.path.join(main_path, 'mka')
  calibs_path   = os.path.join(resources_path, 'calibs')
  sessions_path = os.path.join(resources_path, 'sessions')
  out_path = os.path.join(resources_path, 'sam_results')
  sid_paths = sorted(glob.glob(sessions_path + '/*'))

  sam2_checkpoint = f"{root_path}/pretrained_models/sam2.1_hiera_large.pt"
  model_cfg = f"configs/sam2.1/sam2.1_hiera_l.yaml"
  sam2 = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
  print(sid_paths)

  yolo = YOLO(f"{root_path}/pretrained_models/yolov8x.pt")
  print(sid_paths)

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
        cap = cv.VideoCapture(vid_path)
        fps = int(cap.get(cv.CAP_PROP_FPS))
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        os.makedirs(os.path.join(out_path, "mask", video_name) , exist_ok=True)
        os.makedirs(os.path.join(out_path, "overlay") , exist_ok=True)
        out_vid_path = os.path.join(out_path, "overlay", f"{video_name}_overlay.mp4")
        writer = imageio.get_writer(
          out_vid_path,
          fps=fps, mode='I', format='FFMPEG', macro_block_size=1
        )

        cap.set(cv.CAP_PROP_POS_FRAMES, total_frames//2)
        ret, frame = cap.read()
        cap.release()
        assert ret, f"Could not read first frame of {vid_path}"

        res = yolo(frame, classes=[0], device='cpu')  # class 0 = person
        boxes = res[0].boxes.xyxy.cpu().numpy()  # shape (N, 4)

        confs = res[0].boxes.conf.cpu().numpy()
        top2_idx = np.argsort(confs)[::-1][:2]
        bbox_prompt_a, bbox_prompt_b = boxes[top2_idx]  # shape (2, 4)


        inference_state = sam2.init_state(video_path=vid_path)
        sam2.reset_state(inference_state)

        _, out_obj_ids, out_mask_logits = sam2.add_new_points_or_box(
          inference_state=inference_state,
          frame_idx=0,
          obj_id=4,
          box=np.array(bbox_prompt_a, dtype=np.float32),
        )

        # video_segments = {}  # video_segments contains the per-frame segmentation results
        # for out_frame_idx, out_obj_ids, out_mask_logits in sam2.propagate_in_video(inference_state):
        #   video_segments[out_frame_idx] = {
        #     out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        #     for i, out_obj_id in enumerate(out_obj_ids)
        #   }

        sam2.reset_state(inference_state)
        del inference_state
        torch.cuda.empty_cache()

        cap = cv.VideoCapture(vid_path)
        for out_frame_idx, out_obj_ids out_mask_logits in sam2.propagate_in_video(inference_state):
          ret, image_np = cap.read()
          image_np = cv.cvtColor(image_np, cv.COLOR_BGR2RGB)
          seg_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
          overlayed = image_np.copy()

          curr_segment = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
          }

          # Overlay mask(s)
          # for out_obj_id, out_maks in video_segments[out_frame_idx].items():
          for out_obj_id, out_mask in curr_segment.items():
            mask = np.squeeze(out_mask).astype(bool)

            # --- 1. Update seg mask (binary: 0/1) ---
            seg_mask[mask] = 1

            # --- 2. Draw overlay ---
            color = (255, 0, 0)  # red overlay for mask
            overlay = np.zeros_like(image_np, dtype=np.uint8)
            overlay[mask] = color

            # Alpha blend overlay into original image
            alpha = 0.5  # transparency
            overlayed = overlayed * (1 - mask[..., None]) + ((overlayed * (1 - alpha)) + mask[..., None] * color * alpha) * mask[..., None]
            overlayed = overlayed.astype(np.uint8)

          # Save segmentation mask (single-channel, values 0/1)
          result_path = os.path.join(out_path, "mask", video_name, f"{out_frame_idx:05d}.png")
          Image.fromarray(seg_mask).save(result_path)

          # Append overlayed frame to video
          writer.append_data(overlayed)

        cap.release()
        writer.close()

        # del video_segments
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
  main()
