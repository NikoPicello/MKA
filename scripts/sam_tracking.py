import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm, trange
import gc
import json
import argparse
import cv2
import imageio 

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

import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_path, "dependencies"))
from sam2.build_sam import build_sam2_video_predictor


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument('--prompt_file', type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    return args 


def main(args):
    in_root = args.video_dir
    prompt_file = args.prompt_file
    out_root = args.out_dir 
    os.makedirs(out_root , exist_ok=True)
    
    sam2_checkpoint = f"{root_path}/pretrained_models/sam2.1_hiera_large.pt"
    model_cfg = f"configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    
    with open(prompt_file, 'r') as fin:
        cam_dict = json.load(fin)

    video_filenames = sorted([x for x in os.listdir(in_root) if x.endswith(".mp4") or x.endswith(".mkv")])
    for video_fn in video_filenames:
        video_name = video_fn.split(".")[0]
        if video_fn not in cam_dict :
            continue
        elif len(cam_dict[video_fn]) == 0:
            continue
        else :
            bbox_prompt = cam_dict[video_fn]["bbox"]
        
        
        os.makedirs(os.path.join(out_root, "mask", video_name) , exist_ok=True)
        os.makedirs(os.path.join(out_root, "overlay") , exist_ok=True)
        
        video_path = os.path.join(in_root, video_fn)
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out_video_path = os.path.join(out_root, "overlay", f"{video_name}_overlay.mp4")
        writer = imageio.get_writer(
            out_video_path, 
            fps=fps, mode='I', format='FFMPEG', macro_block_size=1
        )
        
        raw_images = []
        for fidx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_images.append(frame)

        inference_state = predictor.init_state(video_path=video_path)
        predictor.reset_state(inference_state)

        # # Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
        # box = np.array([300, 0, 500, 400], dtype=np.float32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=4,
            box=np.array(bbox_prompt, dtype=np.float32),
        )

        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        for out_frame_idx in trange(len(video_segments)):
            image_np = raw_images[out_frame_idx]
            seg_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
            overlayed = image_np.copy()

            # Overlay mask(s)
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                mask = np.squeeze(out_mask).astype(bool)

                # --- 1. Update seg mask (binary: 0/1) ---
                seg_mask[mask] = 1

                # --- 2. Draw overlay ---
                color = (255, 0, 0)  # red overlay for mask
                overlay = np.zeros_like(image_np, dtype=np.uint8)
                overlay[mask] = color

                # Alpha blend overlay into original image
                alpha = 0.5  # transparency
                overlayed = cv2.addWeighted(overlay, alpha, overlayed, 1 - alpha, 0)

            # Save segmentation mask (single-channel, values 0/1)
            result_path = os.path.join(out_root, "mask", video_name, f"{out_frame_idx:05d}.png")
            Image.fromarray(seg_mask).save(result_path)

            # Append overlayed frame to video
            writer.append_data(overlayed)
    
        cap.release()
        writer.close()


if __name__ == '__main__':
    args = get_args_parser()
    main(args)
    print("=== done", flush=True)