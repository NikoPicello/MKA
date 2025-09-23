import os, sys, re
import shutil
import numpy as np
from tqdm import tqdm, trange
import imageio
import subprocess
import json 
import argparse

def get_frames(basdir, task):
    total_timestamp, total_frame_num = [], []

    if task == 'rgb':
        stream = 'v:0'
    elif task == 'depth':
        stream = 'v:1'
    elif task == 'ir':
        stream = 'v:2'

    for i in range(16):
        select_stream = stream
        mkv_path = os.path.join(basdir, 'k4a_multiview_cam%03d.mkv' % i)
        if os.path.exists( mkv_path ):
            cmd = [
                'ffprobe', mkv_path,
                '-select_streams', select_stream, 
                '-show_entries', 'frame=pkt_pts_time', 
                '-of', 'csv=p=0:nk=1', '-v', '0'
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE)
            timestamp = result.stdout.decode("utf-8").split('\n')[:-1]
            timestamp = [t for t in timestamp if t != '']
            timestamp = np.array([float(t) for t in timestamp])
        else:
            timestamp=np.zeros(0)
        
        total_timestamp.append(timestamp)
        total_frame_num.append(len(timestamp))

    return total_timestamp, total_frame_num

def extract_frames(videodir, outbase, kinect_idx, task, start_num, length, s_idx=0):
    
    videobase = os.path.basename(videodir)
    outdir = os.path.join(outbase, "kinect_"+task, '%02d'%kinect_idx)
    # if os.path.exists(outdir):
    #     shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)

    print("=== outdir", outdir, length, flush=True)
    if task == 'rgb':
        stream = '0:0'
    elif task == 'depth':
        stream = '0:1'
    elif task == 'ir':
        stream = '0:2'   

    if task == 'rgb':
        suffix = 'jpg'
    else:
        suffix = 'png'

    cmd = "ffmpeg -i "+ videodir + \
        f" -vf select=\'between(n\,{start_num}\,{start_num+length-1})\'" + f" -map {stream} -vsync 0" + \
        f" -start_number {s_idx} -hide_banner -loglevel error -qscale:v 2 "+outdir+f"/%05d.{suffix}" 
    os.system(cmd)

    return outdir

def reorganize_frames(datadir, timestamps, start_num, correct_timestamps):
    extracted_frames = sorted(os.listdir(datadir))
    extracted_len = len(extracted_frames)
    refimg = imageio.imread(os.path.join(datadir, extracted_frames[0]))
    suffix = extracted_frames[0][-3:]
    black_img = np.zeros_like(refimg)

    frameids = np.clip(np.rint(timestamps), start_num, start_num + len(correct_timestamps)-1).tolist()
    ## move frame to nearest correct frame
    for i in range(start_num, start_num + len(correct_timestamps)):
        if i in frameids:
            idx = frameids.index(i)
            fid = frameids[idx]
            fname = extracted_frames[idx]
            if fid != idx:
                os.rename(os.path.join(datadir, fname), os.path.join(datadir, '_%05d.%s' % (i+1, suffix)))
    ## insert missing frame
    for i in range(start_num, start_num + len(correct_timestamps)):
        if i not in frameids:
            imageio.imwrite(os.path.join(datadir, '%05d.%s' % (i+1, suffix)), black_img)

    frames = sorted(os.listdir(datadir))
    for frame in frames:
        if frame[0] == '_':
            os.rename(os.path.join(datadir, frame), os.path.join(datadir, frame[1:]))

def get_intrinsic_data(intrinsic_file, save_dir):
    with open(intrinsic_file, "r") as fin:
        all_lines = fin.readlines()
    
    total_line = len(all_lines)  
    view_num = 16
    all_cam_arr = []
    for _ in range(view_num):
        all_cam_arr.append({})
    cam_idx = -1
    for i in range(total_line):
        line = all_lines[i].rstrip()
        if len(line) < 1:
            continue 

        if line.startswith("camera_order"):
            sub_line = line.split(",")
            cam_idx = int(sub_line[0].split(":")[-1])
            # print(cam_idx, flush=True)
            cam_sn = sub_line[1].split(":")[-1]
            all_cam_arr[cam_idx]["serail_number"] = cam_sn
        elif cam_idx > -1:
            sub_line = line.split(":")
            all_cam_arr[cam_idx][sub_line[0]] = sub_line[1]            
    
    # print(all_cam_arr[0], all_cam_arr[1])
    for i in range(view_num):
        image_size = [
            int(all_cam_arr[i]["resolution_width"]),
            int(all_cam_arr[i]["resolution_height"]),
        ]
        K = np.eye(3, dtype=np.float32)
        K[0][0] = float(all_cam_arr[i]["fx"])
        K[0][2] = float(all_cam_arr[i]["cx"])
        K[1][1] = float(all_cam_arr[i]["fy"])
        K[1][2] = float(all_cam_arr[i]["cy"])
        dist_arr = [
            [   
                float(all_cam_arr[i]["k1"]),
                float(all_cam_arr[i]["k2"]),
                float(all_cam_arr[i]["p1"]),
                float(all_cam_arr[i]["p2"]),
                float(all_cam_arr[i]["k3"]),
            ]
        ]
        cam_dict = {
            "cameras": {
                "kinect_rgb/%02d" % i:{
                    "model": "standard",
                    "image_size": image_size,
                    "K" : K.tolist(),
                    "dist" : dist_arr,
                    "serial_num": all_cam_arr[i]["serail_number"]
                }
            }
        }
        save_file = os.path.join(save_dir, "kinect_rgb_%02d.json" % i)
        with open(save_file, 'w') as fout:
            json.dump(cam_dict, fout, indent=2)


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='./results')
    parser.add_argument('--view_num', type=int, default=16)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--s_idx', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = options()
    os.makedirs(args.out_dir, exist_ok=True)
    basedir = args.in_dir 
    outdir = args.out_dir 
    framerate = args.fps 

    # intrinsic_file = os.path.join(basedir, "intrinsic.txt")
    # if os.path.exists(intrinsic_file):
    #     save_dir = os.path.join(outdir, "intrinsic")
    #     os.makedirs(save_dir, exist_ok=True)
    #     get_intrinsic_data(intrinsic_file, save_dir)

    # exit(0)

    tasks = ['rgb']
    correct_frame_num = 1000 ### 3 seconds
    start_num = 0 #195 #35
    view_num = args.view_num
    for task in tasks:
        for kinect_idx in range(view_num):
            videodir = os.path.join(basedir, 'k4a_multiview_cam%03d.mkv' % kinect_idx)
            tmpdir = os.path.join(outdir, "kinect_"+task, '%02d'%kinect_idx)
            # tmpdir = os.path.join(outdir, '%02d'%kinect_idx)
            extract_frames(videodir, outdir, kinect_idx, task, start_num, correct_frame_num, args.s_idx)
        
            # break 
            # exit(0)

    print("=== done", flush=True)