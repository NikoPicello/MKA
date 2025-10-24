import cv2
import os
import logging
import shutil
import hashlib
import gc
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis


def apply_mosaic(face_region, mosaic_size=8):
    (h, w) = face_region.shape[:2]
    mw = min(mosaic_size, int(w / mosaic_size))
    mh = min(mosaic_size, int(h / mosaic_size))
    temp = cv2.resize(face_region, (mw, mh), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    return mosaic


def process_video_opencv_dnn(model, input_video_path, output_video_path):
    # Open video capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    fid = 0
    miss_det_frame_count = []
    false_det_frame_count = []
    part_det_frame_count = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        fid += 1

        frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = model.get(frame_)

        if len(faces) == 0:
            miss_det_frame_count.append(fid)
        else:
            for face in faces :
                x, y, x2, y2 = face['bbox'].astype(int)
                x, y, x2, y2 = max(0, x), max(0, y), min(frame.shape[1], x2), min(frame.shape[0], y2)
                face_region = frame[y:y2, x:x2]
                mosaic_face = apply_mosaic(face_region)
                frame[y:y2, x:x2] = mosaic_face
        out.write(frame)
    
    logging.warning(f'{output_video_path} missing det: {len(miss_det_frame_count)}')
    cap.release()


if __name__ == "__main__":

    src_video_folder = r'E:\TTSH\data_collection_all\uni_vid_mp4\uni_vid_mp4'
    dest_video_folder = r'E:\TTSH\data_collection_all\uni_vid_mp4\mosaic'

    # Load pre-trained model
    model = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], provider_options=[{"device_id": 0}, {}])
    model.prepare(ctx_id=0, det_size=(640, 640))

    RMs = [item.path for item in os.scandir(src_video_folder) if item.is_dir()]
    RMs.sort()
    for i in range(12, 20):
        rm = RMs[i]
        rmid = os.path.basename(rm)
        actions = [item.path for item in os.scandir(rm) if item.is_dir()]
        for action in actions:
            action_name = os.path.basename(action)
            os.makedirs(os.path.join(dest_video_folder, rmid, action_name), exist_ok=True)
            videos = [os.path.basename(item.path) for item in os.scandir(action) if item.is_file()]
            for vid in videos :
                vid_path = os.path.join(src_video_folder, rmid, action_name, vid)
                save_path = os.path.join(dest_video_folder, rmid, action_name, vid)
                process_video_opencv_dnn(model, vid_path, save_path)
            gc.collect()
