# -*- coding: utf-8 -*-
"""
v6/video.py — Cắt video thành chunks và xử lý từng frame với YOLO cls (multi-people tracking).
"""

import glob as _glob
import os
import subprocess
from collections import deque

import cv2
import numpy as np
from tqdm import tqdm

IMG_SIZE        = 224
HISTORY_LEN     = 50
FRAME_THRESHOLD = 0.4   # prob(normal) dưới ngưỡng này → nghi gian lận
ALARM_THRESHOLD = 0.6   # tỉ lệ lịch sử suspect vượt ngưỡng → báo động

track_histories: dict[int, deque] = {}


def chunk_video(input_path: str, chunks_dir: str,
                segment_seconds: int = 300) -> list[str]:
    os.makedirs(chunks_dir, exist_ok=True)
    pattern = os.path.join(chunks_dir, "part_%03d.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-f", "segment", "-segment_time", str(segment_seconds),
        "-c", "copy", pattern, "-loglevel", "error",
    ]
    print(f"✂️  Đang cắt video thành từng đoạn {segment_seconds}s ...")
    subprocess.run(cmd, check=True)
    chunk_files = sorted(_glob.glob(os.path.join(chunks_dir, "*.mp4")))
    print(f"✅ Đã cắt xong {len(chunk_files)} đoạn video.")
    for f in chunk_files:
        print(f"   - {f}")
    return chunk_files


def run_pipeline_multi_people(frame: np.ndarray,
                               yolo_det_model,
                               yolo_cls_model) -> np.ndarray:
    global track_histories
    frame_copy = frame.copy()
    names      = yolo_cls_model.names   # dict {int: str}

    # Tìm index của class 'normal'
    normal_idx = next((k for k, v in names.items() if v.lower() == 'normal'), None)

    results = yolo_det_model.track(frame, persist=True, classes=[0], verbose=False)

    if results[0].boxes.id is not None:
        boxes     = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            cls_results = yolo_cls_model(person_crop, imgsz=IMG_SIZE, verbose=False)
            probs       = cls_results[0].probs

            if normal_idx is not None:
                prob_normal = float(probs.data[normal_idx])
            else:
                # fallback: tìm class chứa 'normal' trong tên
                prob_normal = 0.0
                for k, v in names.items():
                    if v.lower() == 'normal':
                        prob_normal = float(probs.data[k]); break

            is_suspect = 1 if prob_normal < FRAME_THRESHOLD else 0

            if track_id not in track_histories:
                track_histories[track_id] = deque(maxlen=HISTORY_LEN)
            track_histories[track_id].append(is_suspect)

            h          = track_histories[track_id]
            risk_score = sum(h) / len(h) if h else 0

            color = (0, 0, 255) if risk_score > ALARM_THRESHOLD else (0, 255, 0)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_copy, f"ID:{track_id} ({risk_score:.0%})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            bar_width = int((x2 - x1) * risk_score)
            cv2.rectangle(frame_copy, (x1, y1 - 5), (x1 + bar_width, y1),
                          (0, 0, 255), -1)
    return frame_copy


def process_video(video_path: str, output_path: str,
                  yolo_det_model, yolo_cls_model):
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out  = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    pbar = tqdm(total=total, desc=f"Processing {os.path.basename(video_path)}",
                unit="frame", ncols=100)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(run_pipeline_multi_people(frame, yolo_det_model, yolo_cls_model))
        pbar.update(1)

    pbar.close(); cap.release(); out.release()
    print(f"✅ Đã xong. Kết quả: {output_path}")
