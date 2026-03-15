# -*- coding: utf-8 -*-
"""
v6/detector.py — Inference pipeline dùng YOLO detection + YOLO classification.
"""

import os
from glob import glob

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

IMG_SIZE = 224


def run_pipeline(frame: np.ndarray,
                 yolo_det_model,
                 yolo_cls_model) -> np.ndarray:
    frame_copy = frame.copy()
    results    = yolo_det_model(frame, classes=[0], verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            cls_results = yolo_cls_model(crop, imgsz=IMG_SIZE, verbose=False)
            probs = cls_results[0].probs
            names = yolo_cls_model.names          # dict {0: 'cheating', 1: 'normal'} (or reversed)

            pred_class = names[probs.top1]
            confidence = float(probs.top1conf)

            color = (0, 0, 255) if 'cheat' in pred_class.lower() else (0, 255, 0)
            label = pred_class.upper()
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_copy, f"{label} ({confidence:.2f})",
                        (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame_copy


def visual_validation(data_dir: str,
                      yolo_det_model,
                      yolo_cls_model,
                      output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    val_cheating = glob(os.path.join(data_dir, 'valid', 'cheating', '*.jpg'))
    val_normal   = glob(os.path.join(data_dir, 'valid', 'normal',   '*.jpg'))

    n = 5
    rng = np.random.default_rng(42)
    sample_cheat = (rng.choice(val_cheating, n, replace=False)
                    if len(val_cheating) >= n else np.array(val_cheating))
    sample_normal = (rng.choice(val_normal, n, replace=False)
                     if len(val_normal)   >= n else np.array(val_normal))

    rows = max(len(sample_cheat), len(sample_normal))
    if rows == 0:
        print("⚠️  Không có ảnh validation để hiển thị.")
        return

    fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]

    axes[0, 0].set_title("CHEATING SAMPLES", fontsize=16, color='red',   pad=15)
    axes[0, 1].set_title("NORMAL SAMPLES",   fontsize=16, color='green', pad=15)

    for i in range(rows):
        for j, samples in enumerate([sample_cheat, sample_normal]):
            if i < len(samples):
                img = cv2.imread(str(samples[i]))
                if img is not None:
                    res = run_pipeline(img, yolo_det_model, yolo_cls_model)
                    axes[i, j].imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
            axes[i, j].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'visual_validation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu ảnh validation: {save_path}")
