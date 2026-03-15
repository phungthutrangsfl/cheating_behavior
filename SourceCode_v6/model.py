# -*- coding: utf-8 -*-
"""
v6/model.py — YOLO classification (yolo11l-cls.pt) fine-tuned.
Không dùng CNN PyTorch — mọi training/inference đều qua YOLO API.
"""

import os
import shutil

import pandas as pd
from ultralytics import YOLO

IMG_SIZE = 224


def ensure_val_dir(data_dir: str) -> bool:
    """YOLO cls yêu cầu thư mục tên 'val', không phải 'valid'.
    Nếu thấy 'valid' → đổi tên → trả về True."""
    valid_path = os.path.join(data_dir, 'valid')
    val_path   = os.path.join(data_dir, 'val')
    if os.path.exists(valid_path) and not os.path.exists(val_path):
        os.rename(valid_path, val_path)
        print(f"📁 Đã đổi tên thư mục: valid → val")
        return True
    return False


def restore_valid_dir(data_dir: str, was_renamed: bool):
    """Đổi lại tên 'val' → 'valid' nếu đã đổi trước đó."""
    if not was_renamed:
        return
    val_path   = os.path.join(data_dir, 'val')
    valid_path = os.path.join(data_dir, 'valid')
    if os.path.exists(val_path) and not os.path.exists(valid_path):
        os.rename(val_path, valid_path)
        print(f"📁 Đã khôi phục tên thư mục: val → valid")


def train_yolo_classifier(
    data_dir: str,
    cls_model_name: str,
    model_save_path: str,
    epochs: int,
    batch_size: int,
    imgsz: int,
    patience: int,
    lr: float,
    device_str: str,
    output_dir: str,
) -> tuple:
    """Fine-tune YOLO classifier, trả về (best_model, history dict)."""
    was_renamed = ensure_val_dir(data_dir)

    model = YOLO(cls_model_name)
    print(f"\n🚀 Bắt đầu fine-tune YOLO classifier: {cls_model_name}")
    print(f"   Data:     {data_dir}")
    print(f"   Epochs:   {epochs}")
    print(f"   Batch:    {batch_size}")
    print(f"   Img size: {imgsz}")
    print(f"   Patience: {patience}")
    print(f"   LR:       {lr}")
    print(f"   Device:   {device_str}")

    try:
        model.train(
            data=data_dir,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            patience=patience,
            lr0=lr,
            device=device_str,
            project=output_dir,
            name='yolo_cls_run',
            exist_ok=True,
            verbose=True,
        )
    finally:
        restore_valid_dir(data_dir, was_renamed)

    run_dir   = os.path.join(output_dir, 'yolo_cls_run')
    best_pt   = os.path.join(run_dir, 'weights', 'best.pt')
    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
    shutil.copy2(best_pt, model_save_path)
    print(f"✅ Đã copy best.pt → {model_save_path}")

    best_model = YOLO(model_save_path)
    history    = parse_yolo_history(run_dir)
    return best_model, history


def parse_yolo_history(save_dir: str) -> dict:
    """Đọc results.csv → dict history."""
    csv_path = os.path.join(save_dir, 'results.csv')
    if not os.path.exists(csv_path):
        print(f"⚠️ Không tìm thấy results.csv tại {save_dir}")
        return {'train_loss': [], 'valid_loss': [], 'valid_acc': []}

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Cột YOLO classification: train/loss, val/loss, metrics/accuracy_top1
    train_loss_col = next((c for c in df.columns if 'train' in c.lower() and 'loss' in c.lower()), None)
    valid_loss_col = next((c for c in df.columns if 'val'   in c.lower() and 'loss' in c.lower()), None)
    valid_acc_col  = next((c for c in df.columns if 'accuracy' in c.lower() and 'top1' in c.lower()), None)

    history = {
        'train_loss': df[train_loss_col].tolist() if train_loss_col else [],
        'valid_loss': df[valid_loss_col].tolist() if valid_loss_col else [],
        'valid_acc':  df[valid_acc_col].tolist()  if valid_acc_col  else [],
    }
    print(f"✅ YOLO history: {len(history['valid_acc'])} epochs đọc từ {csv_path}")
    return history
