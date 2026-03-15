# -*- coding: utf-8 -*-
"""
v6/train.py — Entry point training pipeline v6 (YOLO det + YOLO cls).

Cách dùng:
    python -m v6.train \\
        --data-dir ./merged_local \\
        --model-save-path ./models/best_cheating_classifier_v6.pt \\
        --dataset-dir ./datasets \\
        --epochs 50

Flags tiện ích:
    --skip-download   Bỏ qua tải dataset
    --skip-merge      Bỏ qua merge/crop ảnh
    --skip-train      Bỏ qua train, chỉ evaluate + visual validation
    --resplit         Pool & chia lại train/valid/test
"""

import argparse
import os
import sys

from .data import (ROBOFLOW_PROJECTS, build_configs, check_data_distribution,
                   download_datasets, merge_datasets, resplit_dataset)
from .evaluate import (evaluate_test_set, plot_label_distribution,
                       plot_training_curves, save_history)
from .model import train_yolo_classifier
from .detector import visual_validation


def _resolve_device(device_arg: str) -> str:
    """Chuyển device arg thành string YOLO hiểu (int string hoặc 'cpu')."""
    if device_arg.lower() == 'cpu':
        return 'cpu'
    try:
        import torch
        if torch.cuda.is_available():
            gpu_idx = int(device_arg) if device_arg.isdigit() else 0
            return str(gpu_idx)
    except Exception:
        pass
    return 'cpu'


def parse_args():
    parser = argparse.ArgumentParser(description="Cheating Detection Training Pipeline v6")
    parser.add_argument('--api-key',         default='4TqRDRgaSqazkFSZxRGx')
    parser.add_argument('--dataset-dir',     default='./datasets')
    parser.add_argument('--data-dir',        default='./merged_local')
    parser.add_argument('--model-save-path', default='./models/best_cheating_classifier_v6.pt')
    parser.add_argument('--yolo-det-model',  default='yolo11l.pt',
                        help='YOLO người-detect model (e.g. yolo11l.pt)')
    parser.add_argument('--yolo-cls-model',  default='yolo11l-cls.pt',
                        help='YOLO classification base model')
    parser.add_argument('--epochs',          type=int,   default=50)
    parser.add_argument('--batch-size',      type=int,   default=32)
    parser.add_argument('--lr',              type=float, default=1e-3)
    parser.add_argument('--patience',        type=int,   default=10)
    parser.add_argument('--imgsz',           type=int,   default=224)
    parser.add_argument('--device',          default='0',
                        help="GPU index (e.g. '0') hoặc 'cpu'")
    parser.add_argument('--skip-download',   action='store_true')
    parser.add_argument('--skip-merge',      action='store_true')
    parser.add_argument('--skip-train',      action='store_true')
    parser.add_argument('--output-dir',      default='./outputs_v6')
    parser.add_argument('--resplit',         action='store_true')
    parser.add_argument('--train-ratio',     type=float, default=0.80)
    parser.add_argument('--valid-ratio',     type=float, default=0.10)
    parser.add_argument('--split-seed',      type=int,   default=42)
    return parser.parse_args()


def main():
    args       = parse_args()
    device_str = _resolve_device(args.device)
    print(f"🖥️  Device string cho YOLO: {device_str!r}")

    # ── Bước 1: Tải dataset ──────────────────────────────────────────────
    if not args.skip_download:
        dataset_objects = download_datasets(args.api_key, args.dataset_dir)
    else:
        print("⏭️  Bỏ qua tải dataset (--skip-download)")
        class _DS:
            def __init__(self, loc): self.location = loc
        dataset_objects = [
            _DS(os.path.join(args.dataset_dir, proj))
            for _, proj, _ in ROBOFLOW_PROJECTS
        ]

    # ── Bước 2: Merge dataset ────────────────────────────────────────────
    if not args.skip_merge:
        configs = build_configs(dataset_objects)
        merge_datasets(configs, args.data_dir)
    else:
        print("⏭️  Bỏ qua merge dataset (--skip-merge)")

    if args.resplit:
        resplit_dataset(args.data_dir, args.train_ratio, args.valid_ratio, args.split_seed)
    else:
        print("⏭️  Bỏ qua resplit (truyền --resplit để bật)")

    check_data_distribution(args.data_dir)
    plot_label_distribution(args.data_dir, args.output_dir)

    # ── Bước 3: Train YOLO classifier ────────────────────────────────────
    if not args.skip_train:
        yolo_cls_model, history = train_yolo_classifier(
            data_dir        = args.data_dir,
            cls_model_name  = args.yolo_cls_model,
            model_save_path = args.model_save_path,
            epochs          = args.epochs,
            batch_size      = args.batch_size,
            imgsz           = args.imgsz,
            patience        = args.patience,
            lr              = args.lr,
            device_str      = device_str,
            output_dir      = args.output_dir,
        )
        save_history(history, args.output_dir)
        plot_training_curves(history, args.output_dir)
        evaluate_test_set(yolo_cls_model, args.data_dir, args.output_dir, args.batch_size)
    else:
        print("⏭️  Bỏ qua train (--skip-train)")
        if not os.path.exists(args.model_save_path):
            print(f"❌ Không tìm thấy model tại: {args.model_save_path}")
            sys.exit(1)
        from ultralytics import YOLO
        yolo_cls_model = YOLO(args.model_save_path)
        evaluate_test_set(yolo_cls_model, args.data_dir, args.output_dir, args.batch_size)

        import json
        history_path = os.path.join(args.output_dir, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
            plot_training_curves(history, args.output_dir)

    # ── Bước 4: Visual Validation ─────────────────────────────────────────
    print("\n🔍 Chạy visual validation...")
    from ultralytics import YOLO
    yolo_det_model = YOLO(args.yolo_det_model)
    yolo_cls_model = yolo_cls_model if not args.skip_train else YOLO(args.model_save_path)
    visual_validation(args.data_dir, yolo_det_model, yolo_cls_model, args.output_dir)

    print("\n🎉 Hoàn tất!")


if __name__ == '__main__':
    main()
