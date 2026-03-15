# -*- coding: utf-8 -*-
"""
v6/test.py — Entry point testing pipeline v6 (YOLO det + YOLO cls).

Cách dùng:
    python -m v6.test \\
        --input-video ./videos/exam.mp4 \\
        --cls-model-path ./models/best_cheating_classifier_v6.pt

Flags tiện ích:
    --skip-parts 0,1,2   Bỏ qua các part đã xử lý
    --chunk-seconds 300  Thời gian mỗi đoạn (mặc định 5 phút)
    --no-chunk           Không cắt, xử lý nguyên video
"""

import argparse
import os
import shutil
import sys

from .video import chunk_video, process_video, track_histories


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cheating Detection Testing Pipeline v6 (YOLO det + YOLO cls)"
    )
    parser.add_argument('--input-video',    default='./VideoTest.mp4')
    parser.add_argument('--cls-model-path', default='./models/best_cheating_classifier_v6.pt')
    parser.add_argument('--yolo-det-model', default='yolo11l.pt')
    parser.add_argument('--output-dir',     default='./detection_results_v6')
    parser.add_argument('--chunks-dir',     default='./video_chunks_v6')
    parser.add_argument('--chunk-seconds',  type=int, default=300)
    parser.add_argument('--no-chunk',       action='store_true')
    parser.add_argument('--skip-parts',     default='')
    parser.add_argument('--copy-to',        default='')
    return parser.parse_args()


def main():
    args = parse_args()
    track_histories.clear()

    if not os.path.exists(args.input_video):
        print(f"❌ Không tìm thấy video: {args.input_video}")
        sys.exit(1)

    if not os.path.exists(args.cls_model_path):
        print(f"❌ Không tìm thấy classification model: {args.cls_model_path}")
        sys.exit(1)

    from ultralytics import YOLO
    yolo_det_model = YOLO(args.yolo_det_model)
    yolo_cls_model = YOLO(args.cls_model_path)
    print(f"✅ Đã load YOLO det model: {args.yolo_det_model}")
    print(f"✅ Đã load YOLO cls model: {args.cls_model_path}")
    print(f"   Classes: {yolo_cls_model.names}")

    os.makedirs(args.output_dir, exist_ok=True)
    skip_indices: set[int] = (
        {int(x.strip()) for x in args.skip_parts.split(',')}
        if args.skip_parts else set()
    )

    video_files = ([args.input_video] if args.no_chunk
                   else chunk_video(args.input_video, args.chunks_dir, args.chunk_seconds))

    for idx, video_part in enumerate(video_files):
        part_name = os.path.basename(video_part)
        if idx in skip_indices:
            print(f"⏭️  Bỏ qua {part_name} (--skip-parts)")
            continue

        output_path = os.path.join(args.output_dir, f"detected_{part_name}")
        process_video(video_part, output_path, yolo_det_model, yolo_cls_model)

        if args.copy_to:
            os.makedirs(args.copy_to, exist_ok=True)
            dest = os.path.join(args.copy_to, f"detected_{part_name}")
            shutil.copy2(output_path, dest)
            print(f"📋 Đã copy sang: {dest}")

    print("\n🎉 TẤT CẢ CÁC PHẦN ĐÃ HOÀN TẤT!")


if __name__ == '__main__':
    main()
