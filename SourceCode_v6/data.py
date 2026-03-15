# -*- coding: utf-8 -*-
"""
v6/data.py — Tải dataset Roboflow, merge/crop ảnh, resplit.
Giống v5/data.py (YOLO cls cần cùng cấu trúc thư mục).
"""

import os
import random
import shutil
from glob import glob

import cv2
import yaml
from roboflow import Roboflow

IMG_SIZE = 224

ROBOFLOW_PROJECTS = [
    ("cheating-yas5o", "cheating_detection_dataset_2-o5yh5",  1),
    ("cheating-yas5o", "cheating-detection-wmju5-jc0ns",      1),
    ("cheating-yas5o", "exam-cheating-9iz1y-vymkj",           1),
    ("cheating-yas5o", "graduation-project-jewvv-47yq0",      1),
]

CHEAT_KEYWORDS  = ['cheat', 'phone']
NORMAL_KEYWORDS = ['normal']

MANUAL_OVERRIDES = {
    'cheating_det_wmju5': ([0], [1]),
    'exam_cheating': ([0, 1, 3, 4, 5, 6, 7], [2]),
}

DATASET_NAMES = ['cheating_det_2', 'cheating_det_wmju5', 'exam_cheating', 'grad_project']


def download_datasets(api_key: str, save_dir: str) -> list:
    rf = Roboflow(api_key=api_key)
    os.makedirs(save_dir, exist_ok=True)
    dataset_objects = []
    for ws, proj, ver in ROBOFLOW_PROJECTS:
        print(f"⏳ Đang tải {proj} v{ver}...")
        ds = rf.workspace(ws).project(proj).version(ver).download(
            "yolov11", location=os.path.join(save_dir, proj)
        )
        dataset_objects.append(ds)
        print(f"  ✅ Lưu tại: {ds.location}")
    print(f"\n✅ Tất cả {len(dataset_objects)} dataset đã tải thành công!")
    return dataset_objects


def auto_detect_class_mapping(dataset_path: str):
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    if not os.path.exists(yaml_path):
        print(f"  ⚠️ Không tìm thấy data.yaml tại {yaml_path}")
        return [], []

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    names = data.get('names', {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}

    cheat_ids, normal_ids = [], []
    for cls_id, cls_name in names.items():
        cls_lower = str(cls_name).lower()
        if any(kw in cls_lower for kw in CHEAT_KEYWORDS):
            cheat_ids.append(int(cls_id))
        elif any(kw in cls_lower for kw in NORMAL_KEYWORDS):
            normal_ids.append(int(cls_id))

    print(f"  📋 Classes: {names}")
    print(f"  ✅ Auto-detect → Cheating IDs: {cheat_ids}, Normal IDs: {normal_ids}")
    return cheat_ids, normal_ids


def build_configs(dataset_objects: list) -> list:
    configs = []
    for ds, name in zip(dataset_objects, DATASET_NAMES):
        print(f"\n🔍 Đang đọc class mapping: {name} ({ds.location})")
        override = MANUAL_OVERRIDES.get(name)
        if override == 'SKIP':
            print(f"  ⏭️ BỎ QUA (labeling bị lỗi, đã verify)")
            continue
        c_ids, n_ids = auto_detect_class_mapping(ds.location)
        if not c_ids and not n_ids:
            if isinstance(override, tuple):
                c_ids, n_ids = override
                print(f"  🔧 Dùng manual override → Cheating IDs: {c_ids}, Normal IDs: {n_ids}")
            else:
                print(f"  ⚠️ Không phát hiện được class nào, bỏ qua!")
                continue
        configs.append({'path': ds.location, 'cheat_ids': c_ids,
                        'normal_ids': n_ids, 'name': name})
    print(f"\n📊 Số dataset sẽ merge: {len(configs)}/{len(DATASET_NAMES)}")
    return configs


def merge_datasets(configs: list, output_path: str):
    for split in ['train', 'valid', 'test']:
        for cls in ['cheating', 'normal']:
            os.makedirs(os.path.join(output_path, split, cls), exist_ok=True)

    print("\n🚀 Đang gộp dữ liệu...")
    for cfg in configs:
        path      = cfg['path']
        c_ids     = cfg['cheat_ids']
        n_ids     = cfg['normal_ids']
        ds_prefix = cfg['name']
        count     = {'cheating': 0, 'normal': 0, 'skipped': 0}

        for split in ['train', 'valid', 'test']:
            actual_split = split
            if split == 'valid' and not os.path.exists(os.path.join(path, 'valid')):
                actual_split = 'val'

            src = os.path.join(path, actual_split)
            if not os.path.exists(src):
                continue

            for lbl_path in glob(os.path.join(src, 'labels', '*.txt')):
                base     = os.path.basename(lbl_path).replace('.txt', '')
                img_path = os.path.join(src, 'images', base + '.jpg')
                if not os.path.exists(img_path):
                    img_path = img_path.replace('.jpg', '.png')

                img = cv2.imread(img_path)
                if img is None:
                    continue
                h, w = img.shape[:2]

                with open(lbl_path) as f:
                    for i, line in enumerate(f):
                        parts = line.split()
                        if not parts:
                            continue
                        cls_id = int(parts[0])
                        if   cls_id in c_ids: label = 'cheating'
                        elif cls_id in n_ids: label = 'normal'
                        else:
                            count['skipped'] += 1
                            continue

                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
                        x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)

                        crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        if crop.size == 0:
                            continue
                        crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

                        save_name = f"{ds_prefix}_{split}_{base}_{i}.jpg"
                        cv2.imwrite(os.path.join(output_path, split, label, save_name), crop)
                        count[label] += 1

        print(f"  ✅ {ds_prefix} → cheating: {count['cheating']}, "
              f"normal: {count['normal']}, skipped: {count['skipped']}")


def resplit_dataset(data_dir: str, train_ratio: float = 0.80,
                    valid_ratio: float = 0.10, seed: int = 42):
    test_ratio = 1.0 - train_ratio - valid_ratio
    if test_ratio < 0:
        raise ValueError("train_ratio + valid_ratio phải < 1.0")

    rng     = random.Random(seed)
    classes = set()
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(data_dir, split)
        if os.path.exists(split_path):
            for cls in os.listdir(split_path):
                if os.path.isdir(os.path.join(split_path, cls)):
                    classes.add(cls)

    print(f"\n🔀 Resplit dataset  "
          f"(train={train_ratio:.0%}, valid={valid_ratio:.0%}, "
          f"test={test_ratio:.0%}, seed={seed})")

    for cls in sorted(classes):
        all_imgs = []
        for split in ['train', 'valid', 'test']:
            cls_path = os.path.join(data_dir, split, cls)
            if os.path.exists(cls_path):
                all_imgs.extend(glob(os.path.join(cls_path, '*.jpg')))
                all_imgs.extend(glob(os.path.join(cls_path, '*.png')))

        rng.shuffle(all_imgs)
        n       = len(all_imgs)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)

        splits_map = {
            'train': all_imgs[:n_train],
            'valid': all_imgs[n_train:n_train + n_valid],
            'test':  all_imgs[n_train + n_valid:],
        }

        tmp_dir = os.path.join(data_dir, '_tmp_resplit', cls)
        os.makedirs(tmp_dir, exist_ok=True)
        for img_path in all_imgs:
            shutil.move(img_path, os.path.join(tmp_dir, os.path.basename(img_path)))

        for split, imgs in splits_map.items():
            dst_dir = os.path.join(data_dir, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for img_path in imgs:
                fname = os.path.basename(img_path)
                shutil.move(os.path.join(tmp_dir, fname), os.path.join(dst_dir, fname))

        print(f"  {cls:>10s}: {n:>6d} tổng  "
              f"→  train: {len(splits_map['train'])}, "
              f"valid: {len(splits_map['valid'])}, "
              f"test:  {len(splits_map['test'])}")

    shutil.rmtree(os.path.join(data_dir, '_tmp_resplit'), ignore_errors=True)
    print("✅ Resplit hoàn tất!")


def check_data_distribution(root_path: str):
    print(f"\n{'Split':<10} | {'Class':<10} | {'Count'}")
    print("-" * 35)
    for split in ['train', 'valid', 'test']:
        path = os.path.join(root_path, split)
        if not os.path.exists(path):
            continue
        total = 0
        counts: dict[str, int] = {}
        for cls in sorted(os.listdir(path)):
            cls_path = os.path.join(path, cls)
            if not os.path.isdir(cls_path):
                continue
            cnt = (len(glob(os.path.join(cls_path, '*.jpg'))) +
                   len(glob(os.path.join(cls_path, '*.png'))))
            counts[cls] = cnt
            total += cnt
        for cls, cnt in counts.items():
            print(f"{split:<10} | {cls:<10} | {cnt}")
        print(f"{'TOTAL':<10} | {'':<10} | {total}")
        print("-" * 35)
