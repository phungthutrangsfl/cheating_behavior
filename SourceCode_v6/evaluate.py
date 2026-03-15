# -*- coding: utf-8 -*-
"""
v6/evaluate.py — Evaluation dùng YOLO cls .probs API (không cần PyTorch DataLoader).
"""

import json
import os
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve,
)
from tqdm import tqdm

IMG_SIZE = 224


def save_history(history: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'training_history.json')
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✅ Đã lưu training history: {path}")


def plot_label_distribution(data_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    splits = ['train', 'valid', 'test']
    counts: dict = {}
    class_names_all: set = set()

    for split in splits:
        path = os.path.join(data_dir, split)
        if not os.path.exists(path):
            continue
        counts[split] = {}
        for cls in os.listdir(path):
            cls_path = os.path.join(path, cls)
            if not os.path.isdir(cls_path):
                continue
            cnt = (len(glob(os.path.join(cls_path, '*.jpg'))) +
                   len(glob(os.path.join(cls_path, '*.png'))))
            counts[split][cls] = cnt
            class_names_all.add(cls)

    class_names_all = sorted(class_names_all)
    x = np.arange(len(class_names_all)); width = 0.25
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (split, color) in enumerate(zip(splits, colors)):
        if split not in counts:
            continue
        vals = [counts[split].get(cls, 0) for cls in class_names_all]
        bars = ax.bar(x + i * width, vals, width, label=split.capitalize(),
                      color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    str(v), ha='center', va='bottom', fontsize=9)
    ax.set_title('Label Distribution per Split', fontsize=14, fontweight='bold')
    ax.set_xlabel('Class'); ax.set_ylabel('Instances')
    ax.set_xticks(x + width); ax.set_xticklabels(class_names_all, fontsize=12)
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'label_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Đã lưu label distribution: {path}")


def plot_training_curves(history: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    has_train_loss = bool(history.get('train_loss'))
    has_valid_loss = bool(history.get('valid_loss'))
    has_valid_acc  = bool(history.get('valid_acc'))

    epochs = range(1, len(history.get('valid_acc', history.get('train_loss', []))) + 1)

    ncols = sum([has_train_loss or has_valid_loss, has_valid_acc])
    if ncols == 0:
        print("⚠️ Không có dữ liệu lịch sử để vẽ.")
        return

    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    ax_idx = 0
    if has_train_loss or has_valid_loss:
        ax = axes[ax_idx]; ax_idx += 1
        if has_train_loss:
            ax.plot(epochs, history['train_loss'], 'b-o', markersize=4, label='Train Loss')
        if has_valid_loss:
            ax.plot(epochs, history['valid_loss'], 'r-o', markersize=4, label='Valid Loss')
        ax.set_title('Loss Curves', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.legend(); ax.grid(True, alpha=0.3)

    if has_valid_acc:
        ax = axes[ax_idx]
        ax.plot(epochs, history['valid_acc'], 'g-o', markersize=4, label='Valid Acc (Top-1)')
        ax.set_title('Accuracy Curve (YOLO Top-1)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Đã lưu training curves: {path}")


def evaluate_test_set(model, data_dir: str, output_dir: str, batch_size: int = 32):
    os.makedirs(output_dir, exist_ok=True)
    test_path = os.path.join(data_dir, 'test')
    if not os.path.exists(test_path):
        print("⚠️  Không tìm thấy tập test, bỏ qua evaluate.")
        return

    class_names = sorted([
        d for d in os.listdir(test_path)
        if os.path.isdir(os.path.join(test_path, d))
    ])
    if not class_names:
        print("⚠️ Không tìm thấy lớp nào trong tập test.")
        return

    all_preds, all_labels, all_probs_list = [], [], []

    for cls_idx, cls_name in enumerate(class_names):
        cls_dir  = os.path.join(test_path, cls_name)
        img_list = (glob(os.path.join(cls_dir, '*.jpg')) +
                    glob(os.path.join(cls_dir, '*.png')))

        pbar = tqdm(range(0, len(img_list), batch_size),
                    desc=f'  [{cls_name}]', unit='batch', ncols=100)

        for start in pbar:
            batch = img_list[start:start + batch_size]
            results = model(batch, imgsz=IMG_SIZE, verbose=False)
            for r in results:
                probs_tensor = r.probs.data.cpu().numpy()           # shape: (num_classes,)
                pred_idx     = int(r.probs.top1)                    # index of top class
                all_preds.append(pred_idx)
                all_labels.append(cls_idx)
                all_probs_list.append(probs_tensor)

        pbar.close()

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.vstack(all_probs_list)

    _save_confusion_matrix(all_labels, all_preds, class_names, output_dir)
    _save_classification_report(all_labels, all_preds, class_names, output_dir)
    _save_metrics_summary(all_labels, all_preds, class_names, output_dir)
    _plot_confidence_curves(all_labels, all_probs, class_names, output_dir)
    _plot_pr_curves(all_labels, all_probs, class_names, output_dir)
    _plot_roc_curves(all_labels, all_probs, class_names, output_dir)
    _plot_per_class_accuracy(all_labels, all_preds, class_names, output_dir)


def _save_confusion_matrix(all_labels, all_preds, class_names, output_dir):
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11); ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('Actual', fontsize=12)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', fontsize=14,
                    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Đã lưu confusion matrix: {path}")


def _save_classification_report(all_labels, all_preds, class_names, output_dir):
    report_str = classification_report(all_labels, all_preds,
                                       target_names=class_names, digits=4)
    print(f"\n📊 Classification Report:\n{report_str}")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    ax.text(0.05, 0.95, 'Classification Report', fontsize=14, fontweight='bold',
            verticalalignment='top', fontfamily='monospace')
    ax.text(0.05, 0.80, report_str, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=0.8))
    plt.tight_layout()
    path = os.path.join(output_dir, 'classification_report.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Đã lưu classification report: {path}")


def _save_metrics_summary(all_labels, all_preds, class_names, output_dir):
    test_acc    = np.mean(all_preds == all_labels)
    report_dict = classification_report(all_labels, all_preds,
                                        target_names=class_names, digits=4, output_dict=True)
    lines = [f"Test Accuracy:  {test_acc:.4f}", f"Test Samples:   {len(all_labels)}", ""]
    for cls in class_names:
        d = report_dict[cls]
        lines.append(f"{cls:>10s}  →  P: {d['precision']:.4f}  R: {d['recall']:.4f}  "
                     f"F1: {d['f1-score']:.4f}  Support: {d['support']}")
    lines += ["",
              f"{'macro avg':>10s}  →  P: {report_dict['macro avg']['precision']:.4f}  "
              f"R: {report_dict['macro avg']['recall']:.4f}  "
              f"F1: {report_dict['macro avg']['f1-score']:.4f}",
              f"{'weighted avg':>10s}  →  P: {report_dict['weighted avg']['precision']:.4f}  "
              f"R: {report_dict['weighted avg']['recall']:.4f}  "
              f"F1: {report_dict['weighted avg']['f1-score']:.4f}"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis('off')
    ax.text(0.05, 0.95, 'Training & Evaluation Summary', fontsize=14, fontweight='bold',
            verticalalignment='top', fontfamily='monospace')
    ax.text(0.05, 0.80, '\n'.join(lines), fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f5e9', alpha=0.8))
    plt.tight_layout()
    path = os.path.join(output_dir, 'metrics_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Đã lưu metrics summary: {path}")


def _plot_confidence_curves(all_labels, all_probs, class_names, output_dir):
    thresholds = np.linspace(0.0, 1.0, 200)
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(class_names)))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles  = ['F1-Confidence Curve', 'Precision-Confidence Curve', 'Recall-Confidence Curve']
    ylabels = ['F1', 'Precision', 'Recall']

    for cls_idx, (cls_name, color) in enumerate(zip(class_names, colors)):
        y_true  = (all_labels == cls_idx).astype(int)
        y_score = all_probs[:, cls_idx]
        f1s, precs, recs = [], [], []
        for thr in thresholds:
            y_pred = (y_score >= thr).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
            precs.append(p); recs.append(r)
        for ax, vals in zip(axes, [np.array(f1s), np.array(precs), np.array(recs)]):
            ax.plot(thresholds, vals, color=color, label=cls_name, linewidth=1.5)

    macro_f1s = []
    for thr in thresholds:
        fs = []
        for ci in range(len(class_names)):
            yt = (all_labels == ci).astype(int)
            yp = (all_probs[:, ci] >= thr).astype(int)
            tp = np.sum((yp == 1) & (yt == 1)); fp = np.sum((yp == 1) & (yt == 0))
            fn = np.sum((yp == 0) & (yt == 1))
            p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fs.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
        macro_f1s.append(np.mean(fs))
    macro_f1s = np.array(macro_f1s)
    best_thr  = thresholds[np.argmax(macro_f1s)]
    axes[0].plot(thresholds, macro_f1s, 'b-', linewidth=2.5,
                 label=f'all classes {np.max(macro_f1s):.2f} at {best_thr:.3f}')

    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Confidence'); ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'confidence_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Đã lưu confidence curves: {path}")


def _plot_pr_curves(all_labels, all_probs, class_names, output_dir):
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(class_names)))
    fig, ax = plt.subplots(figsize=(8, 6))
    common_rec = np.linspace(0, 1, 500); all_interp = []
    for cls_idx, (cls_name, color) in enumerate(zip(class_names, colors)):
        y_true  = (all_labels == cls_idx).astype(int)
        y_score = all_probs[:, cls_idx]
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ax.plot(rec, prec, color=color, linewidth=1.8,
                label=f'{cls_name}  AUC={auc(rec, prec):.3f}')
        all_interp.append(np.interp(common_rec, rec[::-1], prec[::-1]))
    macro_prec = np.mean(all_interp, axis=0)
    ax.plot(common_rec, macro_prec, 'b-', linewidth=2.5,
            label=f'all classes  {auc(common_rec, macro_prec):.3f} mAP')
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'pr_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Đã lưu PR curve: {path}")


def _plot_roc_curves(all_labels, all_probs, class_names, output_dir):
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(class_names)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls_idx, (cls_name, color) in enumerate(zip(class_names, colors)):
        y_true  = (all_labels == cls_idx).astype(int)
        y_score = all_probs[:, cls_idx]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, color=color, linewidth=1.8,
                label=f'{cls_name}  AUC={auc(fpr, tpr):.3f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Đã lưu ROC curve: {path}")


def _plot_per_class_accuracy(all_labels, all_preds, class_names, output_dir):
    report_dict = classification_report(all_labels, all_preds,
                                        target_names=class_names, digits=4, output_dict=True)
    precs = [report_dict[c]['precision'] for c in class_names]
    recs  = [report_dict[c]['recall']    for c in class_names]
    f1s   = [report_dict[c]['f1-score']  for c in class_names]
    x = np.arange(len(class_names)); width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width, precs, width, label='Precision', color='#2196F3', alpha=0.85)
    b2 = ax.bar(x,         recs,  width, label='Recall',    color='#4CAF50', alpha=0.85)
    b3 = ax.bar(x + width, f1s,   width, label='F1-Score',  color='#FF5722', alpha=0.85)
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    ax.set_xlabel('Class'); ax.set_ylabel('Score')
    ax.set_xticks(x); ax.set_xticklabels(class_names, fontsize=12)
    ax.set_ylim(0, 1.12); ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'per_class_metrics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Đã lưu per-class metrics: {path}")
