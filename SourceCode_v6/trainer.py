# -*- coding: utf-8 -*-
"""
v6/trainer.py — Thin wrapper: re-export train_yolo_classifier từ model.py
để giữ cấu trúc package đồng nhất với các version khác.
"""

from .model import train_yolo_classifier, parse_yolo_history  # noqa: F401

__all__ = ['train_yolo_classifier', 'parse_yolo_history']
