"""
AI-Based Exam Proctoring System (Cheating Detection)
=====================================================
A PyQt6 desktop application that processes video feeds (file or webcam)
through an AI pipeline to detect cheating behaviour during exams.

Pipeline: YOLOv11L (person tracking) → MobileNetV3-Large (cheating classification)
          + temporal smoothing per tracked person

Tech Stack: Python 3, PyQt6, OpenCV, Ultralytics, PyTorch
Threading: QThread + pyqtSignal (keeps GUI responsive)

Author : Wangchinn.t
Date   : 2026-03-17
Version: 1.0.0
"""

import sys
import os
import time
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO

from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QFont, QIcon, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QSplitter,
    QGroupBox,
    QMessageBox,
    QSizePolicy,
    QSpinBox,
    QComboBox,
)

# ──────────────────────────────────────────────────────────────
# Base directory for exam records
# ──────────────────────────────────────────────────────────────
RECORDS_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Records")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ══════════════════════════════════════════════════════════════
#  AI MODEL LOADING
# ══════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")
USE_CUDA = DEVICE.type == "cuda"
if USE_CUDA:
    torch.backends.cudnn.benchmark = True

PIPELINE_YOLO_PLUS_CNN = "yolo_plus_cnn"
PIPELINE_YOLO_ONLY = "yolo_only"
DEFAULT_PIPELINE_MODE = os.getenv("PIPELINE_MODE", PIPELINE_YOLO_PLUS_CNN).strip().lower()
if DEFAULT_PIPELINE_MODE not in {PIPELINE_YOLO_PLUS_CNN, PIPELINE_YOLO_ONLY}:
    DEFAULT_PIPELINE_MODE = PIPELINE_YOLO_PLUS_CNN

# --- YOLO v11 Large ---
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolo11l.pt")
yolo_model = YOLO(YOLO_MODEL_PATH)
print(f"✅ YOLO model loaded: {YOLO_MODEL_PATH}")
YOLO_DEVICE = 0 if USE_CUDA else "cpu"

# Optional YOLO classifier model for YOLO-only pipeline.
YOLO_CLASSIFIER_MODEL_PATH = os.path.join(BASE_DIR, "best_cheating_classifier_v7.pt")
yolo_classifier_model = None
if os.path.exists(YOLO_CLASSIFIER_MODEL_PATH):
    yolo_classifier_model = YOLO(YOLO_CLASSIFIER_MODEL_PATH)
    print(f"✅ YOLO classifier loaded: {YOLO_CLASSIFIER_MODEL_PATH}")
else:
    print(
        f"[INFO] YOLO classifier not found at: {YOLO_CLASSIFIER_MODEL_PATH}"
        " -> YOLO-only mode will fallback to YOLO+CNN."
    )

# --- MobileNetV3 Large (cheating classifier) ---
# LƯU Ý: Nếu bạn train bằng mobilenet_v3_small, đổi dòng dưới thành:
#   cnn_model = models.mobilenet_v3_small(weights=None)
CNN_MODEL_PATH = os.path.join(BASE_DIR, "best_cheating_classifier_v5.pth")

cnn_model = models.mobilenet_v3_large(weights=None)
num_ftrs = cnn_model.classifier[3].in_features
cnn_model.classifier[3] = torch.nn.Linear(num_ftrs, 1)

if os.path.exists(CNN_MODEL_PATH):
    cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE, weights_only=True))
    print(f"✅ CNN model loaded: {CNN_MODEL_PATH}")
else:
    print(f"❌ CNN model NOT FOUND: {CNN_MODEL_PATH}")

cnn_model = cnn_model.to(DEVICE)
cnn_model.eval()

# --- Preprocessing transform (must match training) ---
IMG_SIZE = (128, 128)
inference_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- Temporal smoothing parameters ---
HISTORY_LEN = 50        # Track last 50 CNN evaluations per person
FRAME_THRESHOLD = 0.2   # prob_normal < 0.2 → suspect frame
ALARM_THRESHOLD = 0.6   # >60% suspect frames → CHEATING


def _predict_cheating_prob_yolo(person_crop: np.ndarray) -> float | None:
    """Return cheating probability from YOLO classifier output if available."""
    if yolo_classifier_model is None:
        return None

    results = yolo_classifier_model.predict(
        source=person_crop,
        verbose=False,
        device=YOLO_DEVICE,
        imgsz=224,
        half=USE_CUDA,
    )
    if not results:
        return None

    probs = results[0].probs
    if probs is None:
        return None

    names = results[0].names if hasattr(results[0], "names") else {}
    target_name = "cheating"
    if isinstance(names, dict):
        for idx, class_name in names.items():
            if str(class_name).strip().lower() == target_name:
                return float(probs.data[int(idx)].item())

    top1_idx = int(probs.top1)
    top1_name = str(names.get(top1_idx, "")).lower() if isinstance(names, dict) else ""
    top1_conf = float(probs.top1conf.item())
    if "cheat" in top1_name:
        return top1_conf
    return 1.0 - top1_conf


# ══════════════════════════════════════════════════════════════
#  VIDEO PROCESSING WORKER  (runs on a background QThread)
# ══════════════════════════════════════════════════════════════

class VideoWorker(QThread):
    """
    Reads frames from a video source, runs YOLO tracking on every
    frame, classifies person crops with CNN on key-frames, applies
    temporal smoothing per tracked person, and emits annotated frames.

    Signals
    -------
    frame_ready : QImage
        The annotated frame converted to QImage for display.
    log_message : str
        A message to append to the log panel.
    error_occurred : str
        An error message when something goes wrong.
    finished_signal : (none)
        Emitted when the worker finishes cleanly.
    """

    frame_ready = pyqtSignal(QImage)
    log_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(
        self,
        source,
        save_dir: str,
        frame_skip: int = 5,
        pipeline_mode: str = PIPELINE_YOLO_PLUS_CNN,
        parent=None,
    ):
        """
        Parameters
        ----------
        source : str | int
            Path to a video file, an RTSP URL, or an integer camera index.
        save_dir : str
            Directory where cheating snapshots will be saved.
        frame_skip : int
            Run CNN classification every *frame_skip* frames.
            YOLO tracking runs on EVERY frame (required for persistent IDs).
        """
        super().__init__(parent)
        self.source = source
        self.save_dir = save_dir
        self.frame_skip = max(1, frame_skip)
        self.pipeline_mode = pipeline_mode
        self._running = True

        # Per-track-ID history for temporal smoothing
        self._track_histories: dict[int, deque] = {}
        # Cooldown: last save timestamp per track_id (avoid spamming disk)
        self._last_save_time: dict[int, float] = {}
        self._save_cooldown = 5.0  # seconds between saves per person
        self._prev_frame_ts = time.perf_counter()
        self._fps_ema = 0.0

    # ----- public control -------------------------------------------
    def stop(self):
        """Request the worker loop to stop gracefully."""
        self._running = False

    # ----- main loop ------------------------------------------------
    def run(self):
        """Entry point executed on the background thread."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error_occurred.emit(f"Cannot open video source: {self.source}")
            self.finished_signal.emit()
            return

        os.makedirs(self.save_dir, exist_ok=True)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30.0
        delay = 1.0 / fps

        frame_count = 0
        runtime_mode = self.pipeline_mode
        if runtime_mode == PIPELINE_YOLO_ONLY and yolo_classifier_model is None:
            runtime_mode = PIPELINE_YOLO_PLUS_CNN
            self.log_message.emit(
                "[WARN] YOLO-only mode selected but YOLO classifier model is missing. "
                "Fallback to YOLO+CNN mode."
            )

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            now_ts = time.perf_counter()
            dt = now_ts - self._prev_frame_ts
            self._prev_frame_ts = now_ts
            if dt > 0:
                instant_fps = 1.0 / dt
                if self._fps_ema == 0.0:
                    self._fps_ema = instant_fps
                else:
                    self._fps_ema = (0.9 * self._fps_ema) + (0.1 * instant_fps)

            is_keyframe = (frame_count % self.frame_skip == 0)

            # ── YOLO tracking runs EVERY frame (persist=True needs this) ──
            results = yolo_model.track(
                frame,
                persist=True,
                classes=[0],
                verbose=False,
                device=YOLO_DEVICE,
                half=USE_CUDA,
            )

            frame_copy = frame.copy()

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box

                    # ── CNN classification (key-frames only) ───────
                    if is_keyframe:
                        person_crop = frame[y1:y2, x1:x2]
                        if person_crop.size == 0:
                            continue

                        if runtime_mode == PIPELINE_YOLO_ONLY:
                            cheating_prob = _predict_cheating_prob_yolo(person_crop)
                            if cheating_prob is None:
                                continue
                            is_suspect = 1 if cheating_prob > 0.5 else 0
                        else:
                            pil_img = Image.fromarray(
                                cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                            )
                            input_tensor = inference_transform(pil_img).unsqueeze(0).to(
                                DEVICE,
                                non_blocking=USE_CUDA,
                            )

                            with torch.inference_mode():
                                if USE_CUDA:
                                    with torch.cuda.amp.autocast(dtype=torch.float16):
                                        output = cnn_model(input_tensor)
                                else:
                                    output = cnn_model(input_tensor)
                                prob_normal = torch.sigmoid(output).item()

                            # Update temporal history for this track ID
                            is_suspect = 1 if prob_normal < FRAME_THRESHOLD else 0

                        if track_id not in self._track_histories:
                            self._track_histories[track_id] = deque(
                                maxlen=HISTORY_LEN
                            )
                        self._track_histories[track_id].append(is_suspect)

                    # ── Calculate risk score from history ──────────
                    history = self._track_histories.get(track_id)
                    if history and len(history) > 0:
                        risk_score = sum(history) / len(history)
                    else:
                        risk_score = 0.0

                    is_cheating = risk_score > ALARM_THRESHOLD

                    # ── Draw bounding box + label ──────────────────
                    if is_cheating:
                        color = (0, 0, 255)   # Red (BGR)
                        label = f"ID:{track_id} CHEATING ({risk_score:.0%})"
                    else:
                        color = (0, 255, 0)   # Green (BGR)
                        label = f"ID:{track_id} NORMAL ({risk_score:.0%})"

                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        frame_copy,
                        (x1, y1 - th - 10),
                        (x1 + tw + 4, y1),
                        color,
                        -1,
                    )
                    cv2.putText(
                        frame_copy,
                        label,
                        (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # Risk bar on top of bounding box
                    bar_w = int((x2 - x1) * risk_score)
                    cv2.rectangle(
                        frame_copy,
                        (x1, y1 - 5),
                        (x1 + bar_w, y1),
                        (0, 0, 255),
                        -1,
                    )

                    # ── Save cheating snapshot (with cooldown) ─────
                    if is_cheating:
                        self._save_cheating_frame(
                            frame,
                            track_id,
                            (x1, y1, x2, y2),
                            risk_score,
                        )

            fps_text = f"FPS: {self._fps_ema:.1f}"
            cv2.putText(
                frame_copy,
                fps_text,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # ── Convert BGR → RGB → QImage and emit ────────────────
            rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.frame_ready.emit(qimg.copy())

            frame_count += 1
            time.sleep(delay)

        cap.release()
        self.finished_signal.emit()

    # ----- helpers --------------------------------------------------
    def _save_cheating_frame(
        self,
        frame: np.ndarray,
        track_id: int,
        box: tuple[int, int, int, int],
        risk_score: float,
    ):
        """Save frame with a red cheating box, with per-person cooldown."""
        now_ts = time.time()
        last = self._last_save_time.get(track_id, 0.0)
        if now_ts - last < self._save_cooldown:
            return  # still in cooldown

        self._last_save_time[track_id] = now_ts
        now = datetime.now()
        filename = f"cheating_ID{track_id}_{now.strftime('%H%M%S')}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        try:
            x1, y1, x2, y2 = box
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            save_frame = frame.copy()
            color = (0, 0, 255)
            label = f"ID:{track_id} CHEATING ({risk_score:.0%})"
            cv2.rectangle(save_frame, (x1, y1), (x2, y2), color, 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_y1 = max(0, y1 - th - 10)
            cv2.rectangle(save_frame, (x1, label_y1), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                save_frame,
                label,
                (x1 + 3, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imwrite(filepath, save_frame)
            timestamp = now.strftime("%H:%M:%S")
            self.log_message.emit(
                f"[{timestamp}] ⚠ Cheating detected — ID:{track_id}  Saved: {filename}"
            )
        except Exception as exc:
            self.error_occurred.emit(f"Failed to save frame: {exc}")


# ══════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Exam Proctoring — Cheating Detection")
        self.resize(1200, 700)

        self._worker: VideoWorker | None = None

        self._build_ui()
        self._connect_signals()

    # ── UI construction ────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # ---- Session info panel --------------------------------
        session_group = QGroupBox("Session Info (Thông tin phiên thi)")
        session_layout = QHBoxLayout(session_group)

        subject_label = QLabel("Tên môn thi:")
        self.subject_input = QLineEdit()
        self.subject_input.setPlaceholderText("Nhập tên môn thi (vd: Toán cao cấp)")

        room_label = QLabel("Phòng thi:")
        self.room_input = QLineEdit()
        self.room_input.setPlaceholderText("Nhập phòng thi (vd: P201)")

        output_label = QLabel("Thư mục lưu:")
        self.output_base_input = QLineEdit(RECORDS_BASE_DIR)
        self.output_base_input.setPlaceholderText("Chọn thư mục gốc để lưu kết quả detect")
        self.btn_browse_output = QPushButton("Browse...")

        session_layout.addWidget(subject_label)
        session_layout.addWidget(self.subject_input, stretch=1)
        session_layout.addSpacing(20)
        session_layout.addWidget(room_label)
        session_layout.addWidget(self.room_input, stretch=1)
        session_layout.addSpacing(20)
        session_layout.addWidget(output_label)
        session_layout.addWidget(self.output_base_input, stretch=2)
        session_layout.addWidget(self.btn_browse_output)
        root_layout.addWidget(session_group)

        # ---- Top area: video + log side-by-side ----------------
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Video display
        video_group = QGroupBox("Video Feed")
        video_layout = QVBoxLayout(video_group)
        self.video_label = QLabel("No video source selected.")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 400)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet("background-color: #1e1e1e; color: #aaa; font-size: 16px;")
        video_layout.addWidget(self.video_label)

        # Log panel
        log_group = QGroupBox("Detection Log")
        log_layout = QVBoxLayout(log_group)
        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setFont(QFont("Consolas", 10))
        self.log_panel.setMinimumWidth(300)
        log_layout.addWidget(self.log_panel)

        splitter.addWidget(video_group)
        splitter.addWidget(log_group)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root_layout.addWidget(splitter, stretch=1)

        # ---- Bottom control bar --------------------------------
        ctrl_group = QGroupBox("Controls")
        ctrl_layout = QHBoxLayout(ctrl_group)

        self.btn_open_file = QPushButton("📂  Open Video File")
        self.btn_webcam = QPushButton("📷  Start Webcam")
        self.btn_stop = QPushButton("⏹  Stop")
        self.btn_exit = QPushButton("✖  Exit")
        self.btn_stop.setEnabled(False)

        # Frame-skip spinner
        skip_label = QLabel("CNN every N-th frame:")
        self.spin_skip = QSpinBox()
        self.spin_skip.setRange(1, 60)
        self.spin_skip.setValue(5)
        self.spin_skip.setToolTip(
            "CNN classification runs every N-th frame.\n"
            "YOLO tracking always runs every frame.\n"
            "Higher = faster but less responsive detection."
        )

        pipeline_label = QLabel("Pipeline mode:")
        self.combo_pipeline = QComboBox()
        self.combo_pipeline.addItem("YOLO detect + CNN classify", PIPELINE_YOLO_PLUS_CNN)
        self.combo_pipeline.addItem("YOLO detect + YOLO classify", PIPELINE_YOLO_ONLY)
        default_idx = self.combo_pipeline.findData(DEFAULT_PIPELINE_MODE)
        if default_idx >= 0:
            self.combo_pipeline.setCurrentIndex(default_idx)
        self.combo_pipeline.setToolTip(
            "Choose inference pipeline:\n"
            "- YOLO+CNN: current stable pipeline.\n"
            "- YOLO-only: requires yolo11l_cls.pt in project folder."
        )

        ctrl_layout.addWidget(self.btn_open_file)
        ctrl_layout.addWidget(self.btn_webcam)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(pipeline_label)
        ctrl_layout.addWidget(self.combo_pipeline)
        ctrl_layout.addSpacing(12)
        ctrl_layout.addWidget(skip_label)
        ctrl_layout.addWidget(self.spin_skip)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.btn_exit)
        root_layout.addWidget(ctrl_group)

        # ---- Styling -------------------------------------------
        self.setStyleSheet("""
            QMgjugthainWindow { background: #2b2b2b; }
            QGroupBox {
                color: #ddd;
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 14px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }
            QPushButton {
                background: #3c3f41;
                color: #ddd;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 13px;
            }
            QPushButton:hover { background: #505356; }
            QPushButton:disabled { color: #666; }
            QTextEdit {
                background: #1e1e1e;
                color: #c8e6c9;
                border: none;
            }
            QSpinBox {
                background: #3c3f41;
                color: #ddd;
                border: 1px solid #555;
                padding: 4px;
            }
            QLabel { color: #ccc; }
            QLineEdit {
                background: #3c3f41;
                color: #ddd;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
                font-size: 13px;
            }
            QLineEdit:focus { border: 1px solid #6ea1f1; }
        """)

    # ── Signal wiring ──────────────────────────────────────────
    def _connect_signals(self):
        self.btn_open_file.clicked.connect(self._on_open_file)
        self.btn_webcam.clicked.connect(self._on_start_webcam)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_exit.clicked.connect(self.close)
        self.btn_browse_output.clicked.connect(self._on_browse_output_dir)

    # ── Slots / handlers ───────────────────────────────────────
    def _validate_session(self) -> bool:
        """Return True if both session fields are filled, else warn."""
        if not self.subject_input.text().strip() or not self.room_input.text().strip():
            QMessageBox.warning(
                self, "Thiếu thông tin",
                "Vui lòng nhập Tên môn thi và Phòng thi!"
            )
            return False
        return True

    def _build_save_dir(self) -> str:
        """Construct Records/{date}/{subject}/{room}/ path."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        subject = self.subject_input.text().strip()
        room = self.room_input.text().strip()
        base_dir = self.output_base_input.text().strip() or RECORDS_BASE_DIR
        save_dir = os.path.join(base_dir, date_str, subject, room)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    @pyqtSlot()
    def _on_browse_output_dir(self):
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Chọn thư mục lưu kết quả detect",
            self.output_base_input.text().strip() or RECORDS_BASE_DIR,
        )
        if selected_dir:
            self.output_base_input.setText(selected_dir)

    @pyqtSlot()
    def _on_open_file(self):
        if not self._validate_session():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv);;All Files (*)"
        )
        if path:
            self._start_worker(path)

    @pyqtSlot()
    def _on_start_webcam(self):
        if not self._validate_session():
            return
        # Use 0 for default webcam.
        # Replace with an RTSP URL string for IP cameras, e.g.:
        #   self._start_worker("rtsp://user:pass@192.168.1.100:554/stream")
        self._start_worker(0)

    @pyqtSlot()
    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(5000)  # wait up to 5 s for graceful exit
        self._set_controls_running(False)
        self.video_label.clear()
        self.video_label.setText("Stopped.")

    # ── Worker lifecycle ───────────────────────────────────────
    def _start_worker(self, source):
        """Stop any existing worker, then launch a new one."""
        self._on_stop()

        save_dir = self._build_save_dir()
        selected_mode = self.combo_pipeline.currentData()
        self._worker = VideoWorker(
            source,
            save_dir=save_dir,
            frame_skip=self.spin_skip.value(),
            pipeline_mode=selected_mode,
        )
        self._worker.frame_ready.connect(self._update_frame)
        self._worker.log_message.connect(self._append_log)
        self._worker.error_occurred.connect(self._show_error)
        self._worker.finished_signal.connect(self._on_worker_finished)
        self._worker.start()

        self._set_controls_running(True)
        src_name = source if isinstance(source, str) else "Webcam"
        self._append_log(f"[{datetime.now():%H:%M:%S}] Started: {src_name}")
        self._append_log(f"  ↳ Saving to: {save_dir}")
        self._append_log(f"  ↳ Device: {DEVICE}")
        self._append_log(f"  ↳ Pipeline: {selected_mode}")

    @pyqtSlot(QImage)
    def _update_frame(self, qimg: QImage):
        """Scale the frame to fit the label while keeping aspect ratio."""
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    @pyqtSlot(str)
    def _append_log(self, msg: str):
        self.log_panel.append(msg)
        # Auto-scroll to the bottom
        self.log_panel.verticalScrollBar().setValue(
            self.log_panel.verticalScrollBar().maximum()
        )

    @pyqtSlot(str)
    def _show_error(self, msg: str):
        self._append_log(f"[ERROR] {msg}")
        QMessageBox.warning(self, "Error", msg)

    @pyqtSlot()
    def _on_worker_finished(self):
        self._set_controls_running(False)
        self._append_log(f"[{datetime.now():%H:%M:%S}] Video source ended.")

    # ── Helpers ────────────────────────────────────────────────
    def _set_controls_running(self, running: bool):
        self.btn_open_file.setEnabled(not running)
        self.btn_webcam.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.combo_pipeline.setEnabled(not running)
        self.spin_skip.setEnabled(not running)
        self.subject_input.setEnabled(not running)
        self.room_input.setEnabled(not running)
        self.output_base_input.setEnabled(not running)
        self.btn_browse_output.setEnabled(not running)

    def closeEvent(self, event):
        """Ensure worker is stopped before the window closes."""
        self._on_stop()
        event.accept()


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # consistent cross-platform look
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
