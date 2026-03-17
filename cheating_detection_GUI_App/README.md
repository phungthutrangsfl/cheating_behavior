# 🎓 AI-Based Exam Proctoring System (Cheating Detection)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Status](https://img.shields.io/badge/status-Active-brightgreen)

Ứng dụng desktop **PyQt6** sử dụng **AI (YOLO11L + MobileNetV3)** để phát hiện hành vi gian lận trong kỳ thi trực tuyến và tại phòng dự thi.

## 📋 Mục Đích & Tính Năng

### Tính năng chính:
- ✅ **Tracking người** real-time từ video/webcam bằng YOLOv11L
- ✅ **Phô phát hiện gian lận** (nhìn chéo, dùng điện thoại, chuyển động đáng ngờ...)
- ✅ **Giao diện GUI** thân thiện, dễ sử dụng
- ✅ **Ghi lại video** kèm kết quả phân tích
- ✅ **Temporal smoothing** để giảm false positive
- ✅ **Hỗ trợ GPU** (CUDA) để tăng tốc độ

### Hành vi phát hiện được:
- 🔴 Nhìn chéo/quay đầu bất thường
- 🔴 Dùng điện thoại/thiết bị khác
- 🔴 Che chắn mặt hoặc khuôn hình
- 🔴 Rời khỏi vùng thi

## 🏗️ Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────┐
│  Input Layer                                            │
│  ├─ Video File (MP4, AVI, MKV...)                      │
│  └─ Webcam/Camera                                       │
└────────────┬────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────┐
│  YOLO11L Model                                          │
│  ├─ Person Detection                                    │
│  └─ Bounding Box Tracking                               │
└────────────┬────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────┐
│  MobileNetV3-Large Classification                       │
│  ├─ Extract Face/Pose Features                          │
│  └─ Classify: Normal / Cheating                         │
└────────────┬────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────┐
│  Post-Processing                                        │
│  ├─ Temporal Smoothing (per-person)                    │
│  ├─ Confidence Filtering                                │
│  └─ Alert Generation                                    │
└────────────┬────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────┐
│  Output & Storage                                       │
│  ├─ PyQt6 GUI Visualization                             │
│  ├─ Video Recording with Annotations                    │
│  └─ Analysis Report                                     │
└─────────────────────────────────────────────────────────┘
```

### Tech Stack:
| Công nghệ | Phiên bản | Mục đích |
|-----------|----------|---------|
| PyQt6 | 6.6.1 | GUI Framework |
| PyTorch | 2.0.1 | Deep Learning Framework |
| Ultralytics | 8.0.195 | YOLO Model |
| OpenCV | 4.8.1 | Video Processing |
| TorchVision | 0.15.2 | Pre-trained Models |

## 📦 Yêu Cầu Hệ Thống

### Phần cứng (Recommended):
| Thành phần | Tối thiểu | Khuyến nghị |
|-----------|----------|-----------|
| **CPU** | i3-8100 / Ryzen 5 1600 | i5-10400 / Ryzen 7 3700X |
| **GPU** | không | RTX 3060 / 4060 (VRAM 8GB+) |
| **RAM** | 8GB | 16GB |
| **Storage** | 2GB | 5GB+ |
| **OS** | Windows 7+ / Linux | Windows 10/11 / Linux |

> **Note**: GPU giúp tăng FPS từ 5-10 FPS (CPU) lên 20-30 FPS (GPU)

### Phần mềm:
- Python 3.8+
- _(Optional)_ CUDA 11.8+ (GPU acceleration)
- _(Optional)_ cuDNN (NVIDIA GPU support)

## 🚀 Cài Đặt & Chạy

### 1. Clone Repository
```bash
git clone https://github.com/wangchinnt/cheating-detection-gui.git
cd cheating-detection-gui
```

### 2. Tạo Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài đặt Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Models
Khi clone repo, bạn cần đặt 3 model sau vào thư mục gốc dự án:
- `yolo11l.pt` - YOLO detect/tracking model
- `best_cheating_classifier_v5.pth` - model cho mode YOLO + CNN
- `best_cheating_classifier_v7.pt` - model cho mode YOLO only

`yolo11l.pt` có thể tải tự động từ Ultralytics nếu chưa có file.

`best_cheating_classifier_v5.pth` và `best_cheating_classifier_v7.pt` nên được phát hành qua GitHub Releases của repo để người clone tải trực tiếp.

Trang tải model đề xuất:
- [Model v5 (.pth)](https://drive.google.com/file/d/1Yq7gvnbXpwVdyKnhOmJFeue6Jmg1g8Ki/view?usp=sharing)
- [Model v7 (.pt)](https://drive.google.com/file/d/1T7qks2rSxrBhH-JF4kSkEygSG5kQ8o_y/view?usp=sharing)
- [GitHub Releases](https://github.com/wangchinnt/cheating-detection-gui/releases)

Ví dụ tải thủ công (PowerShell):
```powershell
Invoke-WebRequest -Uri "<V5_MODEL_DIRECT_DOWNLOAD_URL>" -OutFile "best_cheating_classifier_v5.pth"
Invoke-WebRequest -Uri "<V7_MODEL_DIRECT_DOWNLOAD_URL>" -OutFile "best_cheating_classifier_v7.pt"
```

Sau khi tải xong, cấu trúc thư mục cần có:
```text
cheating-detection-gui/
├── main.py
├── yolo11l.pt
├── best_cheating_classifier_v5.pth
└── best_cheating_classifier_v7.pt
```

## 📖 Hướng dẫn Sử dụng

### Chạy ứng dụng
```bash
python main.py
```

### Giao diện chính:
1. **Video Source Selection**: Chọn nguồn video (File hoặc Webcam)
2. **Mode Selection**: Chọn chế độ xử lý
   - **YOLO + CNN**: Kết hợp phát hiện + phân loại (chính xác & đủ nhanh)
   - **YOLO Only**: Chỉ phát hiện người (nhanh nhất)

3. **Điều chỉnh tham số**:
   - **Confidence Threshold**: Độ tin cây phát hiện (0.3-0.9)
   - **Smoothing Window**: Cửa sổ làm mượt kết quả (3-15 frames)
   - **Save Output**: Lưu video kết quả

4. **Recording**:
   - Tất cả kết quả sẽ được lưu vào thư mục `Records/`
   - Định dạng: `exam_[TIMESTAMP]_[STATUS].mp4`

### Ví dụ Output:
```
Records/
├── exam_2026-03-17_14-30-45_suspicious.mp4
├── exam_2026-03-17_14-32-10_clean.mp4
└── exam_2026-03-17_15-00-00_flagged.mp4
```

## ⚙️ Cấu hình Nâng cao

### Biến môi trường:
```bash
# Chọn pipeline (mặc định: yolo_plus_cnn)
export PIPELINE_MODE=yolo_plus_cnn  # hoặc yolo_only

# Chạy
python main.py
```

### Tuneables trong code:
- **YOLO Confidence**: Điều chỉnh tại `yolo_model.predict(..., conf=...)`
- **CNN Threshold**: Thay đổi ngưỡng phân loại gian lận
- **Smoothing**: Tăng `deque` size để kết quả mịn hơn

## 📊 Hiệu suất

| Chế độ | FPS | Accuracy | GPU VRAM |
|-------|-----|----------|----------|
| YOLO Only | 25-30 | ~85% | 2GB |
| YOLO + CNN | 12-18 | ~92% | 4GB |

*Kết quả trên RTX 3060 với video 1080p*

## 🔧 Troubleshooting

### Vấn đề 1: CUDA không tìm thấy
```bash
# Kiểm tra
python -c "import torch; print(torch.cuda.is_available())"

# Cài lại PyTorch cho CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Vấn đề 2: GUI bị đứng/lag
- Giảm độ phân giải video input
- Tăng Smoothing Window
- Chuyển sang YOLO-only mode

### Vấn đề 3: Out of Memory (CUDA)
- Giảm batch size
- Dùng CPU (`DEVICE = torch.device("cpu")`)
- Giảm độ phân giải video

### Vấn đề 4: Model không load
```bash
# Kiểm tra file tồn tại
ls *.pth *.pt

# Download lại từ source
# YOLOv11: ultralytics sẽ tự download
# Classifier: train lại hoặc lấy từ backup
```

## 📌 File cấu trúc

```
cheating-detection-gui/
├── main.py                              # Điểm vào chính & GUI
├── requirements.txt                     # Python dependencies
├── .gitignore                          # Git ignore rules
├── README.md                           # Tệp này
│
├── yolo11l.pt                          # YOLOv11 Model (auto-download)
├── best_cheating_classifier_v5.pth    # CNN Classifier
│
├── Records/                            # Exam recordings output
│   └── exam_TIMESTAMP_STATUS.mp4
│
├── build/                              # PyInstaller build output
└── dist/                               # PyInstaller executables
```

## 📝 Training Model Tùy chỉnh

Để huấn luyện lại classifier với dữ liệu của riêng bạn:

```python
# pseudocode
from torch.utils.data import DataLoader
import torch.optim as optim

# 1. Chuẩn bị dataset (cheating vs clean frames)
train_loader = DataLoader(your_dataset, batch_size=32)

# 2. Định nghĩa model
model = models.mobilenet_v3_large(weights='DEFAULT')
num_ftrs = model.classifier[3].in_features
model.classifier[3] = Linear(num_ftrs, 1)

# 3. Training loop
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = BCEWithLogitsLoss()

for epoch in range(num_epochs):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 4. Lưu
torch.save(model.state_dict(), 'best_cheating_classifier.pth')
```

## 🤝 Đóng góp

Dự án dành cho mục đích giáo dục & nghiên cứu. Để đóng góp:
1. Fork repo
2. Tạo branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Tạo Pull Request

## ⚖️ Quy định & Pháp lý

- **Đạo đức**: Chỉ sử dụng cho giám sát hợp pháp, có sự đồng ý của thí sinh
- **Bảo mật**: Không lưu trữ/chia sẻ dữ liệu sinh trắc học (khuôn mặt)
- **Accuracy**: Mô hình không 100% chính xác - luôn yêu cầu xem xét thủ công

## 📄 Giấy phép

MIT License - Xem [LICENSE](LICENSE) file



---

## 🗺️ Roadmap

- [ ] API server (REST/WebSocket)
- [ ] Web dashboard cho quản lý tập trung
- [ ] Multi-camera support
- [ ] Real-time alerts & notifications
- [ ] Database integration (PostgreSQL)
- [ ] Mobile app (chấm điểm từ điện thoại)

---

## ❓ FAQ

**Q: Có thể chạy trên CPU không?**  
A: Có, nhưng chậm hơn (2-3 FPS). GPU khuyến nghị cho thực tế.

**Q: Có chế độ offline không?**  
A: Có, tất cả xử lý là local - không cần internet sau khi download model.

**Q: Lưu kết quả ở đâu?**  
A: Thư mục `Records/` - có thể thay đổi trong code (biến `RECORDS_BASE_DIR`).

**Q: Có thể deploy trên server không?**  
A: Hiện tại là desktop app. Đang phát triển API server version.

---

**Last Updated**: 2026-03-17  
**Version**: 1.0.0 - Release Candidate
