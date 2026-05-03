# 🎯 Multi-Modal Detection System

> Real-time Face Emotion Recognition & Hand Gesture Detection using Computer Vision and Deep Learning.

---

## 📖 Overview

A Python-based computer vision application that performs **simultaneous real-time detection** of:

- 🧑 **Multiple Faces** — with emotion recognition (happy, sad, angry, normal)
- 🖐️ **Hands** — with finger counting and gesture recognition
- 📊 **Data Logging** — all detections are automatically stored in a CSV file for analysis

The system processes a live webcam feed, draws visual overlays on detected features, and logs structured data every second — making it suitable for **Human-Computer Interaction (HCI) research**, **accessibility tools**, and **behavioral analytics**.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Multi-Face Detection** | Detects multiple faces simultaneously using Haar Cascade classifiers |
| **Emotion Recognition** | Classifies emotions into: *happy*, *sad/cry*, *angry*, *normal* using DeepFace AI |
| **Hand Landmark Detection** | Tracks 21 hand landmarks per hand using MediaPipe Tasks API |
| **Finger Counting** | Counts extended fingers in real-time (0–10 for two hands) |
| **Gesture Recognition** | Recognizes 7 gestures: ✌️ Peace, 👍 Thumb Up, 👎 Not OK, 👌 OK, 🤟 I Love You, 🔫 Gun Hand, ✋ Freedom |
| **Heart Gesture** | Special two-hand detection for the ❤️ Heart gesture |
| **Asynchronous Processing** | Emotion analysis runs on a separate thread to keep the video feed smooth |
| **CSV Data Logging** | Logs timestamps, face count, emotions, hand count, fingers, and gestures |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                  DetectionPipeline               │
│                    (app.py)                       │
│                                                   │
│   ┌─────────────┐    ┌──────────────┐            │
│   │ FaceDetector │    │ HandDetector │            │
│   │              │    │              │            │
│   │ • Haar       │    │ • MediaPipe  │            │
│   │   Cascade    │    │   Tasks API  │            │
│   │ • DeepFace   │    │ • Landmark   │            │
│   │   Emotions   │    │   Tracking   │            │
│   │ • Threading  │    │ • Gesture    │            │
│   │              │    │   Engine     │            │
│   └──────┬───────┘    └──────┬───────┘            │
│          │                   │                    │
│          └───────┬───────────┘                    │
│                  ▼                                │
│          ┌──────────────┐                         │
│          │  CSVLogger   │                         │
│          │              │                         │
│          │ • Timestamped│                         │
│          │   Records    │                         │
│          │ • Auto CSV   │                         │
│          └──────────────┘                         │
└─────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Face Recognition/
├── app.py                  # Main entry point — orchestrates the detection pipeline
├── face_detector.py        # Face detection + emotion recognition (Haar + DeepFace)
├── hand_detector.py        # Hand landmark detection + gesture recognition (MediaPipe)
├── csv_logger.py           # CSV logging module for structured data export
├── hand_landmarker.task    # Pre-trained MediaPipe hand landmark model
├── detection_log.csv       # Output log file (auto-generated)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Core programming language |
| **OpenCV** | Video capture, image processing, and rendering |
| **MediaPipe Tasks API** | Hand landmark detection (21 keypoints per hand) |
| **DeepFace** | Deep learning-based facial emotion analysis |
| **Haar Cascade** | Fast face detection for bounding boxes |
| **Threading** | Asynchronous emotion processing for smooth frame rates |

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or higher
- A working webcam

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/face-recognition.git
cd face-recognition

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
python app.py
```

- The webcam feed will open in a new window titled **"Multi-Detection System"**
- Face detections appear as **green rectangles** with emotion labels
- Hand landmarks are drawn with **red dots** and **green connections**
- Hand/finger data is displayed at the **top-left** of the screen
- Press **ESC** to exit

---

## 📊 CSV Output Format

The system logs data every second to `detection_log.csv`:

| Timestamp | Num_Faces | Emotions | Num_Hands | Total_Fingers | Hand_Gestures |
|---|---|---|---|---|---|
| 2026-05-03 01:30:15 | 1 | happy (you are beautiful) | 2 | 7 | Peace, Thumb Up |
| 2026-05-03 01:30:16 | 2 | normal, sad/cry | 1 | 5 | Freedom |

---

## 🎨 Design Decisions

- **Haar Cascade over DNN** — Chosen for fast face localization; DeepFace handles the heavy emotion classification asynchronously.
- **MediaPipe Tasks API** — Used instead of the legacy `mp.solutions.hands` for compatibility with Python 3.12+.
- **IoU Matching** — Emotion regions from DeepFace are matched to Haar Cascade bounding boxes using Intersection over Union (IoU) to correctly label each face.
- **Thread-based Emotion Analysis** — Prevents the video feed from freezing during deep learning inference (~200ms per frame).
- **OOP Architecture** — Each module (Face, Hand, Logger) is a self-contained class, making the system easy to extend and maintain.

---

## 🧩 Extending the System

The modular design makes it easy to add new features:

```python
# Example: Add a new gesture in hand_detector.py
def detect_gesture(self, hand_landmarks):
    states = self.get_fingers_state(hand_landmarks)
    thumb, index, middle, ring, pinky = states

    # Add your custom gesture
    if not thumb and index and not middle and not ring and not pinky:
        return "Pointing"
    
    # ... existing gestures
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Sultan**

---

<p align="center">
  <i>Built with ❤️ using Python, OpenCV, MediaPipe, and DeepFace</i>
</p>
