# 🚦 IntelliTraffic – Traffic Intelligence & Automated Enforcement System

A real-time traffic monitoring system powered by computer vision that detects vehicles, tracks them, estimates speed, analyzes traffic density, identifies violations, performs number plate recognition (ANPR), and automatically generates e-challans with email notifications.

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Video Input │────▶│  YOLOv8      │────▶│  DeepSORT     │
│  (File/Cam)  │     │  Detection   │     │  Tracking     │
└─────────────┘     └──────────────┘     └───────┬───────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────┐
                    │                             │                         │
              ┌─────▼─────┐              ┌───────▼───────┐          ┌─────▼──────┐
              │   Speed    │              │   Density     │          │ Violation  │
              │ Estimation │              │  Analysis     │          │ Detection  │
              └─────┬─────┘              └───────┬───────┘          └─────┬──────┘
                    │                             │                       │
                    │    ┌────────────────────────┐│                       │
                    │    │                        ││    ┌─────────────┐    │
                    └────▶   Risk Engine          ◀┘    │    ANPR     │◀───┘
                         │   (Zone Scoring)       │     │  (EasyOCR)  │
                         └────────┬───────────────┘     └──────┬──────┘
                                  │                            │
                         ┌────────▼────────────────────────────▼──────┐
                         │          Enforcement Engine                 │
                         │  (E-Challan + SQLite + CSV + Email)        │
                         └────────────────────┬───────────────────────┘
                                              │
                                    ┌─────────▼──────────┐
                                    │  Streamlit          │
                                    │  Dashboard          │
                                    └─────────────────────┘
```

---

## 🔧 Prerequisites

- **Python 3.8+** (3.10+ recommended)
- A traffic video file (MP4, AVI, MOV) for testing

---

## 🚀 Installation & Setup

### 1. Clone / Navigate to Project

```bash
cd intelliTraffic
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On first run, YOLOv8 will automatically download the pretrained `yolov8n.pt` model (~6MB). EasyOCR will also download its language models on first use.

### 4. Configure Settings

Edit `config.yaml` to customize:

- **Video source** – file path or camera index
- **Speed limit** – threshold for overspeeding violations
- **ROI points** – region of interest for density counting
- **Perspective points** – calibration for speed estimation
- **Email settings** – SMTP credentials for e-challan notifications

---

## 🎮 Usage

### CLI Mode (main.py)

```bash
# Process a video file
python main.py --source traffic_video.mp4

# Use webcam
python main.py --source 0

# With custom config and output
python main.py --config config.yaml --source video.mp4 --output result.mp4

# Headless mode (no display window)
python main.py --source video.mp4 --no-display
```

### Dashboard Mode (Streamlit)

```bash
streamlit run dashboard/app.py
```

Then open `http://localhost:8501` in your browser. The dashboard provides:
- 📹 Live video feed with detection overlays
- 📊 Real-time speed, density, and risk metrics
- 🚨 Violations log with details
- 📋 E-Challan history table
- 📥 CSV download of all challans
- 🎛️ Interactive controls (speed limit, confidence, heatmap toggle)

---

## ⚙️ Configuration Guide

### Speed Estimation Calibration

The speed estimation uses a **perspective transform** to map pixel coordinates to real-world distances. You need to calibrate the 4 source points to match your camera's view:

```yaml
speed:
  source_points: [[300, 400], [980, 400], [1200, 700], [100, 700]]
  dest_points: [[0, 0], [10, 0], [10, 30], [0, 30]]  # meters
```

**Tip:** Choose 4 points on the road surface that form a rectangle in the real world (e.g., lane markers). Measure the real-world distances between them.

### ROI Configuration

Define the region where vehicles are counted for density analysis:

```yaml
density:
  roi_points: [[100, 300], [1180, 300], [1200, 700], [80, 700]]
```

### Email Setup (Gmail)

1. Enable 2-Step Verification on your Google account
2. Generate an **App Password** at: https://myaccount.google.com/apppasswords
3. Update `config.yaml`:

```yaml
enforcement:
  email:
    enabled: true
    sender_email: "your_email@gmail.com"
    sender_password: "your_app_password"
    recipient_email: "recipient@example.com"
```

---

## 🧠 How It Works

### Module Details

| Module | Technology | Description |
|--------|-----------|-------------|
| **Detection** | YOLOv8n (COCO) | Detects cars, motorcycles, buses, trucks, and persons |
| **Tracking** | DeepSORT | Assigns persistent unique IDs across frames |
| **Speed** | Perspective Transform | Converts pixel displacement to real-world km/h |
| **Density** | ROI Polygon Test | Counts vehicles in zone, classifies LOW/MEDIUM/HIGH |
| **Violations** | Rule Engine | Overspeeding, triple riding (person-motorcycle IoU) |
| **ANPR** | EasyOCR | Reads number plates from vehicle crops |
| **Risk** | Weighted Formula |
| **Enforcement** | SQLite + SMTP | Generates challans, stores logs, sends email alerts |
---

## 📊 Output Files

| File | Description |
|------|-------------|
| `data/logs.csv` | CSV of all challan records |
| `data/challans.db` | SQLite database with full challan history |
| `data/snapshots/` | Violation snapshot images (annotated) |
| `output.mp4` | Processed video with overlays (if --output specified) |

---

## 🔍 Sample Test

1. Download any traffic surveillance video (e.g., from YouTube)
2. Place it in the project directory
3. Update `config.yaml` → `video.source: "your_video.mp4"`
4. Run:

```bash
python main.py --source your_video.mp4 --output result.mp4
```

5. Check `data/logs.csv` for generated challans

---

## 📋 Technical Stack

- **Python 3.8+**
- **OpenCV** – Video processing and image manipulation
- **Ultralytics YOLOv8** – Object detection (pretrained COCO)
- **DeepSORT** (`deep-sort-realtime`) – Multi-object tracking
- **EasyOCR** – Optical character recognition for plates
- **NumPy** – Numerical operations
- **Streamlit** – Interactive web dashboard
- **SQLite** – Lightweight database for challans
- **smtplib** – Email notifications (Python stdlib)

---
