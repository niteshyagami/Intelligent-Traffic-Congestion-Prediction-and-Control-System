# Intelligent Traffic Congestion Prediction & Lane-Aware Signal Control System

A research-grade traffic management system using deep learning for congestion prediction and computer vision for adaptive lane-aware signal control.

## Novel Contributions
1. **Lane-level congestion prediction** using Bi-LSTM + Multi-Head Self-Attention
2. **Per-lane green time allocation** — busier lane gets longer green (proportional control)
3. **End-to-end pipeline**: Prediction → Detection → Signal Control
4. **Validated on Indian heterogeneous traffic** data

## Architecture

```
                          ┌──────────────────────────┐
                          │   FastAPI Backend         │
                          │   (localhost:8000)        │
                          └──────┬─────┬─────┬───────┘
                                 │     │     │
              ┌──────────────────┘     │     └──────────────────┐
              ▼                        ▼                        ▼
    ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
    │ Bi-LSTM+Attention│   │  YOLOv8 Detector │   │ Signal Controller│
    │ (Prediction)     │   │  (Lane Counting) │   │ (Per-Lane Green) │
    └──────────────────┘   └──────────────────┘   └──────────────────┘
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 2. Generate Sample Data (or download from Kaggle)
```bash
python data/sample_generator.py
```

### 3. Train the Prediction Model
```bash
python scripts/train_model.py
```

### 4. Start the Server
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 5. Open Dashboard
Visit **http://localhost:8000** in your browser.

## Deploy (Render)

This repo includes a `render.yaml` so you can deploy directly from GitHub.

1. Push this project to a GitHub repository.
2. In Render, create a **Blueprint** and select your repo.
3. Render will use:
   - Build: `pip install -r backend/requirements.txt`
   - Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. After deploy, open your Render URL (the dashboard is served at `/`).

Notes:
- `frontend/index.html` is already configured to call the API using the same host (`window.location.origin`), so no extra frontend env vars are needed.
- Ensure model/data assets are present in the repository (`saved_models/`, `data/indian_smart_traffic.csv`, `backend/yolo11x.pt`), since the app loads them at startup.

## Dashboard Features

| Tab | Feature |
|-----|---------|
| **Prediction** | Select date/time → get lane-level congestion forecast |
| **Signal Control** | Upload image/webcam → YOLOv8 detects vehicles per lane → adaptive signal timing |
| **Analytics** | Model accuracy, architecture info, training history |

## Tech Stack
- **Prediction Model**: PyTorch Bi-LSTM + Multi-Head Self-Attention
- **Vehicle Detection**: YOLOv8n (pretrained on COCO)
- **Signal Control**: Proportional green-time allocation algorithm
- **Backend**: FastAPI (Python)
- **Frontend**: HTML + TailwindCSS + Chart.js
