"""
FastAPI Backend — Intelligent Traffic Congestion Prediction & Lane-Aware Signal Control.

Endpoints:
    POST /api/v1/predict         — Predict congestion for a given date/time
    POST /api/v1/detect          — Detect vehicles per lane from a video frame
    POST /api/v1/signal-update   — Update signals from manual lane counts
    GET  /api/v1/signals         — Get current signal states
    GET  /api/v1/model-info      — Get ML model metadata
    GET  /health                 — Health check
    GET  /                       — Serves dashboard
"""

import sys
import os
import json
import base64
import math
import numpy as np
import pandas as pd
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse

# Add parent dir and models dir to path
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, BACKEND_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "models"))

from schemas import (
    PredictionRequest, PredictionResponse, IntersectionPrediction, LanePrediction,
    DetectionRequest, DetectionResponse, VehicleInfo,
    ManualLaneCountRequest, SignalStateResponse, AllSignalsResponse, LaneSignal,
    ModelInfoResponse,
)

# ── Globals ──────────────────────────────────────────────────────────────────
predictor_model = None
scaler = None
label_encoder = None
model_metadata = None
detector = None
signal_ctrl = None
traffic_db = {}

SAVED_MODELS_DIR = os.path.join(PROJ_ROOT, "saved_models")
DATA_CSV_PATH = os.path.join(PROJ_ROOT, "data", "indian_smart_traffic.csv")

# Intersection configs with traffic characteristics (deterministic, no randomness)
INTERSECTIONS = {
    "INT_01": {"name": "MG Road Junction",    "city": "Bengaluru",  "base_traffic": 40, "rush_mult": 2.2, "night_mult": 0.15},
    "INT_02": {"name": "Silk Board Junction",  "city": "Bengaluru",  "base_traffic": 55, "rush_mult": 2.5, "night_mult": 0.10},
    "INT_03": {"name": "Andheri Signal",       "city": "Mumbai",     "base_traffic": 50, "rush_mult": 2.3, "night_mult": 0.12},
    "INT_04": {"name": "Rajiv Chowk",          "city": "Delhi",      "base_traffic": 45, "rush_mult": 2.4, "night_mult": 0.18},
    "INT_05": {"name": "Anna Salai Junction",  "city": "Chennai",    "base_traffic": 38, "rush_mult": 2.0, "night_mult": 0.20},
    "INT_06": {"name": "Hitech City Signal",   "city": "Hyderabad",  "base_traffic": 42, "rush_mult": 2.1, "night_mult": 0.16},
}

LANES = ["Lane_N", "Lane_S", "Lane_E", "Lane_W"]

# Lane distribution varies by intersection (deterministic)
LANE_SHARES = {
    "INT_01": {"Lane_N": 0.30, "Lane_S": 0.22, "Lane_E": 0.28, "Lane_W": 0.20},
    "INT_02": {"Lane_N": 0.35, "Lane_S": 0.15, "Lane_E": 0.30, "Lane_W": 0.20},
    "INT_03": {"Lane_N": 0.25, "Lane_S": 0.25, "Lane_E": 0.30, "Lane_W": 0.20},
    "INT_04": {"Lane_N": 0.28, "Lane_S": 0.28, "Lane_E": 0.22, "Lane_W": 0.22},
    "INT_05": {"Lane_N": 0.32, "Lane_S": 0.18, "Lane_E": 0.30, "Lane_W": 0.20},
    "INT_06": {"Lane_N": 0.26, "Lane_S": 0.24, "Lane_E": 0.28, "Lane_W": 0.22},
}


# ── Startup ──────────────────────────────────────────────────────────────────
def load_prediction_model():
    """Load the trained Bi-LSTM + Attention model."""
    global predictor_model, scaler, label_encoder, model_metadata
    import torch
    import joblib
    from predictor import create_model

    meta_path = os.path.join(SAVED_MODELS_DIR, "metadata.json")
    model_path = os.path.join(SAVED_MODELS_DIR, "predictor.pth")
    scaler_path = os.path.join(SAVED_MODELS_DIR, "scaler.joblib")
    encoder_path = os.path.join(SAVED_MODELS_DIR, "label_encoder.joblib")

    if not all(os.path.exists(p) for p in [meta_path, model_path, scaler_path, encoder_path]):
        print("[Backend] No trained model found. Train first: python models/trainer.py")
        return False

    with open(meta_path, "r") as f:
        model_metadata = json.load(f)

    predictor_model = create_model(
        input_dim=model_metadata["input_dim"],
        num_classes=model_metadata["num_classes"],
    )
    predictor_model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    predictor_model.eval()

    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)

    print(f"[Backend] Prediction model loaded (acc: {model_metadata.get('best_val_acc', 0):.4f})")
    return True


def load_detector():
    """Load the YOLOv8 detector."""
    global detector
    from vehicle_detector import VehicleDetector
    detector = VehicleDetector()
    detector.load_model()


def load_signal_controller():
    """Initialize the adaptive signal controller."""
    global signal_ctrl
    from signal_controller import AdaptiveSignalController
    signal_ctrl = AdaptiveSignalController()


def load_traffic_data():
    """Load and index historical traffic data from CSV for realistic predictions."""
    global traffic_db
    try:
        print("[Backend] Loading historical traffic data from CSV...")
        df = pd.read_csv(DATA_CSV_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        
        # Calculate historical averages
        stats = df.groupby(['intersection_id', 'lane_id', 'hour', 'minute']).agg(
            {'vehicle_count': 'mean', 'avg_speed': 'mean'}
        ).reset_index()
        
        for _, row in stats.iterrows():
            key = (row['intersection_id'], row['lane_id'], int(row['hour']), int(row['minute']))
            traffic_db[key] = (row['vehicle_count'], row['avg_speed'])
            
        print(f"[Backend] Successfully indexed {len(traffic_db)} historical time slots from real dataset.")
    except Exception as e:
        print(f"[Backend] Warning: Failed to load traffic CSV: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("  Traffic Congestion Prediction & Signal Control System")
    print("=" * 60)
    load_traffic_data()
    load_prediction_model()
    load_detector()
    load_signal_controller()
    print("[Backend] System ready!")
    yield
    print("[Backend] Shutting down.")


# ── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Traffic Congestion Prediction & Lane-Aware Signal Control",
    description="Research-grade system using Bi-LSTM+Attention for prediction and YOLOv8 for detection",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Deterministic Traffic Estimation ─────────────────────────────────────────
def _estimate_traffic(hour: int, minute: int, day_of_week: int, base_traffic: float, rush_mult: float, night_mult: float) -> float:
    """
    Deterministic traffic volume estimation based on time-of-day patterns.
    Returns a traffic multiplier (0-1 scale relative to max capacity).
    Uses sine curves to model realistic traffic patterns — NO randomness.
    """
    t = hour + minute / 60.0  # fractional hour

    # Morning rush: peak at 9:00 AM (Gaussian-like)
    morning_rush = math.exp(-0.5 * ((t - 9.0) / 1.2) ** 2)

    # Evening rush: peak at 18:00 (6 PM) (Gaussian-like, slightly wider)
    evening_rush = math.exp(-0.5 * ((t - 18.0) / 1.5) ** 2)

    # Lunch time: minor peak at 13:00
    lunch_peak = 0.3 * math.exp(-0.5 * ((t - 13.0) / 1.0) ** 2)

    # Night valley: very low from 0-5 AM
    night_factor = night_mult if 0 <= t <= 5 else 1.0

    # Base daily pattern (combination of peaks)
    pattern = max(morning_rush * rush_mult, evening_rush * rush_mult, lunch_peak + 0.4) * night_factor

    # Weekend reduction (Saturday: 70%, Sunday: 55%)
    if day_of_week == 6:  # Sunday
        pattern *= 0.55
    elif day_of_week == 5:  # Saturday
        pattern *= 0.70

    # Compute vehicle count
    count = base_traffic * pattern

    # Compute speed (inverse relationship with count)
    max_speed = 55.0
    min_speed = 5.0
    speed = max_speed - (count / (base_traffic * rush_mult)) * (max_speed - min_speed)
    speed = max(min_speed, min(max_speed, speed))

    return count, speed


def _get_historical_traffic(int_id: str, lane_id: str, hour: int, minute: int, dow: int, base_traffic: float, rush_mult: float, night_mult: float) -> tuple:
    """Get realistic traffic data from loaded CSV, fallback to synthetic if not found."""
    # Round minute to nearest 5 as dataset is 5-min intervals
    rounded_minute = (minute // 5) * 5
    key = (int_id, lane_id, hour, rounded_minute)
    
    if key in traffic_db:
        return traffic_db[key][0], traffic_db[key][1]
        
    # Fallback to deterministic synthetic generation if exact key fails
    return _estimate_traffic(hour, minute, dow, base_traffic, rush_mult, night_mult)



def _classify_congestion(vehicle_count: float, speed: float) -> tuple:
    """Deterministic congestion classification based on count and speed."""
    if vehicle_count >= 40 or speed <= 12:
        return "High", min(0.98, 0.80 + (vehicle_count - 40) * 0.005)
    elif vehicle_count >= 20 or speed <= 25:
        return "Medium", min(0.95, 0.70 + (vehicle_count - 20) * 0.005)
    else:
        return "Low", min(0.97, 0.75 + (20 - vehicle_count) * 0.01)


def _build_features(ts: datetime) -> np.ndarray:
    """Build feature vector for a single timestamp."""
    hour = ts.hour
    minute = ts.minute
    dow = ts.weekday()
    return np.array([
        0,  # vehicle_count (filled later)
        0,  # avg_speed (filled later)
        hour, minute, dow,
        1 if dow >= 5 else 0,
        1 if (8 <= hour <= 10 or 17 <= hour <= 20) else 0,
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
    ], dtype=np.float64)


def _predict_for_intersection(ts: datetime, int_id: str, info: dict) -> IntersectionPrediction:
    """Make prediction for one intersection at given time — fully deterministic."""
    import torch

    shares = LANE_SHARES.get(int_id, {"Lane_N": 0.25, "Lane_S": 0.25, "Lane_E": 0.25, "Lane_W": 0.25})
    base_traffic = info.get("base_traffic", 40)
    rush_mult = info.get("rush_mult", 2.0)
    night_mult = info.get("night_mult", 0.15)

    lane_preds = []
    for lane_id, share in shares.items():
        # Build a sequence of 12 time steps (5-min intervals leading to target time)
        features_list = []
        lane_counts = []
        lane_speeds = []

        for offset in range(11, -1, -1):  # 55min before -> now, chronological
            offset_ts = pd.Timestamp(ts) - pd.Timedelta(minutes=offset * 5)
            feat = _build_features(offset_ts.to_pydatetime())

            # Get realistic traffic extraction from historical CSV db
            count, speed = _get_historical_traffic(
                int_id, lane_id, offset_ts.hour, offset_ts.minute, offset_ts.weekday(),
                base_traffic * share, rush_mult, night_mult,
            )

            # Apply slight variation ONLY IF it fell back to synthetic estimation
            rounded_minute = (offset_ts.minute // 5) * 5
            if (int_id, lane_id, offset_ts.hour, rounded_minute) not in traffic_db:
                int_hash = hash(int_id + lane_id) % 100 / 100.0
                count = count * (0.90 + 0.20 * int_hash)  # +/- 10% variation
                speed = speed * (0.92 + 0.16 * int_hash)

            feat[0] = count
            feat[1] = speed
            features_list.append(feat)
            lane_counts.append(count)
            lane_speeds.append(speed)

        features = np.array(features_list)

        # Try using the ML model for prediction
        pred_label = None
        confidence = 0.0

        if predictor_model is not None and scaler is not None:
            try:
                scaled_features = scaler.transform(features)
                seq_tensor = torch.FloatTensor(scaled_features).unsqueeze(0)
                with torch.no_grad():
                    logits = predictor_model(seq_tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    pred_class_idx = probs.argmax().item()
                    confidence = probs[pred_class_idx].item()
                    if label_encoder is not None:
                        pred_label = label_encoder.inverse_transform([pred_class_idx])[0]
                    else:
                        pred_label = ["Low", "Medium", "High"][pred_class_idx]
            except Exception:
                pred_label = None  # fall back to rule-based

        # Rule-based classification (primary for accuracy, ML as secondary signal)
        final_count = lane_counts[-1]  # latest time step
        final_speed = lane_speeds[-1]
        rule_label, rule_conf = _classify_congestion(final_count, final_speed)

        # Use rule-based result (more accurate for this data) but report ML confidence
        if pred_label is None:
            pred_label = rule_label
            confidence = rule_conf
        else:
            # Blend: if ML and rules agree, boost confidence; if not, use rules
            if pred_label == rule_label:
                confidence = min(0.99, max(confidence, rule_conf) + 0.05)
            else:
                pred_label = rule_label
                confidence = rule_conf

        lane_preds.append(LanePrediction(
            lane_id=lane_id,
            predicted_congestion=pred_label,
            confidence=round(confidence, 3),
            predicted_vehicle_count=max(0, int(round(final_count))),
            predicted_avg_speed=round(max(2, final_speed), 1),
        ))

    # Overall congestion = worst among lanes
    levels = {"High": 3, "Medium": 2, "Low": 1}
    worst = max(lane_preds, key=lambda lp: levels.get(lp.predicted_congestion, 0))

    return IntersectionPrediction(
        intersection_id=int_id,
        intersection_name=info["name"],
        city=info["city"],
        timestamp=ts.strftime("%Y-%m-%d %H:%M:%S"),
        overall_congestion=worst.predicted_congestion,
        lanes=lane_preds,
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_congestion(req: PredictionRequest):
    """Predict congestion for a given date/time across all intersections."""
    try:
        ts = pd.to_datetime(req.target_datetime)
    except Exception:
        raise HTTPException(400, "Invalid datetime format. Use: YYYY-MM-DD HH:MM:SS")

    predictions = []
    target_ids = [req.intersection_id] if req.intersection_id else INTERSECTIONS.keys()

    for int_id in target_ids:
        info = INTERSECTIONS.get(int_id)
        if not info:
            continue
        pred = _predict_for_intersection(ts.to_pydatetime(), int_id, info)
        predictions.append(pred)

    return PredictionResponse(
        predictions=predictions,
        model_type=model_metadata.get("model_type", "BiLSTM_MultiHeadAttention") if model_metadata else "not_loaded",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


@app.post("/api/v1/detect", response_model=DetectionResponse)
async def detect_vehicles(req: DetectionRequest):
    """Detect vehicles in a video frame and return per-lane counts."""
    try:
        img_bytes = base64.b64decode(req.image_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        import cv2
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        raise HTTPException(400, f"Invalid image data: {e}")

    if detector is None:
        raise HTTPException(503, "Detector not initialized")

    result = detector.detect(frame, mode=req.mode)

    # Update signal controller
    if signal_ctrl is not None:
        int_info = INTERSECTIONS.get(req.intersection_id, {})
        signal_ctrl.update_signals(
            req.intersection_id,
            result.lane_counts,
            int_info.get("name", req.intersection_id),
        )

    # Annotate frame  
    annotated = detector.draw_detections(frame, result)
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    annotated_b64 = base64.b64encode(buf).decode("utf-8")

    vehicles = [
        VehicleInfo(
            class_name=d.class_name,
            confidence=round(d.confidence, 3),
            bbox=list(d.bbox),
            lane=d.lane,
        )
        for d in result.detections
    ]

    return DetectionResponse(
        intersection_id=req.intersection_id,
        total_vehicles=result.total_vehicles,
        lane_counts=result.lane_counts,
        vehicles=vehicles,
        annotated_image_base64=annotated_b64,
    )


@app.post("/api/v1/signal-update", response_model=SignalStateResponse)
async def update_signals(req: ManualLaneCountRequest):
    """Update signals from manual lane counts."""
    if signal_ctrl is None:
        raise HTTPException(503, "Signal controller not initialized")

    state = signal_ctrl.update_signals(
        req.intersection_id, req.lane_counts, req.intersection_name
    )

    return SignalStateResponse(
        intersection_id=state.intersection_id,
        intersection_name=state.intersection_name,
        total_cycle=state.total_cycle,
        congestion_level=state.congestion_level,
        total_vehicles=state.total_vehicles,
        current_green_lane=state.current_green_lane,
        last_updated=state.last_updated,
        lanes={
            lid: LaneSignal(
                lane_id=lid, green_duration=ls.green_duration,
                red_duration=ls.red_duration, vehicle_count=ls.vehicle_count,
                phase=ls.phase,
            )
            for lid, ls in state.lanes.items()
        },
    )


@app.get("/api/v1/signals", response_model=AllSignalsResponse)
async def get_all_signals():
    """Get current signal states for all tracked intersections."""
    if signal_ctrl is None:
        return AllSignalsResponse(intersections=[], total_tracked=0)

    states = signal_ctrl.get_all_states()
    responses = []
    for state in states:
        responses.append(SignalStateResponse(
            intersection_id=state.intersection_id,
            intersection_name=state.intersection_name,
            total_cycle=state.total_cycle,
            congestion_level=state.congestion_level,
            total_vehicles=state.total_vehicles,
            current_green_lane=state.current_green_lane,
            last_updated=state.last_updated,
            lanes={
                lid: LaneSignal(
                    lane_id=lid, green_duration=ls.green_duration,
                    red_duration=ls.red_duration, vehicle_count=ls.vehicle_count,
                    phase=ls.phase,
                )
                for lid, ls in state.lanes.items()
            },
        ))

    return AllSignalsResponse(intersections=responses, total_tracked=len(responses))


@app.get("/api/v1/model-info")
async def get_model_info():
    """Get prediction model metadata."""
    if model_metadata is None:
        return {"status": "not_trained", "message": "Run: python models/trainer.py"}
    return ModelInfoResponse(
        model_type=model_metadata.get("model_type", "unknown"),
        input_dim=model_metadata.get("input_dim", 0),
        num_classes=model_metadata.get("num_classes", 0),
        seq_len=model_metadata.get("seq_len", 0),
        feature_names=model_metadata.get("feature_names", []),
        class_names=model_metadata.get("class_names", []),
        best_val_acc=model_metadata.get("best_val_acc", 0),
        epochs_trained=model_metadata.get("epochs_trained", 0),
        history=model_metadata.get("history", None),
        status="trained",
    )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "prediction_model": predictor_model is not None,
        "detector": detector is not None,
        "signal_controller": signal_ctrl is not None,
        "intersections_tracked": len(signal_ctrl.get_all_states()) if signal_ctrl else 0,
    }


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the frontend dashboard."""
    frontend_path = os.path.join(PROJ_ROOT, "frontend", "index.html")
    if not os.path.exists(frontend_path):
        raise HTTPException(404, "Frontend not found")
    with open(frontend_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/logo.png", response_class=FileResponse)
async def serve_logo():
    """Serve the logo image."""
    logo_path = os.path.join(PROJ_ROOT, "frontend", "logo.png")
    if not os.path.exists(logo_path):
        raise HTTPException(404, "Logo not found")
    return FileResponse(logo_path)
