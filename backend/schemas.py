"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime


# ── Prediction Schemas ─────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """User requests congestion prediction for a specific date/time."""
    target_datetime: str = Field(..., description="Target datetime, e.g. '2025-03-15 08:30:00'")
    intersection_id: Optional[str] = Field(None, description="Specific intersection, or None for all")

class LanePrediction(BaseModel):
    lane_id: str
    predicted_congestion: str
    confidence: float
    predicted_vehicle_count: int
    predicted_avg_speed: float

class IntersectionPrediction(BaseModel):
    intersection_id: str
    intersection_name: str
    city: str
    timestamp: str
    overall_congestion: str
    lanes: List[LanePrediction]

class PredictionResponse(BaseModel):
    predictions: List[IntersectionPrediction]
    model_type: str = "BiLSTM_MultiHeadAttention"
    timestamp: str


# ── Detection Schemas ──────────────────────────────────────────────────────

class DetectionRequest(BaseModel):
    """Video frame for vehicle detection."""
    image_base64: str = Field(..., description="Base64-encoded image frame")
    intersection_id: str = Field("INT_01", description="Intersection identifier")
    mode: str = Field("intersection", description="Detection mode ('intersection' or 'single')")

class VehicleInfo(BaseModel):
    class_name: str
    confidence: float
    bbox: List[int]
    lane: Optional[str]

class DetectionResponse(BaseModel):
    intersection_id: str
    total_vehicles: int
    lane_counts: Dict[str, int]
    vehicles: List[VehicleInfo]
    annotated_image_base64: Optional[str] = None


# ── Signal Control Schemas ─────────────────────────────────────────────────

class LaneSignal(BaseModel):
    lane_id: str
    green_duration: int
    red_duration: int
    vehicle_count: int
    phase: str

class SignalStateResponse(BaseModel):
    intersection_id: str
    intersection_name: str
    total_cycle: int
    congestion_level: str
    total_vehicles: int
    current_green_lane: str
    last_updated: str
    lanes: Dict[str, LaneSignal]

class AllSignalsResponse(BaseModel):
    intersections: List[SignalStateResponse]
    total_tracked: int


# ── Model Info ─────────────────────────────────────────────────────────────

class ModelInfoResponse(BaseModel):
    model_type: str
    input_dim: int
    num_classes: int
    seq_len: int
    feature_names: List[str]
    class_names: List[str]
    best_val_acc: float
    epochs_trained: int
    history: Optional[Dict[str, List[float]]] = None
    status: str


# ── Lane Counts (manual input) ────────────────────────────────────────────

class ManualLaneCountRequest(BaseModel):
    """Manually input lane counts for signal calculation."""
    intersection_id: str = "INT_01"
    intersection_name: str = "Test Intersection"
    lane_counts: Dict[str, int] = Field(
        ...,
        description="Vehicle count per lane",
        examples=[{"Lane_N": 15, "Lane_S": 8, "Lane_E": 22, "Lane_W": 5}]
    )
