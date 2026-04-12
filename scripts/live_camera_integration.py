import cv2
import time
import sys
import os

# Set up paths to import models
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_ROOT, "models"))

from vehicle_detector import VehicleDetector
from signal_controller import AdaptiveSignalController

# Real-world IP CCTV Camera Stream URL (Example)
# RTSP is the standard protocol for CCTV cameras setup at traffic junctions.
RTSP_URL = "rtsp://admin:password@192.168.1.100:554/stream1"

# For testing right now, 0 means your laptop's/pc's WebCam
CAMERA_SOURCE = 0 

def main():
    print("="*50)
    print("🚦 LIVE HARDWARE CAMERA INTEGRATION DEMO 🚦")
    print("="*50)
    
    # 1. Initialize YOLOv8 Detector and Signal Controller
    print("[1] Loading YOLOv8 Model with PERFECT GHOST FILTERING...")
    detector = VehicleDetector()
    detector.load_model()
    
    print("[2] Initializing Adaptive Signal Controller...")
    signal_ctrl = AdaptiveSignalController()
    
    # 2. Connect to the Live CCTV Camera
    print(f"[3] Connecting to Camera Source: {CAMERA_SOURCE}...")
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    
    if not cap.isOpened():
        print("Error: Live Camera could not be opened. Check connection.")
        return

    print("Camera connected. Processing Live Stream... (Press 'q' to stop)")

    frame_skip = 5  # Process every 5th frame to save CPU
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or disrupted.")
            break
            
        frame_count += 1
        
        # 3. Detect Vehicles on the live frame (using YOLO)
        if frame_count % frame_skip == 0:
            # Detect vehicles and distribute them in 4 virtual lanes (North, South, East, West)
            result = detector.detect(frame, mode="intersection")
            
            # 4. Feed counts into the Signal Controller to change Traffic Lights
            state = signal_ctrl.update_signals("INT_LIVE", result.lane_counts, "Real-Time Junction")
            
            # Print the real-time background signal changes
            busiest_lane = state.current_green_lane
            print(f"\n--- Live Update ---")
            print(f"Detected: N:{result.lane_counts.get('Lane_N',0)} S:{result.lane_counts.get('Lane_S',0)} "
                  f"E:{result.lane_counts.get('Lane_E',0)} W:{result.lane_counts.get('Lane_W',0)}")
            print(f"Light Changed: Busiest lane {busiest_lane} given GREEN.")
        
        # 5. Visualize the detections on the screen
        annotated_frame = detector.draw_detections(frame, result) if 'result' in locals() else frame
        
        # Adding a visual banner for the signal state
        if 'state' in locals():
            cv2.putText(annotated_frame, f"GREEN LIGHT: {state.current_green_lane}", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Show the live feed
        cv2.imshow("Real-Time Traffic Light Controller", annotated_frame)
        
        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
