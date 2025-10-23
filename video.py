import os
import sys
from ultralytics import YOLO
import cv2
from tqdm import tqdm

print("[INFO] Loading YOLOv8 model...")
try:
    model = YOLO("yolov8n.pt")
    print("[INFO] Model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    sys.exit(1)

video_path = r"video_path" #replace with your video path

# Check if video file exists
if not os.path.exists(video_path):
    print(f"[ERROR] Video file '{video_path}' not found.")
    sys.exit(1)

# Input video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[ERROR] Could not open video file '{video_path}'.")
    sys.exit(1)

# Output setup
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

base_name = os.path.splitext(os.path.basename(video_path))[0]
output_name = f"{base_name}_out.mp4"
out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
pbar = tqdm(total=frame_count, desc="Processing", unit="frame")

# Frame-by-frame detection
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        output = results[0].plot()
        out.write(output)
        pbar.update(1)
except Exception as e:
    print(f"[ERROR] Error during processing: {e}")
    cap.release()
    out.release()
    pbar.close()
    sys.exit(1)


cap.release()
out.release()
pbar.close()
print(f"[DONE] Video processed successfully -> {output_name}")
