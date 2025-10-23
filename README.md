
# 📌 YOLOv8 Video Object Detection

This project uses the **Ultralytics YOLOv8 model** to perform real-time object detection on a video file. It processes the video frame-by-frame and exports a new output video with bounding boxes and labels for every detected object.

---

## ✅ Features
- Loads the **YOLOv8 model** automatically
- Supports **MP4 video input**
- Detects objects frame-by-frame
- Shows progress using a **tqdm progress bar**
- Saves a new processed video with visual detection results

---

## 📌 Requirements

Install the required libraries before running the script:

```sh
pip install ultralytics opencv-python tqdm
