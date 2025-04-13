# 🧍‍♂️ Human Pose Classification using YOLOv8 Keypoint Detection

This project demonstrates a complete pipeline for real-time human pose classification using YOLOv8 Pose Estimation. It classifies common human actions such as **Standing**, **Sitting**, **Walking**, and **Lying**, along with computing the **hip joint angle** while sitting.

---

## 📌 Features

- 🔍 Keypoint detection using **YOLOv8 Pose Model**
- 🕒 Timestamped keypoint logging for each frame
- 📐 Heuristic rule-based classification:
  - Distinguishes static poses (Standing, Sitting, Lying)
  - Uses velocity across frames to identify dynamic actions (Walking)
- 🦵 Calculates **hip angle** (Shoulder–Hip–Knee) while sitting
- 🖼️ Annotated real-time video output
- 🎥 Saves the final output to a video file

---

## 🧠 How It Works

### 1. Keypoint Detection
- Utilizes `YOLOv8n-pose.pt` from Ultralytics to detect 17 human body keypoints.

### 2. Keypoint Logging with Timestamps
- Captures each frame, stores keypoints, and associates them with the corresponding timestamp.

### 3. Pose Classification Logic
- **Lying:** Low body height and small hip-to-shoulder difference
- **Sitting:** Small hip-to-shoulder difference + large ankle-to-knee difference
- **Walking:** Detected via velocity of hip/knee/ankle keypoints across frames
- **Standing:** Default case if above are not met

### 4. Hip Angle Calculation
- When sitting, calculates angle between vectors:
  - Shoulder → Hip
  - Knee → Hip
- Provides ergonomic insight into the sitting posture

---

## 🚀 Getting Started

### Requirements
```bash
pip install ultralytics opencv-python numpy pandas
```

### Run the Script
```bash
python pose_classification.py
```

### Input
- Replace `your_video.mp4` with your input video file.

### Output
- `output_pose_classification.mp4`: Annotated video with pose labels and sitting angles

---

## 📁 Project Structure
```
├── pose_detection_real_time.py         # Main script
├── my_vid1.mp4                # Input video
├── output_pose_classification.mp4 # Output annotated video
```

---

## 🧠 Future Improvements
- Replace heuristic logic with a trained classifier (e.g. SVM, LSTM)
- Add person tracking for better per-identity pose smoothing
- Extend to additional actions (e.g. running, jumping)

---

## 🙌 Acknowledgements
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

---

## 📬 Contact
tkseneee@gmail.com - Feel free to reach out for suggestions, feedback, or collaboration!

**#DeepLearning #ComputerVision #YOLOv8 #PoseEstimation #Python #AI #OpenCV**
