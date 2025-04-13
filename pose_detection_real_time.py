from ultralytics import YOLO
import cv2
import numpy as np
import collections

# Load the YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

# Open the video file and set FPS
cap = cv2.VideoCapture("my_vid1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_num = 0

# Get frame dimensions from the input video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize VideoWriter for output video
output_filename = "output_pose_classification.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# Dictionaries for storing previous detections and smoothing history per detection index
prev_detections = {}
state_history = {}
history_window = 3  # Number of frames used for temporal smoothing

def classify_pose_from_keypoints(keypoints, previous_keypoints=None, velocity_threshold=50):
    """
    Classify the pose using a combination of static keypoint features and dynamic rate of change.
    
    Static analysis first checks if the person is "Lying" or "Sitting" based solely on joint positions.
    If neither condition is met and previous keypoints are available, the function calculates the
    average velocity (from hip, knee, and ankle centers). If that velocity exceeds the threshold,
    the pose is flagged as "Walking". Otherwise, it defaults to "Standing".
    
    Parameters:
        keypoints (np.array): [17, 3] array containing current keypoints (COCO order).
        previous_keypoints (np.array or None): Previous frame's keypoints for dynamic comparison.
        velocity_threshold (float): Velocity threshold (pixels per second) for "Walking".

    Returns:
        str: Predicted pose ("Lying", "Sitting", "Walking", or "Standing").
    """
    kp = np.array(keypoints)[:, :2]  # Use only the (x, y) positions
    try:
        # Calculate static joint positions:
        shoulder_y = np.mean([kp[5][1], kp[6][1]])
        hip_y      = np.mean([kp[11][1], kp[12][1]])
        knee_y     = np.mean([kp[13][1], kp[14][1]])
        ankle_y    = np.mean([kp[15][1], kp[16][1]])
        
        hip_shoulder_diff = hip_y - shoulder_y
        knee_hip_diff     = knee_y - hip_y
        ankle_knee_diff   = ankle_y - knee_y
        body_height       = ankle_y - shoulder_y
        
        # PRIORITY 1: Check if the person is lying:
        if body_height < 20 and hip_shoulder_diff < 40:
            return "Lying"
        
        # PRIORITY 2: Check if the person is sitting:
        # Criteria: a low hip-to-shoulder difference (< 60) and a high ankle-to-knee difference (> 40)
        if hip_shoulder_diff < 60 and ankle_knee_diff > 40:
            return "Sitting"
        
        # PRIORITY 3: Use dynamic (velocity) cues if previous keypoints exist.
        if previous_keypoints is not None and previous_keypoints.shape[0] >= 17:
            prev_kp = np.array(previous_keypoints)[:, :2]
            prev_hip_center = np.mean([prev_kp[11], prev_kp[12]], axis=0)
            prev_knee_center = np.mean([prev_kp[13], prev_kp[14]], axis=0)
            prev_ankle_center = np.mean([prev_kp[15], prev_kp[16]], axis=0)
            
            current_hip_center = np.mean([kp[11], kp[12]], axis=0)
            current_knee_center = np.mean([kp[13], kp[14]], axis=0)
            current_ankle_center = np.mean([kp[15], kp[16]], axis=0)
            
            # Calculate displacements between frames
            disp_hip = np.linalg.norm(current_hip_center - prev_hip_center)
            disp_knee = np.linalg.norm(current_knee_center - prev_knee_center)
            disp_ankle = np.linalg.norm(current_ankle_center - prev_ankle_center)
            
            # Calculate velocities (pixels per second)
            velocity_hip = disp_hip * fps
            velocity_knee = disp_knee * fps
            velocity_ankle = disp_ankle * fps
            avg_velocity = (velocity_hip + velocity_knee + velocity_ankle) / 3
            
            if avg_velocity > velocity_threshold:
                return "Walking"
        
        # Default fallback:
        return "Standing"
    
    except Exception as e:
        print("Error in classification:", e)
        return "Unknown"

def compute_sitting_angle(keypoints):
    """
    Computes the angle at the hip joint (in degrees) for a sitting posture.
    The angle is defined at the hip between the line connecting the hip center to the shoulder center,
    and the line connecting the hip center to the knee center.
    
    The hip center is the average of the left and right hip points (indices 11 and 12),
    shoulder center is the average of points at indices 5 and 6, and
    knee center is the average of points at indices 13 and 14.
    
    Returns:
        float: The computed angle at the hip, in degrees.
    """
    kp = np.array(keypoints)[:, :2]
    
    # Compute centers
    hip_center = np.mean([kp[11], kp[12]], axis=0)
    shoulder_center = np.mean([kp[5], kp[6]], axis=0)
    knee_center = np.mean([kp[13], kp[14]], axis=0)
    
    # Compute vectors from the hip
    vector_shoulder = shoulder_center - hip_center
    vector_knee = knee_center - hip_center
    
    # Calculate the angle between the two vectors using the dot product
    dot_product = np.dot(vector_shoulder, vector_knee)
    norm_shoulder = np.linalg.norm(vector_shoulder)
    norm_knee = np.linalg.norm(vector_knee)
    
    if norm_shoulder == 0 or norm_knee == 0:
        return 0.0
    
    cos_angle = np.clip(dot_product / (norm_shoulder * norm_knee), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    # Use a detection confidence threshold of 0.5
    results = model.predict(frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()
    keypoints_data = results[0].keypoints.data.cpu().numpy()  # Expected shape: [num_persons, 17, 3]
    
    for i, keypoints in enumerate(keypoints_data):
        if keypoints.shape[0] < 17:
            #print(f"[Frame {frame_num}, Person {i}] Incomplete keypoints, skipping detection.")
            continue

        # Retrieve previous keypoints for temporal dynamics (if available)
        prev_kp = prev_detections.get(i, None)
        # Classify the pose
        current_prediction = classify_pose_from_keypoints(keypoints, previous_keypoints=prev_kp, velocity_threshold=50)
        # Update previous detection for this index
        prev_detections[i] = keypoints
        
        # Maintain a short history for smoothing (using majority voting)
        if i not in state_history:
            state_history[i] = []
        state_history[i].append(current_prediction)
        if len(state_history[i]) > history_window:
            state_history[i].pop(0)
        final_pose = collections.Counter(state_history[i]).most_common(1)[0][0]
        
        # Determine text placement using the nose keypoint if available,
        # otherwise fallback to the midpoint of the shoulder keypoints.
        try:
            nose_x, nose_y = keypoints[0][:2]
            if nose_x == 0 or nose_y == 0:
                raise ValueError("Invalid nose coordinates")
        except Exception:
            nose_x = np.mean([keypoints[5][0], keypoints[6][0]])
            nose_y = np.mean([keypoints[5][1], keypoints[6][1]])
        
        # Draw the main pose label on the first line
        text_position = (int(nose_x), int(nose_y) - 5)
        cv2.putText(annotated_frame, final_pose, text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # If the pose is sitting, compute and display the sitting angle (at the hip) on the next line
        if final_pose == "Sitting":
            angle = compute_sitting_angle(keypoints)
            # Offset the y coordinate for the second line (e.g., 25 pixels below the first line)
            text_position2 = (int(nose_x)-85, int(nose_y) + 15)
            cv2.putText(annotated_frame, f"Angle: {angle:.1f}deg", text_position2,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Write annotated frame to output video
    writer.write(annotated_frame)
    
    cv2.imshow("Pose Classification", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"Output video saved as {output_filename}")
