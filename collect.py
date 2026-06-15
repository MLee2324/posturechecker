import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
from datetime import datetime
from drawers import (
    draw_lm
)

# MediaPipe setup 
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, 
min_tracking_confidence=0.5, model_complexity=2)

# Directories 
RUNS_DIR = "runs"
GOOD_DIR = os.path.join(RUNS_DIR, "good_posture")
BAD_DIR  = os.path.join(RUNS_DIR, "bad_posture")
os.makedirs(GOOD_DIR, exist_ok=True)
os.makedirs(BAD_DIR,  exist_ok=True)

#  Calculate Angle 
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    a = a - b
    c = c - b

    cosine = np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c) + 1e-6)
    angle = np.arccos((cosine))
    return np.degrees(angle)

# Feature extraction 
def extract_features(lm):
    """
    Returns a dict of posture features from MediaPipe landmarks.

    Features chosen for side-profile desk posture:
      neck_angle          ear → neck-midpoint (turtle-neck)
      spine_angle         ear → shoulder → hip (hunched back)
      forward_head_offset horizontal distance ear vs shoulder (head jutting)
      shoulder_tilt       vertical difference L/R shoulders (uneven shrug)
    """
    # Key points 
    ear = np.array([lm[7].x,  lm[7].y])
    neck = np.array([(lm[11].x + lm[12].x) / 2,
                           (lm[11].y + lm[12].y) / 2])
    l_shoulder = np.array([lm[11].x, lm[11].y])
    r_shoulder = np.array([lm[12].x, lm[12].y])
    l_hip = np.array([lm[23].x, lm[23].y])
    r_hip = np.array([lm[24].x, lm[24].y])
    hip = (l_hip + r_hip) / 2

    # Angles 
    neck_angle = calculate_angle(ear, neck, l_shoulder)
    spine_angle  = calculate_angle(ear, l_shoulder, l_hip)

    # Distances 
    # Positive = ear is forward (in front of shoulder) — turtle neck
    forward_head_offset = ear[0] - neck[0]

    # Vertical difference between left and right shoulders (shrug / tilt)
    shoulder_tilt = abs(l_shoulder[1] - r_shoulder[1])

    return {
        "neck_angle": round(neck_angle, 4),
        "spine_angle": round(spine_angle, 4),
        "forward_head_offset": round(float(forward_head_offset), 4),
        "shoulder_tilt": round(float(shoulder_tilt), 4),
    }

# CSV helpers 
FIELDNAMES = ["neck_angle", "spine_angle", "forward_head_offset", "shoulder_tilt"]

def save_session(data, label):
    """Save a list of feature dicts to a timestamped CSV."""
    folder = GOOD_DIR if label == "good" else BAD_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = os.path.join(folder, f"{label}_{timestamp}.csv")
    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(data)

    print(f"[✓] Saved {len(data)} frames → {filename}")


# Collect Data 
def main():
    cap = cv2.VideoCapture(1)  # change to 1 if using external webcam

    label = None   # "good" or "bad"
    recording = False
    session_data = []
    countdown = 0
    countdown_start = 0

    print("\n╔══════════════════════════════════════╗")
    print("║       POSTURE DATA COLLECTOR         ║")
    print("╠══════════════════════════════════════╣")
    print("║  G  – start recording GOOD posture   ║")
    print("║  B  – start recording BAD posture    ║")
    print("║  S  – stop & save current session    ║")
    print("║  Q  – quit                           ║")
    print("╚══════════════════════════════════════╝\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # Extract & record 
        features = None
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            features = extract_features(lm)

            if recording and features:
                session_data.append(features)

            # Draw key landmarks
            draw_lm(frame,results)

        # Status bar
        if recording:
            color  = (0, 200, 0) if label == "good" else (0, 0, 220)
            status = f"REC [{label.upper()}]  {len(session_data)} frames"
        else:
            color  = (180, 180, 180)
            status = "IDLE – press G or B to record"

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (30, 30, 30), -1)
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2)

        # Live feature readout
        if features:
            lines = [
                f"Neck angle: {features['neck_angle']:.1f}",
                f"Spine angle: {features['spine_angle']:.1f}",
                f"Head offset: {features['forward_head_offset']:.3f}",
                f"Shoulder tilt: {features['shoulder_tilt']:.3f}",
            ]
            # Background panel
            cv2.rectangle(
                frame,
                (0, 50),
                (350, 190),
                (20, 20, 20),
                -1
            )

            for i, line in enumerate(lines):
                y = 85 + i * 30

                cv2.putText(
                    frame,
                    line,
                    (15, y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

        # Countdown overlay
        if countdown > 0:
            elapsed  = time.time() - countdown_start
            remaining = countdown - int(elapsed)
            if remaining > 0:
                cv2.putText(frame, str(remaining),
                            (frame.shape[1] // 2 - 30, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 4, (0, 255, 255), 6)
            else:
                countdown = 0
                recording = True
                print(f"[●] Recording {label.upper()} posture…  press S to stop")

        # Controls reminder at bottom
        cv2.rectangle(frame, (0, frame.shape[0] - 35),
                      (frame.shape[1], frame.shape[0]), (30, 30, 30), -1)
        cv2.putText(frame, "G=good  B=bad  S=save & stop  Q=quit",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("Posture Collector", frame)

        # Key handling 
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('g') and not recording and countdown == 0:
            label          = "good"
            session_data   = []
            countdown      = 3
            countdown_start = time.time()
            print(f"\n[→] Get into your GOOD posture… starting in 3s")

        elif key == ord('b') and not recording and countdown == 0:
            label          = "bad"
            session_data   = []
            countdown      = 3
            countdown_start = time.time()
            print(f"\n[→] Get into your BAD posture… starting in 3s")

        elif key == ord('s'):
            if recording and len(session_data) > 0:
                recording = False
                save_session(session_data, label)
                session_data = []
                label        = None
                print("[■] Session stopped and saved.\n")
            elif recording:
                print("[!] No frames recorded yet.")
            else:
                print("[!] Not currently recording.")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()