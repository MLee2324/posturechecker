import cv2
import os
import mediapipe as mp
import numpy as np
import time
import joblib 
import threading
import pandas as pd
from drawers import (
    draw_lm
)

model = joblib.load("train/posture_model.pkl")

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

def play_alert():
    os.system("afplay /System/Library/Sounds/Glass.aiff")

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    a = a - b
    c = c - b

    cosine = np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c) + 1e-6)
    angle = np.arccos(cosine)

    return np.degrees(angle)


def extract_features(lm):
    ear = np.array([lm[7].x, lm[7].y])

    neck = np.array([
        (lm[11].x + lm[12].x) / 2,
        (lm[11].y + lm[12].y) / 2
    ])

    l_shoulder = np.array([lm[11].x, lm[11].y])
    r_shoulder = np.array([lm[12].x, lm[12].y])
    l_hip = np.array([lm[23].x, lm[23].y])

    neck_angle = calculate_angle(ear, neck, l_shoulder)
    spine_angle = calculate_angle(ear, l_shoulder, l_hip)

    forward_head_offset = ear[0] - neck[0]
    shoulder_tilt = abs(l_shoulder[1] - r_shoulder[1])

    return {
        "neck_angle": round(neck_angle, 4),
        "spine_angle": round(spine_angle, 4),
        "forward_head_offset": round(float(forward_head_offset), 4),
        "shoulder_tilt": round(float(shoulder_tilt), 4),
    }


def main():
    cap = cv2.VideoCapture(0)

    last_beep_time = 0
    BEEP_INTERVAL = 2

    print("\n╔══════════════════════════════════════╗")
    print("║        POSTURE CHECKER               ║")
    print("╠══════════════════════════════════════╣")
    print("║  Live posture feature display        ║")
    print("║  Q  – quit                           ║")
    print("╚══════════════════════════════════════╝\n")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        features = None

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            features = extract_features(lm)

            draw_lm(frame, results)

        status = "Live Posture Checker"

        cv2.rectangle(frame, (0, 50), (430, 225), (20, 20, 20), -1)
        cv2.putText(
            frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.85,
            (0, 255, 0),
            2
        )

        if features:
            X_live = pd.DataFrame([features], columns=[
            "neck_angle",
            "spine_angle",
            "forward_head_offset",
            "shoulder_tilt"
            ])

            prediction = model.predict(X_live)[0]

            if prediction == 0:
                posture_status = "GOOD POSTURE"
                posture_color = (0,255,0)
            else:
                posture_status = "BAD POSTURE"
                posture_color = (0,0,255)

                if time.time() - last_beep_time > BEEP_INTERVAL:
                    threading.Thread(
                        target=play_alert,
                        daemon=True,
                    ).start()
                    last_beep_time = time.time()

            lines = [
                f"Prediction: {posture_status}",
                f"Neck angle: {features['neck_angle']:.1f}",
                f"Spine angle: {features['spine_angle']:.1f}",
                f"Head offset: {features['forward_head_offset']:.3f}",
                f"Shoulder tilt: {features['shoulder_tilt']:.3f}",
            ]

            cv2.rectangle(frame, (0, 50), (350, 190), (20, 20, 20), -1)

            for i, line in enumerate(lines):
                y = 85 + i * 30

                text_color = posture_color if i == 0 else (255, 255, 255)

                cv2.putText(
                    frame,
                    line,
                    (15, y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    text_color,
                    2
                )

        cv2.rectangle(
            frame,
            (0, frame.shape[0] - 35),
            (frame.shape[1], frame.shape[0]),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            frame,
            "Q=quit",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1
        )

        cv2.imshow("Slouch Checker", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()