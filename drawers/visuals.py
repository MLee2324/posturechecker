import cv2
import mediapipe as mp
import numpy as np

def draw_lm(frame, results):
    lm = results.pose_landmarks.landmark
    h, w, _ = frame.shape

    # Convert normalized coordinates to pixels
    ear = (int(lm[7].x * w), int(lm[7].y * h))

    neck = (int(((lm[11].x + lm[12].x) / 2) * w),
            int(((lm[11].y + lm[12].y) / 2) * h))

    shoulder = (int(lm[11].x * w), int(lm[11].y * h))

    hip = (int(((lm[23].x + lm[24].x) / 2) * w),
           int(((lm[23].y + lm[24].y) / 2) * h))

    # Draw lines
    cv2.line(frame, ear, neck, (0, 0, 255), 2)
    cv2.line(frame, neck, shoulder, (0, 0, 255), 2)
    cv2.line(frame, shoulder, hip, (0, 200, 255), 2)

    # Draw dots
    cv2.circle(frame, ear, 8, (0, 255, 255), -1)
    cv2.circle(frame, neck, 8, (255, 0, 255), -1)
    cv2.circle(frame, shoulder, 8, (0, 255, 255), -1)
    cv2.circle(frame, hip, 8, (0, 255, 255), -1)