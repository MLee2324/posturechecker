import cv2
import mediapipe as mp
import numpy as np

def draw_lm(frame, results):
    lm = results.pose_landmarks.landmark
    h, w, _ = frame.shape

    # Convert normalized coordinates to pixels
    ear = (int(lm[7].x * w),  int(lm[7].y * h))
    neck = (int(((lm[11].x + lm[12].x) / 2) * w),
              int(((lm[11].y + lm[12].y) / 2) * h))
    shoulder = (int(lm[11].x * w), int(lm[11].y * h))

    # Draw lines
    cv2.line(frame, ear, neck, (0, 0, 255), 2)       # red line
    cv2.line(frame, neck, shoulder, (0, 0, 255), 2)  # red line

    # Draw dots
    cv2.circle(frame, ear, 8, (0, 255, 255), -1)  # yellow
    cv2.circle(frame, neck, 8, (255, 0, 255), -1)  # pink
    cv2.circle(frame, shoulder, 8, (0, 255, 255), -1)  # yellow