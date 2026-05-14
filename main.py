import cv2
import mediapipe as mp
import numpy as np
from playsound import playsound
#initalize pose class
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, 
min_tracking_confidence=0.5, model_complexity=2)
cap = cv2.VideoCapture(0)

# #Calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    a = a - b
    c = c - b

    cosine = (np.dot(a,c)) / ((np.linalg.norm(a)) * (np.linalg.norm(c)))
    angle = np.arccos((cosine))
    return np.degrees(angle)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        for i in range(2):
            print(f'{mp_pose.PoseLandmark(i).name}:\n{results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]}')

    cv2.imshow("Slouch Checker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# #score posture (usage with Ergonomics)
# def get_neck_score(angle):
#     return

    cap.release()
cv2.destroyAllWindows()
