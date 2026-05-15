import cv2
import mediapipe as mp
import numpy as np
from playsound import playsound
from drawers import(
    draw_lm
)
#initalize pose class
mp_pose = mp.solutions.pose

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
        lm = results.pose_landmarks.landmark
        ear = (lm[7].x,  lm[7].y)
        neck = ((lm[11].x + lm[12].x) / 2,
        (lm[11].y + lm[12].y) / 2)
        shoulders = (lm[11].x, lm[11].y + 0.3)
        # hips = (lm[23].x, lm[23].y)

        neck_angle = calculate_angle(ear, neck, shoulders)
        print(f"Neck angle: {neck_angle:.1f}°")
        cv2.putText(frame, f"Neck: {neck_angle:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        draw_lm(frame, results)

        # for i in range(2):
        #     print(f'{mp_pose.PoseLandmark(i).name}:\n{results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]}')

    cv2.imshow("Slouch Checker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# #score posture (usage with Ergonomics)
# def get_neck_score(angle):
#     return

cap.release()
cv2.destroyAllWindows()
