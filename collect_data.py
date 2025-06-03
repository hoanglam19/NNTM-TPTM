
import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

os.makedirs("data", exist_ok=True)
csv_file = "data/fall_data.csv"

if not os.path.isfile(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = []
        for i in range(33):
            header += [f'x{i}', f'y{i}', f'z{i}']
        header.append('label')
        writer.writerow(header)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
label = input("Nhap nhan hanh dong (fall/normal): ").strip()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Khong doc duoc frame tu camera.")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        row = []
        for lm in results.pose_landmarks.landmark:
            row += [lm.x, lm.y, lm.z]

        row.append(label)
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    cv2.imshow("Thu thap du lieu - nhan 'q' de thoat", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
