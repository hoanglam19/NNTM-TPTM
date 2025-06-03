import cv2
import mediapipe as mp
import numpy as np
import joblib
import smtplib
from email.mime.text import MIMEText

# Hàm gửi email cảnh báo
def send_email_alert():
    sender = "hoangvanlam19abc@gmail.com"
    receiver = "jenber2k3@gmail.com"
    password = "kcer yoxu xmpa etei"

    msg = MIMEText("⚠️ Cảnh báo: Phát hiện người bị ngã!")
    msg["Subject"] = "Fall Detected Alert"
    msg["From"] = sender
    msg["To"] = receiver

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.send_message(msg)
            print("✅ Đã gửi email cảnh báo.")
    except Exception as e:
        print("❌ Gửi email thất bại:", e)

# Load model đã huấn luyện
model = joblib.load("fall_model.pkl")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

email_sent = False  # ✅ Biến kiểm soát gửi email

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được camera.")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        row = []
        for lm in results.pose_landmarks.landmark:
            row += [lm.x, lm.y, lm.z]

        X = np.array(row).reshape(1, -1)
        prediction = model.predict(X)[0]

        if prediction == 1:
            cv2.putText(frame, "FALL DETECTED!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if not email_sent:
                send_email_alert()
                email_sent = True
        else:
            cv2.putText(frame, "Normal", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            email_sent = False  # reset nếu trở lại trạng thái bình thường

    cv2.imshow("Fall Detection - Press 'q' to exit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
