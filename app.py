from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import joblib
import smtplib
from email.mime.text import MIMEText

app = Flask(__name__)
model = joblib.load("fall_model.pkl")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

email_sent = False  # cờ kiểm tra đã gửi mail hay chưa


def send_email_alert():
    sender = "hoangvanlam19abc@gmail.com"
    receiver = "jenber2k3@gmail.com"
    password = "kcer yoxu xmpa etei"

    msg = MIMEText("⚠️ Cảnh báo: Có người bị ngã!")
    msg["Subject"] = "Fall Detected"
    msg["From"] = sender
    msg["To"] = receiver

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.send_message(msg)
            print("✅ Đã gửi email cảnh báo.")
    except Exception as e:
        print("❌ Gửi email thất bại:", e)


def gen_frames():
    global email_sent
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
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
                cv2.putText(frame, "FALL DETECTED!", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                if not email_sent:
                    send_email_alert()
                    email_sent = True
            else:
                cv2.putText(frame, "Normal", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                email_sent = False

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
