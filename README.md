ự án nhận diện té ngã theo thời gian thực bằng webcam, kết hợp giao diện web với Flask và gửi cảnh báo qua email.

🔧 Chức năng chính
Nhận diện hành vi té ngã từ video.

Hiển thị video và trạng thái ("Normal" hoặc "FALL DETECTED!") trên giao diện web.

Gửi email cảnh báo khi phát hiện té ngã.

🛠 Công nghệ sử dụng
Python + Flask: Giao diện web và xử lý server.

OpenCV: Xử lý video từ webcam.

scikit-learn: Huấn luyện mô hình RandomForest.

TensorFlow Lite: Dự đoán nhanh trong thời gian thực.

SMTP (Gmail): Gửi email cảnh báo.

🖼 Giao diện
Web đơn giản chạy tại localhost:5000.

Hiển thị video webcam kèm dòng cảnh báo nếu có té ngã.
