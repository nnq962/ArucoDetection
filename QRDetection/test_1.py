import cv2
import cv2.aruco as aruco
import numpy as np

# Khởi tạo kết nối webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở webcam")
    exit()

# Lấy từ điển ArUco DICT_5X5_100 bằng cách sử dụng hàm getPredefinedDictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)

# Khởi tạo đối tượng tham số cho việc phát hiện marker
parameters = aruco.DetectorParameters()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không nhận được khung hình từ webcam")
        break

    # Chuyển đổi khung hình sang grayscale (tối ưu cho phát hiện marker)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện marker
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    # Nếu phát hiện được marker
    if ids is not None:
        # Vẽ khung quanh các marker được phát hiện
        aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Lặp qua từng marker để tính toán và hiển thị ID, góc xoay
        for i in range(len(ids)):
            # Mỗi marker có 4 góc (corners): thứ tự thường là [trên-trái, trên-phải, dưới-phải, dưới-trái]
            marker_corners = corners[i][0]
            marker_id = ids[i][0]
            
            # Tính góc xoay của marker: sử dụng vector từ góc trên-trái đến góc trên-phải
            vector = marker_corners[1] - marker_corners[0]
            angle = np.degrees(np.arctan2(vector[1], vector[0]))
            # Điều chỉnh góc về khoảng [0, 360)
            if angle < 0:
                angle += 360
                
            # Tính tâm của marker để đặt text hiển thị
            center = marker_corners.mean(axis=0).astype(int)
            text = f"ID: {marker_id}, Angle: {angle:.1f}"
            
            # Hiển thị text lên khung hình
            cv2.putText(frame, text, tuple(center), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        # Nếu không phát hiện marker, hiển thị thông báo
        cv2.putText(frame, "No markers detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Hiển thị khung hình kết quả
    cv2.imshow("AR Marker Detection", frame)
    
    # Nhấn ESC để thoát
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()