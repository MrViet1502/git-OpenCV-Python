# import cv2
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# webcam = cv2.VideoCapture(0)
# while True:
#     _,img = webcam.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.5,4)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
#     cv2.imshow("face detection", img)
#     key = cv2.waitKey(10)
#     if key == 27:
#         break
# webcam.release()
# cv2.destroyAllWindows()

import cv2

# Tải các cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Mở webcam
webcam = cv2.VideoCapture(0)

while True:
    # Đọc hình ảnh từ webcam
    _, img = webcam.read()
    
    # Chuyển đổi hình ảnh sang màu xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Lật ảnh để tránh hình ảnh bị ngược
    img = cv2.flip(img, 1)
    # Phát hiện mặt
    faces = face_cascade.detectMultiScale(gray, 1.5, 4)
    
    for (x, y, w, h) in faces:
        # Vẽ hình chữ nhật quanh mặt
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Cắt vùng mặt từ hình ảnh xám
        roi_gray = gray[y:y + h, x:x + w]
        
        # Phát hiện nụ cười trong vùng mặt
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        
        if len(smiles) > 0:
            cv2.putText(img, 'Crying', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, 'Not Crying', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Hiển thị hình ảnh
    cv2.imshow("Face Detection and Smile Detection", img)
    
    # Thoát khi nhấn phím ESC
    key = cv2.waitKey(10)
    if key == 27:
        break

# Giải phóng webcam và đóng tất cả các cửa sổ
webcam.release()
cv2.destroyAllWindows()
