
import cv2
import mediapipe as mp
import pyautogui

# Khởi tạo các biến
x1 = y1 = x2 = y2 = 0

# Mở webcam
webcam = cv2.VideoCapture(0)

# Khởi tạo Mediapipe
my_hands = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

while True:
    # Lấy khung hình từ webcam
    ret, image = webcam.read()
    if not ret:
        print("Không thể lấy hình ảnh từ webcam")
        break
    
    # Lật ảnh để tránh hình ảnh bị ngược
    image = cv2.flip(image, 1)
    
    # Lấy kích thước khung hình
    frame_height, frame_width, _ = image.shape
    
    # Chuyển đổi hình ảnh sang RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Xử lý hình ảnh để phát hiện tay
    output = my_hands.process(rgb_image)
    hands = output.multi_hand_landmarks
    
    if hands:
        for hand in hands:
            # Vẽ các điểm trên bàn tay
            drawing_utils.draw_landmarks(image, hand)
            landmarks = hand.landmark
            
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                
                if id == 8:  # Đầu ngón trỏ
                    cv2.circle(img=image, center=(x, y), radius=8, color=(0, 255, 255), thickness=3)
                    x1 = x
                    y1 = y
                
                if id == 4:  # Đầu ngón cái
                    cv2.circle(img=image, center=(x, y), radius=8, color=(0, 0, 255), thickness=3)
                    x2 = x
                    y2 = y
        
        # Tính khoảng cách giữa hai điểm
        dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        
        # Điều khiển âm lượng
        if dist > 50:
            pyautogui.press("volumeup")
            print("Tăng âm lượng")
        else :
            pyautogui.press("volumedown")
            print("Giảm âm lượng")
    
    # Hiển thị hình ảnh
    cv2.imshow("Hand volume control using python", image)
    
    # Nhấn phím ESC để thoát
    key = cv2.waitKey(10)
    if key == 27:
        break

# Giải phóng webcam
webcam.release()
cv2.destroyAllWindows()
