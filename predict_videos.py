import cv2
import numpy as np
from PIL import Image
from utils.utils_yolo import YOLO

video_path = 'Image_samples/person1.avi'  # 测试视频路径  
cap = cv2.VideoCapture(video_path)

yolo = YOLO()

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(np.uint8(frame))
    frame = np.array(yolo.detect_image(frame))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
