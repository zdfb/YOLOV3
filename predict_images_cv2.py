from PIL import Image
from utils_cv2.utils_yolo import YOLO

image_path = 'Image_samples/street.jpg'  # 测试图片路径

yolo = YOLO()

image = Image.open(image_path)
image = yolo.detect_image(image)
image.show()
