from PIL import Image
from utils_cv2.utils_yolo import YOLO

yolo = YOLO()
image = Image.open('Image_samples/street.jpg')
image = yolo.detect_image(image)
image.show()