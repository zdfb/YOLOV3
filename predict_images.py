from PIL import Image
from utils.utils_yolo import YOLO

yolo = YOLO()
image = Image.open('bird1.jpg')
image = yolo.detect_image(image)
image.show()
