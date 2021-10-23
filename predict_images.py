from PIL import Image
from utils.utils_yolo import YOLO

yolo = YOLO()
image = Image.open('street.jpg')
image = yolo.detect_image(image)
image.show()
