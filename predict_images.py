from PIL import Image
from utils.utils_yolo import YOLO

yolo = YOLO()
image = Image.open('Image_samples/street.jpg')
image = yolo.detect_image(image)
image.save('Image_samples/result.jpg')
image.show()
