import random
import colorsys
import numpy as np
from PIL import Image


###### 功能：定义用到的一些工具函数 ######

random.seed(30)  # 设置随机数种

# 将输入图像统一为RGB形式
def cvtColor(image):
    image = image.convert('RGB')
    return image


# resize输入图像尺寸
def resize_image(image, size):
    w, h = size
    new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


# 获得具体预测类别信息
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


# 获得先验框尺寸
def get_anchors(anchors_path):
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)  # reshape 为(R, 2)形式
    return anchors, len(anchors)


# 获取学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# 输入图像预处理 将255范围内的像素值转化值0～1
def preprocess_input(image):
    image /= 255.0
    return image

# 随机选取N个HLS颜色


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


# 随机选取N个RGB颜色
def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append((r, g, b))
    random.shuffle(rgb_colors)

    return rgb_colors

