import os
import shutil
from PIL import Image
from tqdm import tqdm
from utils.utils_yolo import YOLO
import xml.etree.ElementTree as ET
from utils.utils import get_classes
from utils.utils_map import get_map


###### 功能：计算模型mAP ######


class Calculate_mAP():
    def __init__(self):
        super(Calculate_mAP, self).__init__()

        classes_path = 'model_data/name_classes.txt'  # 类别信息存放路径

        self.VOCdevkit_path = 'VOCdevkit'  # 数据集存储路径
        self.map_out_path = 'map_out'  # 存放模型输出信息的文件夹   
        self.map_threshold = 0.50  # 计算map的阈值 如：map50 

        # 取出每张图片的id
        self.image_ids = open(os.path.join(self.VOCdevkit_path, 'VOC2007/ImageSets/Main/test.txt')).read().strip().split()
  
        # 创建原图标签信息，预测信息及图片路径
        if not os.path.exists(self.map_out_path):
            os.makedirs(self.map_out_path)
        if not os.path.exists(os.path.join(self.map_out_path, 'ground-truth')):
            os.makedirs(os.path.join(self.map_out_path, 'ground-truth'))
        if not os.path.exists(os.path.join(self.map_out_path, 'detection-results')):
            os.makedirs(os.path.join(self.map_out_path, 'detection-results'))
        
        self.class_names, _ = get_classes(classes_path)

    def calculate_map(self):
        yolo = YOLO()  # 实例化yolo
        
        # 生成测试集的预测结果
        print(" Get predict results.")
        for image_id in tqdm(self.image_ids):  # 获取每张图像id
            image_path = os.path.join(self.VOCdevkit_path, 'VOC2007/JPEGImages/' + image_id + '.jpg')  # 获取每张图像的路径
            image = Image.open(image_path)  
            yolo.get_map_txt(image_id, image, self.map_out_path)  # 获取预测结果并写入txt文件中
        print(" Get predict results done!")

        # 生成真实标签结果
        print(" Get ground truth results.")
        for image_id in tqdm(self.image_ids):
            f = open(os.path.join(self.map_out_path, 'ground-truth/' + image_id + '.txt'), 'w')
            # 打开保存标签的xml文件
            root = ET.parse(os.path.join(self.VOCdevkit_path, 'VOC2007/Annotations/' + image_id + '.xml')).getroot()
            for obj in root.findall('object'):
                obj_name = obj.find('name').text  # 获取真实类别名
                if obj_name not in self.class_names:
                    continue
                # 获取真实框位置信息
                bndbox = obj.find('bndbox')
                xmin = bndbox.find('xmin').text
                ymin = bndbox.find('ymin').text
                xmax = bndbox.find('xmax').text
                ymax = bndbox.find('ymax').text
                # 写入txt文件
                f.write('%s %s %s %s %s\n' % (obj_name, xmin, ymin, xmax, ymax))
            f.close()
        print('Get ground truth results done!')

        # 获取map
        get_map(self.map_threshold, path=self.map_out_path)
        shutil.rmtree(self.map_out_path)


if __name__ == '__main__':
    map = Calculate_mAP()
    map.calculate_map()

