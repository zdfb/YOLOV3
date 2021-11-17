import os
import random
import xml.etree.ElementTree as ET
from utils.utils import get_classes


###### 功能：用于读取及分割数据集 ######


# VOC形式数据集路径
VOCdevkit_path = 'VOCdevkit'
VOCdevit_sets = ['train', 'test']

# 训练集:测试集 = 9:1
train_percent = 0.9

# 定义类别存储路径
classes_path = 'model_data/name_classes.txt'
# 获取所有类别的名称及数量
classes, classes_num = get_classes(classes_path)


# 生成数据集划分txt文件
def generate_split_txt():

    # 标签存储路径
    xmlfilepath = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
    # 生成的训练集与测试集id存储路径
    saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')

    # 遍历路径下所有文件
    total_xml = os.listdir(xmlfilepath) 
    # 所有样本总数
    num_samples = len(total_xml)
    # 随机打乱样本list
    random.shuffle(total_xml)
    # 训练样本总数
    train_num = int(train_percent * num_samples)
    
    # 创建存储训练及测试样本id的txt文件
    f_train = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    f_test = open(os.path.join(saveBasePath, 'test.txt'), 'w')

    for index, xml in enumerate(total_xml):
        # 剔除.xml后缀
        name = xml[:-4] + '\n'
        # 判断是否处于训练样本内
        if index < train_num:
            f_train.write(name)  # 写入训练集txt文件
        else:
            f_test.write(name)   # 写入测试集txt文件

    # 关闭打开的txt文件    
    f_train.close()  
    f_test.close()
    print("Generate txts in ImageSets/main done.")


# 加载xml内的annotation写入txt文件内
def convert_annotation(image_id, list_file):

    # 获得xml路径 VOCdevkit/VOC2007/Annotations/1234.xml
    in_file_path = os.path.join(VOCdevkit_path, 'VOC2007/Annotations/%s.xml' % (image_id))
    in_file = open(in_file_path, encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        # 取出类别名称
        cls = obj.find('name').text
        # 取出当前标注类别索引
        cls_id = classes.index(cls)

        # 取出标注框
        xmlbox = obj.find('bndbox')
        # 取出当前标注的框的位置信息 (xmin, ymin, xmax, ymax)
        coordinates = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))

        # 写入文件 格式形如  -xmin,ymin,xmax,ymax,cls_id  其中-为空格
        list_file.write(" " + ",".join([str(coordinate) for coordinate in coordinates]) + ',' + str(cls_id))


# 生成存储标签信息的txt文件
def generate_annotation_txt():

    saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')

    for image_set in VOCdevit_sets:
        # 存储数据集样本id的txt文件
        set_txt_path = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main/%s.txt'%(image_set))
        image_ids = open(set_txt_path, encoding = 'utf-8').read().strip().split()
        
        # 打开存储标签的txt文件
        list_file = open(os.path.join(saveBasePath, '2007_%s.txt'%(image_set)),'w', encoding = 'utf-8')

        for image_id in image_ids:
            # 获取图片绝对路径
            image_path = '%s/VOC2007/JPEGImages/%s.jpg'%(os.path.abspath(VOCdevkit_path), image_id)
            # 写入txt文件图片路径
            list_file.write(image_path)
            # 获取位置标签并写入txt文件
            convert_annotation(image_id, list_file)
            # 换行
            list_file.write('\n')
        list_file.close()
    print("Generate annotation txt done.")



if __name__ == "__main__":
    # 设置随机种
    random.seed(0)
    generate_split_txt()
    generate_annotation_txt()