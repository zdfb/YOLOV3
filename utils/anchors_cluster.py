import os
import numpy as np
import xml.etree.ElementTree as ET

###### 功能: 聚类产生9个尺寸的anchors ######

num_anchors = 9  # 生成anchor的数量
input_shape = [416, 416]  # 输入图片尺寸 前面是宽，后面是高
VOCdevkit_path = 'VOCdevkit'  # VOC格式的数据集存储根目录 


def load_dataset():
    # xml文件存储根目录
    xmlfilepath = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
    # 遍历路径下所有文件
    total_xml = os.listdir(xmlfilepath)

    bbox_list = []
    for xml_id in total_xml:
        xml_path = os.path.join(xmlfilepath, xml_id)  # xml文件路径
        xml_file = open(xml_path, encoding='utf-8')  # 打开xml文件
        tree = ET.parse(xml_file)

        height = int(tree.findtext("./size/height"))  # 获取图片高度
        width = int(tree.findtext("./size/width"))  # 获取图片宽度
     
        if height == 0 or width == 0:
            continue

        for obj in tree.iter("object"):

            # 获取位置信息
            xmin = int(obj.findtext("bndbox/xmin")) / width
            ymin = int(obj.findtext("bndbox/ymin")) / height
            xmax = int(obj.findtext("bndbox/xmax")) / width
            ymax = int(obj.findtext("bndbox/ymax")) / height

            # 将上述位置信息转化为float形式
            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # 记录anchor的宽以及高
            bbox_list.append([xmax - xmin, ymax - ymin])
    return np.array(bbox_list)


# 计算单个ground_truth与多个候选anchor的iou
def calculate_iou(bbox, anchors):
    
    xmax = np.minimum(anchors[:, 0], bbox[0])
    ymax = np.minimum(anchors[:, 1], bbox[1])
    
    # 重合区域的面积
    inter = xmax * ymax
    area_a = bbox[0] * bbox[1]  # ground_truth的面积
    area_b = anchors[:, 0] * anchors[:, 1]  # anchors的面积

    union = area_a + area_b - inter 
    iou = inter / union
    return iou


# 计算ground_truth和候选anchor的平均交并比
def avg_iou(bbox, anchors):
    return np.mean([np.max(calculate_iou(bbox[i], anchors)) for i in range(bbox.shape[0])])

# 定义kmeans
def kmeans(bbox, k):

    num_bbox = bbox.shape[0]  # 所有框的个数
    distances = np.empty((num_bbox, k))  # 用于记录每个bbox与候选anchors的距离
    last_anchors = np.zeros((num_bbox,))  # 上一次每个bbox距离最近的anchor索引

    # 初始化anchors, 随机选取k个bbox
    anchors = bbox[np.random.choice(num_bbox, k, replace=False)]

    # 开始聚类
    while True:
        # 计算每个bbox与k个anchor的距离, 距离用1-IOU表示
        for index in range(num_bbox):
            distances[index] = 1 - calculate_iou(bbox[index], anchors)
        
        # 记录距离最小的anchor索引
        nearest_anchors = np.argmin(distances, axis=1)
        # 若当前索引与上次索引一致, 停止聚类
        if (last_anchors == nearest_anchors).all():
            break
        # 新的anchor为所有与之临近anchor的均值
        for anchor_index in range(k):
            anchors[anchor_index] = np.median(bbox[nearest_anchors == anchor_index], axis=0)
        
        last_anchors = nearest_anchors
    return anchors


if __name__ == '__main__':
    bbox = load_dataset()  # 读取数据

    anchors = kmeans(bbox, k=num_anchors)  # 进行kmeans聚类
    avarage_iou = avg_iou(bbox, anchors)  # 计算平均IOU

    anchors = anchors * input_shape  # 转化为图像尺寸
    anchors = np.array(anchors)   # 转化为numpy格式
    anchors_area = anchors[:, 0] * anchors[:, 1]  # 计算anchors的面积

    anchors_index = np.argsort(anchors_area)  # 根据面积从小到大进行排序
    anchors = anchors[anchors_index]
    print(anchors)
    print("平均IOU：%.2f"%(avarage_iou))
