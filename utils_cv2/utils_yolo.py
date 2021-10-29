import cv2
import numpy as np
from PIL import ImageDraw, ImageFont
from utils_cv2.utils_bbox import DecodeBox
from utils.utils import cvtColor, get_classes, get_anchors, preprocess_input, resize_image, ncolors


###### 功能：解析模型，生成最终结果 ######


class YOLO(object):
    def __init__(self):
        super(YOLO, self).__init__()

        model_path = 'model_data/yolo_weights.onnx'  # 模型存储路径
        classes_path = 'model_data/coco_classes.txt'  # 类别信息存储路径
        anchors_path = 'model_data/yolo_anchors.txt'  # anchors存储路径

        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]  # anchor mask
        self.input_shape = [416, 416]  # 输入图像尺寸 [高, 宽] 

        self.confidence = 0.5  # 置信率初筛阈值
        self.nms_iou = 0.3  # 非极大值抑制IOU阈值
        
        # 获取种类名及数量
        self.class_names, self.num_classes = get_classes(classes_path)
        # 获取anchors及数量
        self.anchors, self.num_anchors = get_anchors(anchors_path)
        # 获取种类个数的随机RGB颜色
        self.colors = ncolors(self.num_classes)

        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        # 加载模型
        self.model = cv2.dnn.readNetFromONNX(model_path)
        self.output_names = ['output1', 'output2', 'output3']
    
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])  # 输入图像的宽和高
        image = cvtColor(image)  # 将输入图片转化为RGB形式

        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        # 对输入图像进行预处理
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype = 'float32')),(2, 0, 1)), 0)
        
        outputs = []  # 用于存放三个尺度的输出
        self.model.setInput(image_data)
        for output_name in self.output_names:
            output = self.model.forward(output_name)  # 前向传播
            outputs.append(output)  # 将该尺度输出存放至list内
        
        # 根据上述网络输出结果调整anchor框
        outputs = self.bbox_util.decode_box(outputs)
        # 进行非极大值抑制处理
        results = self.bbox_util.non_max_suppression(np.concatenate(outputs, axis=1), self.num_classes, image_shape, self.confidence, self.nms_iou)
        if results[0] is None:
            return image
        
        top_label = np.array(results[0][:, 6], dtype = 'int32')  # 预测类别
        top_conf = results[0][:, 4] * results[0][:, 5]  # 预测置信率
        top_boxes = results[0][:, :4]  # 预测框位置 (num_bbox, (ymin, xmin, ymax, xmax))

        # 绘制图像上的标注框
        font_size = np.floor(2e-2 * image.size[1]).astype('int32')  # 定义字体大小
        font = ImageFont.truetype(font='model_data/simhei.ttf', size=font_size)  # 定义字体样式
        
        for index, class_id in list(enumerate(top_label)):
            predicted_class = self.class_names[int(class_id)]  # 取出预测类别名称

            box = top_boxes[index]  # 预测框的位置信息 (ymin, xmin, ymax, xmax)
            score = top_conf[index]  # 预测框的置信度

            ymin, xmin, ymax, xmax = box  # 取出坐标详细信息
            
            # 标签内容
            label_text = '{} {:.2f}'.format(predicted_class, score)
            
            # 绘制图像
            draw = ImageDraw.Draw(image)

            # 获取标签区域大小
            label_size = draw.textsize(label_text, font)

            # 绘制标签包围框
            draw.rectangle((xmin, ymin - label_size[1], xmin + label_size[0], ymin), fill = self.colors[class_id])
            # 绘制目标框
            draw.rectangle((xmin, ymin, xmax, ymax), outline = self.colors[class_id], width = 3)
            # 绘制标签
            draw.text((xmin, ymin - label_size[1]), label_text, fill = (255, 255, 255), font=font)
            del draw
        return image
