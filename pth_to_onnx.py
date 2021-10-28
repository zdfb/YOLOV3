import torch
from nets.yolo import YoloBody
from utils.utils import get_classes


###### 功能: 将pth文件转化为onnx文件


classes_path = 'model_data/coco_classes.txt'  # 类别信息存储路径
pth_path = 'model_data/yolo_weights.pth'  # pth文件存储路径
onnx_path = 'model_data/yolo_weights.onnx'  # onnx文件存储路径

anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]  # anchor mask
class_names, num_classes = get_classes(classes_path)  # 获取种类名及数量

model = YoloBody(anchors_mask, num_classes)  # 定义Yolo模型
model.load_state_dict(torch.load(pth_path, map_location='cpu'))  # 加载预训练模型
model.eval()  # 将模型置为推理模式

input_tensor = torch.randn(1, 3, 416, 416)  # 设置随机输入
torch.onnx.export(model, input_tensor, onnx_path, verbose=False, input_names=['input'], output_names=['output1', 'output2', 'output3'], opset_version=11)
