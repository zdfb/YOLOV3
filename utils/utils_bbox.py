import torch
import numpy as np
from torchvision.ops import nms

class DecodeBox():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super(DecodeBox, self).__init__()
        
        self.anchors = anchors  
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape

        self.anchors_mask = anchors_mask
    
    def decode_box(self, inputs):
        outputs = []  # 存放输出的列表
        for index, input in enumerate(inputs):
            batch_size = input.size(0)  # 输入的batch_size
            input_height = input.size(2) # 输入的特征图高度
            input_width = input.size(3)  # 输入的特征图宽度

            stride_h = self.input_shape[0] / input_height  # 高度缩小倍率
            stride_w = self.input_shape[1] / input_width   # 宽度缩小倍率

            # 缩小该尺度对应的anchor
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors[self.anchors_mask[index]]]
            # 将输出转化为 bs, 3, 13, 13, 5 + num_class的形式
            prediction = input.view(batch_size, len(self.anchors_mask[index]), self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

            # 坐标偏移参数
            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])

            # 长度偏移参数
            w = prediction[..., 2]
            h = prediction[..., 3]

            # 框置信率
            conf = torch.sigmoid(prediction[..., 4])
            # 种类置信率
            pred_cls = torch.sigmoid(prediction[..., 5:])
            
            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            # 每个grid_cell的左上角坐标 (bs, 3, 13, 13)
            grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
                batch_size * len(self.anchors_mask[index]), 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
                batch_size * len(self.anchors_mask[index]), 1, 1).view(y.shape).type(FloatTensor)
            
            # 每一个grid_cell赋予3种anchor的长和宽
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)
            
            # 根据预测参数调整anchors的大小及位置
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
            
            # 缩小尺度, 将上述调整后的anchor框均进行归一化
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            # 将调整后的anchor位置、坐标与框和种类的置信度进行合并
            # (bs, 3 * 13 * 13, 5 + num_classes)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale, conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)

            outputs.append(output.data)  # 添加至输出列表中
        return outputs

    def yolo_correct_boxes(self, box_xy, box_wh, image_shape):

        # 调换x轴与y轴
        box_yx = box_xy[..., ::-1]  
        box_hw = box_wh[..., ::-1]
        image_shape = np.array(image_shape)
        
        # 转化为ymin, xymin, ymax, xmax的形式
        box_mins = box_yx - (box_hw / 2.) 
        box_maxs = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxs[..., 0:1], box_maxs[..., 1:2]], axis = -1)

        # 应该为 boxes * input_shape * image_shape / input_shape == boxes * image_shape
        boxes *= np.concatenate([image_shape, image_shape], axis = -1)

        return boxes

    def non_max_suppression(self, prediction, num_classes, image_shape, conf_thres = 0.5, nms_thres = 0.4):

        # prediction的形状 (bs, num_anchors, 5 + num_class)
        box_corner = prediction.new(prediction.shape)
        # 将 x, y, w, h 形式转化为 xmin, ymin, xmax, ymax
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]

        for index, image_pred in enumerate(prediction):
            # image_pred 形状为 (num_anchors, 5 + num_class)

            # 取得每个anchor置信率最大的种类及其置信率 形状为 (num_anchors, 1)
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
            
            # 取得输出置信率大于置信率阈值的索引
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()
            
            # 取得经过初筛后的anchor框
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            
            # 排除初筛后没有目标存在的情况
            if not image_pred.size(0):
                continue
            
            # detections的形状为 (num_anchors, 7)
            # 内容为 (xmin, ymin, xmax, ymax, anchor_conf, class_conf, class_id)
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

            # 获取预测框中包含的所有类别
            unique_labels = detections[:, -1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()
            
            # 单独处理每个种类
            for c in unique_labels:
                detections_class = detections[detections[:, -1] == c]
                # 经过nms处理
                keep = nms(
                    detections_class[:, :4],  # 坐标
                    detections_class[:, 4] * detections_class[:, 5], # 输出置信率
                    nms_thres
                )
                max_detections = detections_class[keep]  # nms处理后剩下的目标框
                # 添加到最终输出结果
                output[index] = max_detections if output[index] is None else torch.cat((output[index], max_detections))

            if output[index] is not None:
                output[index] = output[index].cpu().numpy()

                # 转化为中点及长宽的形式
                box_xy, box_wh = (output[index][:, 0:2] + output[index][:, 2:4]) / 2, output[index][:, 2:4] - output[index][:, 0:2]
                output[index][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, image_shape)
        return output
