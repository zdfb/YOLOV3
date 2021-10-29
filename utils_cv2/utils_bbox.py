import numpy as np


###### 功能：定义输出框处理过程 ######


class DecodeBox():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape

        self.anchors_mask = anchors_mask

    def sigmoid(self, x):  # 实现sigmoid函数
        return 1/(1 + np.exp(- x))

    def decode_box(self, inputs):
        outputs = []
        for index in range(len(inputs)):
            input = inputs[index]

            batch_size = input.shape[0]  # 输入的batch_size
            input_height = input.shape[2]  # 输入的高度
            input_width = input.shape[3]  # 输入的宽度

            stride_h = self.input_shape[0] / input_height  # 高度下采样倍数
            stride_w = self.input_shape[1] / input_width  # 宽度下采样倍数
            
            # 该尺度下对应的放缩后多anchors
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors[self.anchors_mask[index]]]
            
            prediction = input.reshape(batch_size, len(self.anchors_mask[index]), self.bbox_attrs, input_height, input_width).transpose(0, 1, 3, 4, 2)
            
            # anchor x,y的调整参数
            x = self.sigmoid(prediction[..., 0])
            y = self.sigmoid(prediction[..., 1])

            # anchor 宽度和高度的调整参数
            w = prediction[..., 2]
            h = prediction[..., 3]

            # 每个grid_cell内每一个anchor的置信率
            conf = self.sigmoid(prediction[..., 4])
            # 种类置信度
            pred_cls = self.sigmoid(prediction[..., 5:])
            
            # grid_cell的位置
            grid_x = np.linspace(0, input_width - 1, input_width).repeat(input_height).reshape(input_width, input_height).transpose(1, 0)
            grid_x = grid_x.reshape(batch_size, input_width, input_height).repeat(len(self.anchors_mask[index]), axis=0).reshape(x.shape)

            grid_y = np.linspace(0, input_height - 1, input_height).repeat(input_width).reshape(input_width, input_height)
            grid_y = grid_y.reshape(batch_size, input_width, input_height).repeat(len(self.anchors_mask[index]), axis=0).reshape(y.shape)

            # 生成anchor的宽和高
            anchor_w = np.array(scaled_anchors)[:, 0].reshape(1, -1).transpose(1, 0)
            anchor_h = np.array(scaled_anchors)[:, 1].reshape(1, -1).transpose(1, 0)
            anchor_w = anchor_w.repeat(input_height * input_width, axis=1).reshape(batch_size, len(self.anchors_mask[index]), input_height, input_width)
            anchor_h = anchor_h.repeat(input_height * input_width, axis=1).reshape(batch_size, len(self.anchors_mask[index]), input_height, input_width)
            
            # 根据预测参数调整anchors的坐标及大小
            pred_boxes = np.zeros(prediction[..., :4].shape)
            pred_boxes[..., 0] = x + grid_x
            pred_boxes[..., 1] = y + grid_y
            pred_boxes[..., 2] = np.exp(w.data) * anchor_w
            pred_boxes[..., 3] = np.exp(h.data) * anchor_h

            # 缩小尺度，将上述调整后的anchor框进行归一化
            _scale = np.array([input_width, input_height, input_width, input_height])
            # 将调整后的anchor位置，坐标与框和种类多置信度进行合并
            output = np.concatenate((pred_boxes.reshape(batch_size, -1, 4) / _scale, conf.reshape(batch_size, -1, 1), pred_cls.reshape(batch_size, -1, self.num_classes)), axis=-1)
            
            outputs.append(output.data)  # 添加至输出列表中
        return outputs
    
    # nms算法
    def nms(self, boxes, scores, threshold):
        # 取出坐标
        xmin = boxes[:, 0]
        ymin = boxes[:, 1]
        xmax = boxes[:, 2]
        ymax = boxes[:, 3]

        order = scores.argsort()[::-1]  # 将得分按照从大到小的顺序排列,存储其索引
        area = (xmax - xmin) * (ymax - ymin)  # 计算anchor框的面积

        keep = [] 
        while order.shape[0] > 0:
            index = order[0]
            keep.append(index)  # 将得分最高的添加至最后输出list中

            # 计算IOU
            xx1 = np.maximum(xmin[index], xmin[order[1:]])
            yy1 = np.maximum(ymin[index], ymin[order[1:]])
            xx2 = np.minimum(xmax[index], xmax[order[1:]])
            yy2 = np.minimum(ymax[index], ymax[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            iou = (w * h) / (area[index] + area[order[1:]] - w * h)
            index_loss = np.where(iou <= threshold)[0]
            order = order[index_loss + 1]
        return keep
    
    def yolo_correct_boxes(self, box_xy, box_wh, image_shape):

        # 调换x轴与y轴
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        # 转化为ymin, xmin, ymax, xmax的形式
        box_mins = box_yx - (box_hw / 2.)
        box_maxs = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxs[..., 0:1], box_maxs[..., 1:2]], axis = -1)
        # 转化为原图尺寸
        boxes *= np.concatenate([image_shape, image_shape], axis = -1)

        return boxes

    # 进行非极大值抑制处理
    def non_max_suppression(self, prediction, num_classes, image_shape, conf_thres=0.5, nms_thres=0.4):
        # prediction的形状 (bs, num_anchors, 5 + num_class)
        box_corner = np.zeros(prediction.shape)
        # 将x, y, w, h形式转化为 xmin, ymin, xmax, ymax
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]
        
        output = [None for _ in range(len(prediction))]

        for index, image_pred in enumerate(prediction):
            # image_pred 形状为 (num_anchors, 5 + num_class)

            # 取得每个anchor置信率最大的种类及置信率
            class_conf = np.max(image_pred[:, 5:5 + num_classes], 1, keepdims=True)
            class_pred = np.argmax(image_pred[:, 5:5 + num_classes], 1).reshape(len(image_pred), -1)

            # 取得输出置信率大于置信率阈值的索引
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres)
            
            # 取得经过初筛的anchor框
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]

            if not image_pred.shape[0]:
                continue

            # detections的形状为 (num_anchors, 7)
            # 内容为 (xmin, ymin, xmax, ymax, anchor_conf, class_conf, class_id)
            detections = np.concatenate((image_pred[:, :5], class_conf, class_pred), axis=1)
            
            # 获取预测框中包含的所有类别
            unique_labels = np.unique(detections[:, -1]) 
            
            for c in unique_labels:
                detections_class = detections[detections[:, -1] == c]
                # 经过nms处理
                keep = self.nms(
                    detections_class[:, :4],  # 框的坐标
                    detections_class[:, 4] * detections_class[:, 5],  # 输出置信率
                    nms_thres  # nms阈值
                )
                max_detections = detections_class[keep]  # 经过nms处理之后剩下的框
                # 添加到最终的输出结果
                output[index] = max_detections if output[index] is None else np.concatenate((output[index], max_detections))
            
            if output[index] is not None:
                # 将框转化为中点及坐标的形式
                box_xy, box_wh = (output[index][:, 0:2] + output[index][:, 2:4]) / 2, output[index][:, 2:4] - output[index][:, 0:2]
                output[index][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, image_shape)
        return output
