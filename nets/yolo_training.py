import math
import torch
import numpy as np
import torch.nn as nn


###### 功能：定义yolo损失函数部分 ######


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, device, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super(YOLOLoss, self).__init__()

        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask

        self.ignore_threshold = 0.5  # 忽略阈值
        self.device = device

    # 将tensor的值限制在一个范围之内（t_min, t_max），超出边界的值均使用边界值代替
    def clip_by_tensor(self, t, t_min, t_max):

        t = t.float()  # 转化为float

        # 小于极小值的部分用极小值代替
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        # 大于极大值的部分用极大值代替
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max

        return result

    # 均方误差
    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)

    # 二值cross-entropy loss
    def BCELoss(self, pred, target):

        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)  # 将tensor进行限制
        # BCEloss
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def forward(self, index, input, targets=None):

        # index代表输入的尺度索引

        bs = input.size(0)    # batch_size 
        in_h = input.size(2)  # 输入特征图的高  
        in_w = input.size(3)  # 输入特征图的宽 

        stride_h = self.input_shape[0] / in_h  # 高的下采样倍数
        stride_w = self.input_shape[1] / in_w  # 宽的下采样倍数
         
        # 将正常尺度的anchors转化为下采样后的尺寸
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        # 将输入转化为 (bs, 3, 13, 13, 5 + num_class)
        prediction = input.view(bs, len(self.anchors_mask[index]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # 相对于左上角grid_cell的偏移参数
        x = torch.sigmoid(prediction[..., 0])  # (bs, 3, 13, 13) 
        y = torch.sigmoid(prediction[..., 1])  # (bs, 3, 13, 13)

        # 相对于anchor的大小比例参数
        w = prediction[..., 2]  # (bs, 3, 13, 13)
        h = prediction[..., 3]  # (bs, 3, 13, 13)

        # 框的置信率
        conf = torch.sigmoid(prediction[..., 4]) # (bs, 3, 13, 13)

        # 物体类的预测置信率
        pred_cls = torch.sigmoid(prediction[..., 5:])  # (bs, 3, 13, 13, num_classes)
        
        # 获得拟合标签值, 指示正样本的标志位, 损失函数加权值
        y_true, noobj_mask, box_loss_scale = self.get_target(index, targets, scaled_anchors, in_h, in_w)

        # 丢弃IOU大于阈值的样本
        noobj_mask = self.get_ignore(index, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask)
 
        y_true = y_true.to(self.device)
        noobj_mask = noobj_mask.to(self.device)
        box_loss_scale = box_loss_scale.to(self.device)
        
        # 正样本损失函数加权
        box_loss_scale = 2 - box_loss_scale  
        
        # box_loss_scale 损失加权, 
        # y_true[..., 4] (bs, 3, 13, 13) 指示了正样本位置
        # 正例mask = y_true[..., 4]

        # x, y, w, h均只对正样本计算loss
        # 计算x损失
        loss_x = torch.sum(self.BCELoss(x, y_true[..., 0]) * box_loss_scale * y_true[..., 4])
        # 计算y损失
        loss_y = torch.sum(self.BCELoss(y, y_true[..., 1]) * box_loss_scale * y_true[..., 4])
        # 计算w损失
        loss_w = torch.sum(self.MSELoss(w, y_true[..., 2]) * 0.5 * box_loss_scale * y_true[..., 4])
        # 计算h损失
        loss_h = torch.sum(self.MSELoss(h, y_true[..., 3]) * 0.5 * box_loss_scale * y_true[..., 4])

        # 计算置信率损失, 对正样本及负样本计算
        loss_conf = torch.sum(self.BCELoss(conf, y_true[..., 4]) * y_true[..., 4]) + \
                    torch.sum(self.BCELoss(conf, y_true[..., 4]) * noobj_mask) 
        
        # 计算分类损失, 只对正样本计算
        loss_cls = torch.sum(self.BCELoss(pred_cls[y_true[..., 4] == 1], y_true[..., 5:][y_true[..., 4] == 1]))

        # 总体loss
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        # 正样本数量
        num_pos = torch.sum(y_true[..., 4])
        num_pos = torch.max(num_pos, torch.ones_like(num_pos))
        
        return loss, num_pos
    
    def calculate_iou(self, _box_a, _box_b):

        # 真实框的xmin, xmax
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
        # 真实框的ymin, ymax  
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2

        # anchor的xmin, xmax
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        # anchor的ymin, ymax  
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2  

        box_a = torch.zeros_like(_box_a)  # (R, 4)
        box_b = torch.zeros_like(_box_b)  # (3, 4)

        # 将真实框转化为(xmin, ymin, xmax, ymax)的形式
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        # 将anchor转化为(xmin, ymin, xmax, ymax)的形式
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

        A = box_a.size(0)  # 真实框的数量 -- R
        B = box_b.size(0)  # anhors的数量 -- 3
       
        # min(x1_max, x2_max) 与 min(y1_max, y2_max)
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        # max(x1_min, x2_min) 与 max(y1_min, y2_max)
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        
        # max(max_xy - min_xy, 0)
        inter = torch.clamp((max_xy - min_xy), min=0)
        # 重合区域的面积, 形状为(A, B)
        inter = inter[:, :, 0] * inter[:, :, 1]
        
        # 真实框区域的面积
        area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        # anchor区域的面积
        area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]

        union = area_a + area_b - inter
        return inter / union  # [A,B]

    def get_target(self, index, targets, anchors, in_h, in_w):
       
        # targets为一个列表， 每个元素代表batch里面的一张图
        # 每个元素的形状为 (R, 5), 其中R为该张图片中存在的目标的数量
        # (x中点，y中点，x长度，y长度，类别) 除类别外在数据加载阶段均除以了输入图像尺寸

        bs = len(targets)

        # bs, 3, 13, 13, 用于指示负样本
        noobj_mask = torch.ones(bs, len(self.anchors_mask[index]), in_h, in_w, requires_grad=False)

        # bs, 3, 13, 13, 用于指示每个框的loss加权
        box_loss_scale = torch.zeros(bs, len(self.anchors_mask[index]), in_h, in_w, requires_grad=False)

        # bs, 3, 13, 13, 5+num_classes, 拟合标签
        y_true = torch.zeros(bs, len(self.anchors_mask[index]), in_h, in_w, self.bbox_attrs, requires_grad=False)

        for b in range(bs):
            # 若图片中不存在真实标注框, 则不进行计算
            if len(targets[b]) == 0:
                continue

            batch_target = torch.zeros_like(targets[b])  # R,5

            # ground_truth在特征图尺度下的坐标及大小
            batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h

            # ground_truth指示的类别index
            batch_target[:, 4] = targets[b][:, 4]  
            batch_target = batch_target.cpu()

            # 创造R个左上角坐标为(0,0), 大小为gt大小的框, 形状为(R, 4)
            gt_box = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 2:4]), 1))

            # 创造9个坐标为(0,0), 大小为anchor长宽的框, 形状为(9, 4) 
            anchor_shapes = torch.FloatTensor(torch.cat((torch.zeros((len(anchors), 2)), torch.FloatTensor(anchors)), 1))
            
            # 每个GT对应的最大anchor的索引序号, 输出形状为(R, 9)
            best_ns = torch.argmax(self.calculate_iou(gt_box, anchor_shapes), dim = -1)

            for t, best_n in enumerate(best_ns):
                # 判断重合程度最大的anchor是否在当前尺度内
                if best_n not in self.anchors_mask[index]: 
                    continue
                
                # 重合程度最大的框在当前尺度下的索引
                k = self.anchors_mask[index].index(best_n)  

                # GT所在grid_cell的x值
                i = torch.floor(batch_target[t, 0]).long()
                # GT所在grid_cell的y值
                j = torch.floor(batch_target[t, 1]).long()
                # GT标注的类别序号
                c = batch_target[t, 4].long()

                # noobj_mask (b,3,13,13)
                # 将GT所在Grid_cell, 重合程度最大anchor对应的值设置为正样本
                noobj_mask[b, k, j, i] = 0  

                # 该正样本对应的真实的需要拟合的值

                # x坐标相对于左上角的偏移值
                y_true[b, k, j, i, 0] = batch_target[t, 0] - i.float()
                # y坐标相对于左上角的偏移值  
                y_true[b, k, j, i, 1] = batch_target[t, 1] - j.float()
                # 相对于anchor长度的偏移值
                y_true[b, k, j, i, 2] = math.log(batch_target[t, 2] / anchors[best_n][0])  
                # 相对于anchor宽度的偏移值
                y_true[b, k, j, i, 3] = math.log(batch_target[t, 3] / anchors[best_n][1]) 
                # 框的置信度真实值 
                y_true[b, k, j, i, 4] = 1
                # 第C类对应的位设置为1
                y_true[b, k, j, i, c + 5] = 1

                # 正样本对应的loss权重, 真实框面积 // 特征图面积
                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h
        return y_true, noobj_mask, box_loss_scale

    def get_ignore(self, index, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):

        bs = len(targets) 

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor  
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # 生成所有grid_cell 左上角x值
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(int(bs * len(self.anchors_mask[index])), 1, 1).view(x.shape).type(FloatTensor)
        # 生成所有grid_cell 左上角y值  
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(int(bs * len(self.anchors_mask[index])), 1, 1).view(y.shape).type(FloatTensor)  

        # 特征图尺度下对应的anchors, 形状为(3, 2)
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[index]]  
        
        anchor_w = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([0]))  # anchor的宽
        anchor_h = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([1]))  # anchor的高
        
        # anchor的宽 (bs, 3, 13, 13)
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        # anchor的高 (bs, 3, 13, 13)  
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)  

        # 计算调整后的anchor位置 (bs, 3, 13, 13, 1)
        pred_boxes_x = torch.unsqueeze(x.data + grid_x, -1)  
        pred_boxes_y = torch.unsqueeze(y.data + grid_y, -1)
        
        # 计算调整后anchor的宽度及高度 (bs, 3, 13, 13, 1)
        pred_boxes_w = torch.unsqueeze(torch.exp(w.data) * anchor_w, -1)  
        pred_boxes_h = torch.unsqueeze(torch.exp(h.data) * anchor_h, -1)

        # 整合上述调整后的数据 (bs, 3, 13, 13, 4)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim = -1)  

        for b in range(bs):
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)  # 形状转化为 (3 * 13 * 13, 4)

            if len(targets[b]) > 0:
                batch_target = torch.zeros_like(targets[b])

                batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
                batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h

                # 取前四位 (R, 4), 具体内容为(x中点, y中点, x长度, y长度) 
                batch_target = batch_target[:, :4]  
                
                # 对每个GT及调整后的anchors计算IOU, 形状为(R, 3 * 13 * 13)
                anch_ious = self.calculate_iou(batch_target, pred_boxes_for_ignore)  # 计算GT与所有调整后的anchors的IOU （R，3*13*13）
                # 对于每个anchor, IOU最大的GT （3 * 13 * 13）
                anch_ious_max, _ = torch.max(anch_ious, dim=0)  
                # 转化为(3,13,13)
                anch_ious_max = anch_ious_max.view(pred_boxes[b].size()[:3])
                # 将IOU大于阈值的位置于0, 既不是正样本，也不是负样本
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0
        
        return noobj_mask