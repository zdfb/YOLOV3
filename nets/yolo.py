import torch
import torch.nn as nn
from nets.darknet import darknet53
from collections import OrderedDict


###### 功能：用于定义YOLO网络的主体部分 ######


def conv2d(filter_in, filter_out, kernel_size):
    # 每一个conv2d模块都由 conv + bn + relu 组成
    pad = (kernel_size-1) // 2
    conv = nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size = kernel_size, stride = 1, padding = pad, bias = False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

    return conv


def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),

        # 后两层用于产生最终输出维度
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size = 1, stride = 1, padding = 0, bias = True)
    )

    return m


class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes):
        super(YoloBody, self).__init__()

        self.backbone = darknet53()  # 主干网络

        out_filters = [64, 128, 256, 512, 1024]  # 主干网络每层输出维度

        self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))

        self.last_layer1_conv = conv2d(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor = 2, mode='nearest')

        self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))

        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor = 2, mode='nearest')

        self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)
        
        # 用于上采样的输出特征
        out0_branch = self.last_layer0[:5](x0)
        # 输出第一个尺度的预测结果
        out0 = self.last_layer0[5:](out0_branch)  
        
        # 减少特征维度
        x1_in = self.last_layer1_conv(out0_branch)
        # 上采样
        x1_in = self.last_layer1_upsample(x1_in)
        # 合并特征
        x1_in = torch.cat([x1_in, x1], 1)

        # 用于上采样的输出特征
        out1_branch = self.last_layer1[:5](x1_in)
        # 输出第二个尺度的预测结果
        out1 = self.last_layer1[5:](out1_branch)
        
        # 减少特征维度
        x2_in = self.last_layer2_conv(out1_branch)
        # 上采样
        x2_in = self.last_layer2_upsample(x2_in)
        # 合并特征
        x2_in = torch.cat([x2_in, x2], 1)
        
        # 输出第三个尺度的预测结果
        out2 = self.last_layer2(x2_in)

        return out0, out1, out2
