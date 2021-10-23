import torch.nn as nn
from collections import OrderedDict


###### 功能：定义darknet 网络结构 ######


# BasicBlock由两部分组成
# 第一部分：卷积核大小为1 + bn + relu 由planes[1]降至planes[0]
# 第二部分：卷积核大小为3 + bn + relu 由planes[0]升至planes[1]
# 输入维度为 planes[1], 输出维度为 planes[1]

class BasicBlock(nn.Module):
    def __init__(self, planes):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(planes[1], planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32

        # 416,416,3 -> 416,416,32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)

        # 416,416,32 -> 208,208,64  
        self.layer1 = self._make_layer([32, 64], layers[0])

        # 208,208,64 -> 104,104,128  
        self.layer2 = self._make_layer([64, 128], layers[1])

        # 104,104,128 -> 52,52,256   
        self.layer3 = self._make_layer([128, 256], layers[2])

        # 52,52,256 -> 26,26,512     
        self.layer4 = self._make_layer([256, 512], layers[3])
        
        # 26,26,512 -> 13,13,1024    
        self.layer5 = self._make_layer([512, 1024], layers[4])

    def _make_layer(self, planes, blocks):
        layers = []

        # 由planes[0]维升至planes[1]维
        layers.append(("ds_conv", nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))

        # 经过多个残差连接
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5


def darknet53():
    model = DarkNet([1, 2, 8, 8, 4])
    return model