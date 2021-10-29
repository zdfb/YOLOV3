import torch
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import YOLOLoss
from utils.utils_fit import fit_one_epoch
from utils.utils import get_anchors, get_classes
from utils.dataloader import YoloDataset, yolo_dataset_collate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class train_yolo():
    def __init__(self):
        super(train_yolo, self).__init__()

        classes_path = 'model_data/coco_classes.txt'  # 类别存储路径
        anchors_path = 'model_data/yolo_anchors.txt'  # anchors保存路径
        model_path = 'model_data/yolo_weights.pth'  # 模型存储路径
        
        train_annotation_path = 'VOCdevkit/VOC2007/ImageSets/Main/2007_train.txt'  # 训练集标签文件存储路径
        test_annotation_path = 'VOCdevkit/VOC2007/ImageSets/Main/2007_test.txt'  # 测试集标签文件存储路径

        self.input_shape = [416, 416]  #  输入尺寸
        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]  # anchor_mask
        
        # 类别名称及类别总数
        self.class_names, self.num_classes = get_classes(classes_path)
        # anchors及anchors总数
        self.anchors, self.num_anchors = get_anchors(anchors_path)

        # 创建Yolo模型
        model = YoloBody(self.anchors_mask, self.num_classes)
        print('Load Weights from {}.'.format(model_path))

        model_dict = model.state_dict()  # 模型参数
        pretrained_dict = torch.load(model_path, map_location = device)  # 预训练模型的参数
        # 替换key相同且shape相同的值
        pretrained_dict = {k:v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)  # 更新参数
        model.load_state_dict(model_dict)  # 加载参数

        if torch.cuda.is_available():
            cudnn.benchmark = True
            model = model.to(device)
        
        self.model = model
        
        # 定义损失函数
        self.yolo_loss = YOLOLoss(self.anchors, self.num_classes, self.input_shape, device)

        with open(train_annotation_path, 'r', encoding = 'utf-8') as f:
            self.train_lines = f.readlines()  # 读取训练集数据
        with open(test_annotation_path, 'r', encoding = 'utf-8') as f:
            self.test_lines = f.readlines()  # 读取测试集数据
        
        self.loss_test_min = 1e9  # 初始化最小测试集loss
    
    def train(self, batch_size, learning_rate, start_epoch, end_epoch, Freeze = False):
        
        # 定义优化器
        optimizer = optim.Adam(self.model.parameters(), learning_rate, weight_decay = 5e-4)

        # 学习率下降策略
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.92)

        # 定义训练集与测试集
        train_dataset = YoloDataset(self.train_lines, self.input_shape, self.num_classes, train = True)
        test_dataset = YoloDataset(self.test_lines, self.input_shape, self.num_classes, train = False)
        train_data = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = 4,
                                  pin_memory = True, drop_last = True, collate_fn = yolo_dataset_collate)
        test_data = DataLoader(test_dataset, shuffle = True, batch_size = batch_size,num_workers = 4,
                                  pin_memory = True, drop_last = True, collate_fn = yolo_dataset_collate)
        

        # 冻结backbone参数
        if Freeze:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.model.backbone.parameters():
                param.requires_grad = True
        
        # 开始训练
        for epoch in range(start_epoch, end_epoch):
            print('Epoch: ', epoch)
            train_loss, test_loss = fit_one_epoch(self.model, self.yolo_loss, optimizer, train_data, test_data, device)
            lr_scheduler.step()
            
            # 若测试集loss小于当前极小值，保存当前模型
            if test_loss < self.loss_test_min:
                self.loss_test_min = test_loss
                torch.save(self.model.state_dict(), 'yolo_v3.pth')
        
    def total_train(self):

        # 首先进行backbone冻结训练
        Freeze_batch_size = 8
        Freeze_lr = 1e-3
        Init_epoch = 0
        Freeze_epoch = 10

        self.train(Freeze_batch_size, Freeze_lr, Init_epoch, Freeze_epoch, Freeze = True)

        #解冻backbone训练
        batch_size = 4
        learning_rate = 1e-4
        end_epoch = 30

        self.train(batch_size, learning_rate, Freeze_epoch, end_epoch, Freeze = False)

if __name__ == "__main__":
    train = train_yolo()
    train.total_train()
