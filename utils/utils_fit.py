import time
import torch
import numpy as np


###### 功能：yolov3模型训练一个epoch ######


def fit_one_epoch(model, yolo_loss, optimizer, train_data, test_data, device): 
    
    start_time = time.time() # 获取当前时间
    model.train()  # 训练过程 
    
    loss_train_list = []
    for step, data in enumerate(train_data):   
        images, targets = data[0], data[1]  # 取出图片及标签

        # 将数据转化为torch.tensor形式
        with torch.no_grad():
            images = torch.from_numpy(images).type(torch.FloatTensor).to(device)
            targets = [torch.from_numpy(ann).type(torch.FloatTensor).to(device) for ann in targets]
    
        optimizer.zero_grad()  # 清零梯度
        outputs = model(images) # 前向传播

        loss_value_all = 0 # 三个尺度总的loss
        num_pos_all = 0 # 正样本个数

        for index in range(len(outputs)):
            # 计算该尺度下的loss及正样本数量
            loss_item, num_pos = yolo_loss(index, outputs[index], targets)
            loss_value_all += loss_item
            num_pos_all += num_pos
        
        loss_value = loss_value_all / num_pos_all  # 除以正样本数

        loss_value.backward()  # 反向传播
        optimizer.step()  # 优化器迭代

        loss_train_list.append(loss_value.item())

        # 画进度条
        rate = (step + 1) / len(train_data)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss_value), end="")
    print()  

    model.eval()  # 测试过程

    loss_test_list = []
    for step, data in enumerate(test_data):
        images, targets = data[0], data[1]  # 取出图片及标签

        with torch.no_grad():
            # 转化为torch.tensor
            images = torch.from_numpy(images).type(torch.FloatTensor).to(device)
            targets = [torch.from_numpy(ann).type(torch.FloatTensor).to(device) for ann in targets]
        
            optimizer.zero_grad()  # 清零梯度
            outputs = model(images)  # 前向传播

            loss_value_all = 0
            num_pos_all = 0

            for index in range(len(outputs)):
                # 计算该尺度下的loss及正样本数量
                loss_item, num_pos = yolo_loss(index, outputs[index], targets)
                loss_value_all += loss_item
                num_pos_all += num_pos
            
            loss_value = loss_value_all / num_pos_all  # 除以正样本数

            loss_test_list.append(loss_value.item())

        # 画进度条
        rate = (step + 1) / len(test_data)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtest loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss_value), end="")
    print()

    train_loss = np.mean(loss_train_list)  # 该epoch总的训练loss
    test_loss = np.mean(loss_test_list)  # 该epoch总的测试loss
    stop_time = time.time()  # 获取当前时间
    
    print('total_train_loss: %.3f, total_test_loss: %.3f, epoch_time: %.3f.'%(train_loss, test_loss, stop_time - start_time))
    return train_loss, test_loss