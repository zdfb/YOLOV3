import os
import glob
import json
import shutil
import numpy as np


###### 功能：计算mAP ######


def txt_to_list(path):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


def VOC_ap(recall, precision):
    recall.insert(0, 0.0)  # 在开头插入0.0
    recall.append(1.0)  # 在末尾插入1.0

    precision.insert(0, 0.0)  # 在开头插入0.0
    precision.append(0.0)  # 在末尾插入0.0

    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])

    index_list = []
    for index in range(1, len(recall)):
        if recall[index] != recall[index - 1]:
            index_list.append(index)

    ap = 0.0
    for index in index_list:
        ap += ((recall[index] - recall[index - 1]) * precision[index])
    return ap


def get_map(MINOVERLAP, path='./map_out'):

    ground_truth_path = os.path.join(path, 'ground-truth')  # 用于存放GT信息
    detection_results = os.path.join(path, 'detection-results')  # 用于存放检测结果
    temp_files = os.path.join(path, '.temp_files')  # 临时存储路径

    if not os.path.exists(temp_files):
        os.makedirs(temp_files)  # 创建临时文件夹

    ground_truth_files_list = glob.glob(ground_truth_path + '/*.txt')  # 获取所有标签txt文件
    ground_truth_counter_per_class = {}  # 用于计数每一类的GT框

    # 处理GT标签数据
    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))

        lines = txt_to_list(txt_file)  # 打开保存GT的txt文件，并转化为list

        bounding_boxes = []  # 用于存储bbox信息

        for line in lines:
            class_name, left, top, right, bottom = line.split()
            bbox = left + " " + top + " " + right + " " + bottom
            bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})

            if class_name in ground_truth_counter_per_class:
                ground_truth_counter_per_class[class_name] += 1  # 每一个GT框都参与一次计数
            else:
                ground_truth_counter_per_class[class_name] = 1

        with open(temp_files + "/" + file_id + "_ground_truth.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)  # 写入临时json文件

    gt_classes = list(ground_truth_counter_per_class.keys())  # 列出所有的种类
    n_classes = len(gt_classes)  # 类的数量

    detection_results_files_list = glob.glob(detection_results + '/*.txt')

    # 处理预测数据标签
    for class_name in gt_classes:  # 对每个种类进行处理
        bounding_boxes = []  # 用于存放bbox信息

        for txt_file in detection_results_files_list:
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))

            lines = txt_to_list(txt_file)

            for line in lines:
                tmp_class_name, confidence, left, top, right, bottom = line.split()  # 取出bbox框信息

                if tmp_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})

        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)  # 将框按照置信率由高到低排列
        with open(temp_files + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)  # 写入临时文件夹

    # GT框是按照图像名来存，输出预测框是按照类别来存

    sum_AP = 0.0
    ap_dictionary = {}

    for class_name in gt_classes:
        detection_results_file = temp_files + "/" + class_name + "_dr.json"
        detection_results_data = json.load(open(detection_results_file))

        num_detections = len(detection_results_data)  # 该类别下的预测框数量
        TP = [0] * num_detections   # 用于记录TP
        FP = [0] * num_detections  # 用于记录FP
        score = [0] * num_detections

        score_index = 0
        for index, detection in enumerate(detection_results_data):
            file_id = detection["file_id"]
            score[index] = float(detection["confidence"])
            if score[index] > MINOVERLAP:
                score_index = index

            ground_truth_file = temp_files + "/" + file_id + "_ground_truth.json"
            ground_truth_data = json.load(open(ground_truth_file))

            iou_max = 0
            ground_truth_match = -1
            bbox = [float(x) for x in detection["bbox"].split()]

            # 计算该框与所有图片中属于该类的GT框之间的IOU
            for obj in ground_truth_data:
                if obj["class_name"] == class_name:

                    bbox_gt = [float(x) for x in obj["bbox"].split()]
                    bbox_inter = [max(bbox[0], bbox_gt[0]), max(bbox[1], bbox_gt[1]), min(bbox[2], bbox_gt[2]), min(bbox[3], bbox_gt[3])]
                    inter_w = max(bbox_inter[2] - bbox_inter[0] + 1, 0)
                    inter_h = max(bbox_inter[3] - bbox_inter[1] + 1, 0)
                    
                    union = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1) + (bbox_gt[2] - bbox_gt[0] + 1) * (bbox_gt[3] - bbox_gt[1] + 1) - inter_w * inter_h
                    iou = inter_w * inter_h / union

                    # 计算并取出该类的最大IOU对应的框
                    if iou > iou_max:
                        iou_max = iou
                        ground_truth_match = obj

            min_overlap = MINOVERLAP
            if iou_max >= min_overlap:
                if not bool(ground_truth_match["used"]):
                    TP[index] = 1  # 将该框对应的TP置为1
                    ground_truth_match["used"] = True
                    with open(ground_truth_file, 'w') as f:
                        f.write(json.dumps(ground_truth_data))
                else:
                    FP[index] = 1
            else:
                FP[index] = 1

        cumsum = 0
        for index, val in enumerate(FP):
            FP[index] += cumsum
            cumsum += val

        cumsum = 0
        for index, val in enumerate(TP):
            TP[index] += cumsum
            cumsum += val

        recall = TP[:]
        for index, val in enumerate(TP):
            recall[index] = float(TP[index]) / ground_truth_counter_per_class[class_name]

        precision = TP[:]
        for index, val in enumerate(TP):
            precision[index] = float(TP[index]) / (FP[index] + TP[index])

        ap = VOC_ap(recall, precision)
        F1_score = np.array(recall) * np.array(precision) * 2 / np.where((np.array(precision) + np.array(recall)) == 0, 1, (np.array(precision) + np.array(recall)))

        sum_AP += ap
        text = class_name + " AP " + "= " + "{0:.2f}%".format(ap * 100)

        print(text + " ; score_threhold= " + '{}'.format(MINOVERLAP) + " ; F1=" + "{0:.2f}".format(F1_score[score_index]) + \
              " ; Recall=" + "{0:.2f}%".format(recall[score_index] * 100) + " ; Precision=" + "{0:.2f}%".format(precision[score_index] * 100))
        ap_dictionary[class_name] = ap

    mAP = sum_AP / n_classes
    text = "mAP = {0:.2f}%".format(mAP * 100)
    print(text)

    shutil.rmtree(temp_files)
