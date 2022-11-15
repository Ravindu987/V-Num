import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


def localize(imgpath, dnn):
    img = cv2.imread(imgpath)
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(
        img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    dnn.setInput(blob)
    output_layer_names = dnn.getUnconnectedOutLayersNames()
    layer_outputs = dnn.forward(output_layer_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = float(detect[0] * width )
                center_y = float(detect[1] *height )
                w = float(detect[2] * width )
                h = float(detect[3] * height)
                x = float(center_x - w / 2)
                y = float(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    return indexes, boxes, confidences


def get_plate_coordinates(imgpath, dnn):
    indexes, boxes, confidences = localize(imgpath, dnn)
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            confidence = str(round(confidences[i], 2))
        coordinates = (x, y, w, h)
    return coordinates



def intersection_over_union(gt_box, pred_box):
    inter_box_top_left = [max(gt_box[0], pred_box[0]), max(gt_box[1], pred_box[1])]
    inter_box_bottom_right = [min(gt_box[0]+gt_box[2], pred_box[0]+pred_box[2]), min(gt_box[1]+gt_box[3], pred_box[1]+pred_box[3])]

    inter_box_w = inter_box_bottom_right[0] - inter_box_top_left[0]
    inter_box_h = inter_box_bottom_right[1] - inter_box_top_left[1]

    intersection = inter_box_w * inter_box_h
    union = gt_box[2] * gt_box[3] + pred_box[2] * pred_box[3] - intersection

    iou = intersection / union
    return iou, intersection, union


def mean_iou(image_paths,txt_paths,dnn):
    ious= []
    for i in range(len(image_paths)):
        prediction = get_plate_coordinates(image_paths[i], dnn)
        prediction = [round(val,5) for val in prediction]
        print(prediction)

        img = cv2.imread(image_paths[i])
        height, width, _ = img.shape
        label = open(txt_paths[i]).readline().split()
        label = [ float(x) for x in label]
        # print(label)
        center_x = float(label[1] * width )
        center_y = float(label[2] *height )
        w = float(label[3] * width )
        h = float(label[4] * height)
        x = float(center_x - w / 2)
        y = float(center_y - h / 2)
        gt_box = [x,y,w,h]
        print(gt_box)

        iou, intersection, union = intersection_over_union(gt_box, prediction)
        print(iou)
        ious.append(iou)
    return sum(ious)/len(ious)

    

# Define relative path for weights and configuration file
weight_path = "./yolov3-train_final.weights"
cfg_path = "./yolov3-train.cfg"


# Get image paths
image_paths = [file for file in glob.glob(
    '../YOLO train data/*.jpg')]
txt_paths = [file for file in glob.glob('../YOLO train data/*.txt') if file != '../YOLO train data/classes.txt']
image_paths.sort()
txt_paths.sort()
# print(image_paths)
# print(txt_paths)

# Read dnn from weights and config file
dnn = cv2.dnn.readNet(weight_path, cfg_path)


mean_iou = mean_iou(image_paths, txt_paths, dnn)

print(mean_iou)