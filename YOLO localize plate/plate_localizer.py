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
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    return indexes, boxes, confidences


def show_plate(imgpath, dnn):
    img = cv2.imread(imgpath)
    temp_img = img.copy()
    indexes, boxes, confidences = localize(imgpath, dnn)
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            confidence = str(round(confidences[i], 2))
            cv2.rectangle(temp_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(temp_img, "license plate " + confidence,
                        (x, y + h + 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)
        coordinates = (x, y, w, h)
    plt.figure(figsize=(24, 24))
    plt.imshow(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))
    plt.show()
    return


def show_plate_all(image_paths, dnn):
    for image_path in image_paths:
        show_plate(image_path, dnn)


# Define relative path for weights and configuration file
weight_path = "./yolov3-train_final.weights"
cfg_path = "./yolov3-train.cfg"


# Get image paths
image_paths = [file for file in glob.glob(
    'D:\Sem 5\Project\V-Num\DataSet\Images\*.jpg')]
image_paths.sort()

# Read dnn from weights and config file
dnn = cv2.dnn.readNet(weight_path, cfg_path)

# crop_all(image_files)
show_plate_all(image_paths, dnn)
# show_plate_cropped(image_files)
