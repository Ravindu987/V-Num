import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


def localize(imgpath, dnn):
    img = cv2.imread(imgpath)
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(
        img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False
    )
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
            cv2.putText(
                temp_img,
                "license plate " + confidence,
                (x, y + h + 40),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 255, 255),
                1,
            )
        coordinates = (x, y, w, h)
    plt.figure(figsize=(24, 24))
    plt.imshow(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))
    plt.show()
    return


def show_plate_with_orig(imgpath, txtpath, dnn):
    img = cv2.imread(imgpath)
    height, width, _ = img.shape
    temp_img = img.copy()
    indexes, boxes, confidences = localize(imgpath, dnn)
    label = open(txtpath).readline().split()
    label = [float(x) for x in label]
    print(label)
    center_x = int(label[1] * width)
    center_y = int(label[2] * height)
    w = int(label[3] * width)
    h = int(label[4] * height)
    x = int(center_x - w / 2)
    y = int(center_y - h / 2)
    gt_box = [x, y, w, h]
    print(gt_box)

    cv2.rectangle(
        temp_img,
        (int(gt_box[0]), int(gt_box[1])),
        (int(gt_box[0]) + int(gt_box[2]), int(gt_box[1]) + int(gt_box[3])),
        (255, 0, 0),
        2,
    )
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            confidence = str(round(confidences[i], 2))
            cv2.rectangle(temp_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(
                temp_img,
                "license plate " + confidence,
                (x, y + h + 40),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 255, 255),
                1,
            )
        coordinates = (x, y, w, h)

    prediction = get_plate_coordinates(imgpath, dnn)
    prediction = [round(val, 5) for val in prediction]
    iou, intersection, union = intersection_over_union(gt_box, prediction)
    print(iou, intersection, union)

    plt.figure(figsize=(24, 24))
    plt.imshow(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))
    plt.show()
    return


def get_plate_coordinates(imgpath, dnn):
    indexes, boxes, confidences = localize(imgpath, dnn)
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            confidence = str(round(confidences[i], 2))
        coordinates = (x, y, w, h)
        if coordinates != None:
            return coordinates
        else:
            return (0, 0, 0, 0)
    return (0, 0, 0, 0)


def show_plate_all(image_paths, dnn):
    for image_path in image_paths:
        show_plate(image_path, dnn)


def show_plate_all_with_orig(image_paths, text_paths, dnn):
    for i in range(len(image_paths)):
        show_plate_with_orig(image_paths[i], text_paths[i], dnn)


def intersection_over_union(gt_box, pred_box):
    inter_box_top_left = [max(gt_box[0], pred_box[0]), max(gt_box[1], pred_box[1])]
    inter_box_bottom_right = [
        min(gt_box[0] + gt_box[2], pred_box[0] + pred_box[2]),
        min(gt_box[1] + gt_box[3], pred_box[1] + pred_box[3]),
    ]

    inter_box_w = inter_box_bottom_right[0] - inter_box_top_left[0]
    inter_box_h = inter_box_bottom_right[1] - inter_box_top_left[1]

    intersection = inter_box_w * inter_box_h
    union = gt_box[2] * gt_box[3] + pred_box[2] * pred_box[3] - intersection

    iou = intersection / union
    # print(inter_box_top_left,inter_box_bottom_right,inter_box_w,inter_box_h)
    return iou, intersection, union


def iou_all(image_paths, txt_paths, dnn):
    for i in range(len(image_paths)):
        prediction = get_plate_coordinates(image_paths[i], dnn)
        prediction = [round(val, 5) for val in prediction]
        print(prediction)

        img = cv2.imread(image_paths[i])
        height, width, _ = img.shape
        label = open(txt_paths[i]).readline().split()
        label = [float(x) for x in label]
        print(label)
        center_x = int(label[1] * width)
        center_y = int(label[2] * height)
        w = int(label[3] * width)
        h = int(label[4] * height)
        x = int(center_x - w / 2)
        y = int(center_y - h / 2)
        gt_box = [x, y, w, h]
        print(gt_box)

        iou, intersection, union = intersection_over_union(gt_box, prediction)
        print(iou, intersection, union)


def crop_all(image_paths, dnn):
    i = 1
    for image_path in image_paths:
        (x, y, w, h) = get_plate_coordinates(image_path, dnn)
        if (x, y, w, h) != (0, 0, 0, 0):
            img = cv2.imread(image_path)
            img_copy = img.copy()
            cropped = img_copy[y : y + h, x : x + w]
            cv2.imwrite("./Cropped License Plates/detected" + str(i) + ".jpg", cropped)
            i += 1


# Define relative path for weights and configuration file
weight_path = "./YOLO localize plate/yolov4-train_final.weights"
cfg_path = "./YOLO localize plate/yolov4-train.cfg"


# Get image paths
image_paths = [file for file in glob.glob("./YOLO Test data/*.jpg")]
txt_paths = [file for file in glob.glob("./YOLO Test data/*.txt")]
image_paths.sort()
txt_paths.sort()
# print(image_paths)
# print(txt_paths)

# Read dnn from weights and config file
dnn = cv2.dnn.readNet(weight_path, cfg_path)

# crop_all(image_paths, dnn)
# show_plate_all(image_paths, dnn)
# show_plate_cropped(image_files)
iou_all(image_paths, txt_paths, dnn)
# show_plate_all_with_orig(image_paths, txt_paths, dnn)
