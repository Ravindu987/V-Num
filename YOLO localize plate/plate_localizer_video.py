import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


def localize_video(frame, dnn):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(
        frame, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
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


def show_plate_video(frame, dnn):
    indexes, boxes, confidences = localize_video(frame, dnn)
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            confidence = str(round(confidences[i], 2))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(frame, "license plate " + confidence,
                        (x, y + h + 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)
        coordinates = (x, y, w, h)
    return


# Define relative path for weights and configuration file
weight_path = "./yolov3-train_final.weights"
cfg_path = "./yolov3-train.cfg"

# Read dnn from weights and config file
dnn = cv2.dnn.readNet(weight_path, cfg_path)

dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

cap = cv2.VideoCapture(
    "../DataSet/Videos/KIC-1_Lane-04_1_20211213070000_20211213073000.avi")

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1080, 720))
    show_plate_video(frame, dnn)
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
