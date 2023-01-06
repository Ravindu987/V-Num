import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Localize plate in given video frame
def localize(frame, dnn):
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
            if confidence > 0.7:
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


# Predict bounding box
def show_plate_video(frame, dnn, name_counter):
    indexes, boxes, confidences = localize(frame, dnn)
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            confidence = str(round(confidences[i], 2))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(frame, "license plate " + confidence,
                        (x, y + h + 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)
            roi = frame[y:y+h,x:x+w]
            rand = "detect"+str(name_counter)
            cv2.imwrite("./Final_Product/cropped_plates/"+rand+".jpg",roi)
        return True
    return False



def run_default(cap, name_counter):
    i=0
    j=201
    while (True):
        i+=1
        j+=1
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1080, 720))
        h,w,c = frame.shape
        roi_frame = frame[0:h,w//5:4*w//5]

        if (i % 10 == 0 and j>300):
            if show_plate_video(frame, dnn, name_counter):
                print(i,j)
                j=1
                name_counter+=1

        cv2.imshow('frame', frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    # Define relative path for weights and configuration file
    weight_path = "./YOLO localize plate/yolov4-train_final.weights"
    cfg_path = "./YOLO localize plate/yolov4-train.cfg"
    name_counter = 1

    # Read dnn from weights and config file
    dnn = cv2.dnn.readNet(weight_path, cfg_path)

    dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    cap = cv2.VideoCapture(
        "./DataSet/Videos/KIC-1_Lane-04_1_20211213073000_20211213080000.avi")

    # show_on_video()
    # run_multi_configs(cap)
    run_default(cap, name_counter)


