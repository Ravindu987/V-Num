import cv2


# Localize plate in given video frame
def localize(frame, dnn):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False
    )
    dnn.setInput(blob)
    output_layer_names = dnn.getUnconnectedOutLayersNames()
    layer_outputs = dnn.forward(output_layer_names)

    boxes = []
    confidences = []

    for output in layer_outputs:
        for detect in output:
            confidence = detect[5]
            if confidence > 0.7:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                if x > 0 and y > 0:
                    boxes.append([x, y, w, h])
                    confidences.append(round(float(confidence), 2))

    # Run non max suppresion on bounding boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    # Return only the boxes selected by NMS
    ret_boxes = []
    for i in indexes:
        ret_boxes.append((boxes[i], confidences[i]))

    return ret_boxes


# Predict bounding box and save to storage
def show_plate_video(frame, dnn, name_counter):
    boxes = localize(frame, dnn)
    if len(boxes) > 0:
        for box in boxes:
            x, y, w, h = box[0]
            confidence = str(box[1])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(
                frame,
                "license plate " + confidence,
                (x, y + h + 40),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 255, 255),
                1,
            )
            roi = frame[y : y + h, x : x + w]
            rand = "detect" + str(name_counter)
            print(x, y, w, h)
            cv2.imwrite("./Final_Product/cropped_plates/" + rand + ".jpg", roi)
        return True
    return False


# Default settings for plate detection
def run_default(cap, name_counter):

    # Set counters
    i = 0
    j = 201

    while True:

        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1080, 720))

        # Run detection algorithm once evry 10 frames,
        if i % 10 == 0 and j > 200:
            h, w, c = frame.shape
            if show_plate_video(frame[h // 2 :, :, :], dnn, name_counter):
                print(i, j)
                j = 1
                name_counter += 1

        cv2.imshow("frame", frame)

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

        i += 1
        j += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # Define relative path for weights and configuration file
    weight_path = "./YOLO localize plate/yolov4-train_final.weights"
    cfg_path = "./YOLO localize plate/yolov4-train.cfg"
    name_counter = 1

    # Read dnn from weights and config file
    dnn = cv2.dnn.readNet(weight_path, cfg_path)

    dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # cap = cv2.VideoCapture(
    #     "./DataSet/Videos/KIC-1_Lane-04_1_20211213183000_20211213190000.avi")

    cap = cv2.VideoCapture(
        "./DataSet/Videos/KIC-1_Lane-04_1_20211213080000_20211213083000.avi"
    )

    # show_on_video()
    run_default(cap, name_counter)
