import cv2
import matplotlib.pyplot as plt
import time


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

    for output in layer_outputs:
        for detect in output:
            confidence = detect[5]
            if confidence > 0.2:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(round(float(confidence), 2))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    ret_boxes = []
    for i in indexes:
        ret_boxes.append((boxes[i], confidences[i]))

    return ret_boxes


# Predict bounding box
def draw_rectangle(frame, dnn):
    boxes = localize(frame, dnn)
    for box in boxes:
        x, y, w, h = box[0]
        confidence = str(box[1])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(frame, "license plate " + confidence,
                    (x, y + h + 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)
    return


# Run detections on multiple configurations and show results
def run_multi_configs(cap):
    fps_values = []

    # Run for several detection frequencies
    for j in range(4):

        start = time.time()

        if j == 1:
            freq = 1
        elif j == 2:
            freq = 4
        elif j == 3:
            freq = 10

        fps_readings = []

        i = 1  # Count frames

        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1080, 720))

            # Run detection according to relevant frequencies
            if j != 0:
                if (i % freq == 0):
                    draw_rectangle(frame, dnn)

            cv2.imshow('frame', frame)

            # Calculate FPS value every 100
            if (i % 100 == 0):
                fps = i/(time.time()-start)
                print(fps)
                fps_readings.append(fps)

            # Run with new frequency after 1000 frames
            if i == 500:
                break
            i += 1

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        fps_values.append(fps_readings)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    # Plot FPS values
    num_frames = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    plt.plot(num_frames, fps_values[0], label="No detection")
    plt.plot(num_frames, fps_values[1], label="Detection every frame")
    plt.plot(num_frames, fps_values[2], label="Detection every 4 frames")
    plt.plot(num_frames, fps_values[3], label="Detection every 10 frames")
    plt.xlabel("Number of frames")
    plt.ylabel("Frame rate")
    plt.legend(bbox_to_anchor=(0.65, 0.9))
    plt.show()


# Run detection every 100 frames
def run_default(cap):
    i = 0
    while (True):
        i += 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1080, 720))

        if (i % 100 == 0):
            t = time.time()
            draw_rectangle(frame, dnn)
            print(time.time() - t)

        cv2.imshow('frame', frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break


# Run detections every frame
def run_each(cap):
    i = 0
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1080, 720))

        draw_rectangle(frame, dnn)

        cv2.imshow('frame', frame)
        i += 1

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Define relative path for weights and configuration file
    weight_path = "./YOLO localize plate/yolov4-train_final.weights"
    cfg_path = "./YOLO localize plate/yolov4-train.cfg"

    # Read dnn from weights and config file
    dnn = cv2.dnn.readNet(weight_path, cfg_path)

    dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    cap = cv2.VideoCapture(
        "./DataSet/Videos/KIC-1_Lane-04_1_20211213073000_20211213080000.avi")

    # run_each(cap)
    # run_multi_configs(cap)
    run_default(cap)
