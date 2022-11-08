import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


class LicensePlateDetector:
    def __init__(self, pth_weights: str, pth_cfg: str):
        self.net = cv2.dnn.readNet(pth_weights, pth_cfg)
        self.classes = ["license-plate"]
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.color = (255, 0, 0)
        self.coordinates = None
        self.img = None
        self.fig_image = None
        self.roi_image = None

    def detect(self, img_path: str):
        orig = cv2.imread(img_path)
        self.img = orig
        img = orig.copy()
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(
            img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        output_layer_names = self.net.getUnconnectedOutLayersNames()
        layer_outputs = self.net.forward(output_layer_names)
        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                cv2.rectangle(img, (x, y), (x + w, y + h), self.color, 3)
                cv2.putText(img, label + ' ' + confidence,
                            (x, y + h + 40), self.font, 2, (255, 255, 255), 1)
            self.fig_image = img
            self.coordinates = (x, y, w, h)
        return

    def crop_plate(self):
        x, y, w, h = self.coordinates
        roi = self.img[y:y + h, x:x + w]
        self.roi_image = roi
        return


def crop_all(images):
    i = 0
    for image in images:
        lpd.detect(image)
        lpd.crop_plate()
        plt.figure(figsize=(10, 4))
        plt.imshow(cv2.cvtColor(lpd.roi_image, cv2.COLOR_BGR2RGB))
        plt.savefig('Crop/detected'+str(i)+'.jpg')
        i += 1


def show_plate_full(images):
    for image in images:
        lpd.detect(image)
        plt.figure(figsize=(24, 24))
        plt.imshow(cv2.cvtColor(lpd.fig_image, cv2.COLOR_BGR2RGB))
        plt.show()


def show_plate_cropped(images):
    for image in images:
        lpd.detect(image)
        lpd.crop_plate()
        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(lpd.roi_image, cv2.COLOR_BGR2RGB))
        plt.show()


lpd = LicensePlateDetector(
    pth_weights="./yolov3-train_final.weights",
    pth_cfg="./yolov3-train.cfg"
)

image_files = [file for file in glob.glob(
    'D:\Sem 5\Project\V-Num\DataSet\Images\*.jpg')]

image_files.sort()

# crop_all(image_files)
show_plate_full(image_files)
# show_plate_cropped(image_files)
