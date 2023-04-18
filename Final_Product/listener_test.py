from time import sleep
import cv2 as cv
import tensorflow as tf
import numpy as np
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer
from keras.losses import SparseCategoricalCrossentropy
import os


# Write text to file
def WriteToFile(filename1, filename2, imgpath, id):
    file1 = open(filename1, "a+")
    file2 = open(filename2, "a+")
    file1.seek(0)
    file2.seek(0)
    last_entry = ""
    lines = file2.readlines()
    if len(lines) != 0:
        last_entry = lines[-1]
    if id not in last_entry:
        file1.write(imgpath + " : " + id + "\n")
        file2.write(id + "\n")


def sort_contours(contours):
    contours_boxes = [list(cv.boundingRect(contour)) for contour in contours]

    for i in range(len(contours_boxes)):
        contours_boxes[i].append(i)

    c = np.array(contours_boxes)
    max_height = np.max(c[:, 3])

    # Sort the contours by y-value
    by_y = sorted(contours_boxes, key=lambda x: x[1])  # y values

    line_y = by_y[0][1]  # first y
    line = 1
    by_line = []

    # Assign a line number to each contour
    for x, y, w, h, i in by_y:
        if y > line_y + 2 * max_height / 3:
            line_y = y
            line += 1

        by_line.append((line, x, y, w, h, i))

    # This will now sort automatically by line then by x
    return [i for line, x, y, w, h, i in sorted(by_line)]


# Filter contours with size ratio to drop too small and too large contours and
# eliminate contours with overlap
def filter_contours_without_overlap(contours, hierarchy, img):
    filtered_contours = []
    indexes = []

    for i in range(len(contours)):
        x, y, w, h = cv.boundingRect(contours[i])
        height, width, channels = img.shape
        if width / height > 2:
            if (
                h / height < 0.3
                or w / width < 0.05
                or h / height > 0.8
                or w / width > 0.15
            ):
                continue
            elif h / height < 0.325 and w / width < 0.325:
                continue
            elif y > height * 0.6:
                continue
            else:
                print(h / height, w / width)
                # filtered_contours.append(contours[i])
                indexes.append(i)
        else:
            if (
                h / height < 0.2
                or w / width < 0.1
                or h / height > 0.6
                or w / width > 0.4
            ):
                continue
            elif h / height < 0.325 and w / width < 0.325:
                continue
            elif y > height * 0.7:
                continue
            else:
                print(h / height, w / width)
                # filtered_contours.append(contours[i])
                indexes.append(i)

    for index in indexes:
        if hierarchy[0][index][3] in indexes:
            continue
        else:
            filtered_contours.append(contours[index])

    sorted_contours_indexes = sort_contours(filtered_contours)

    sorted_contours = [filtered_contours[i] for i in sorted_contours_indexes]

    return sorted_contours


# Get cropped letters from image
def get_letters(img):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # ret, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,15,4)
    noise_reduced = cv.fastNlMeansDenoising(gray, None, h=7, searchWindowSize=31)
    thresh = cv.adaptiveThreshold(
        noise_reduced, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 2
    )

    erode_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    eroded = cv.erode(thresh, erode_kernel)

    dilate_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    dilated = cv.dilate(eroded, dilate_kernel)

    try:
        contours, hierarchy = cv.findContours(
            dilated, cv.RETR_TREE, cv.CHAIN_APPROX_NONE
        )
    except:
        ret_img, contours, hierarchy = cv.findContours(
            thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE
        )

    sorted_contours = filter_contours_without_overlap(contours, hierarchy, img)

    letter_contours = []

    for contour in sorted_contours:
        x, y, w, h = cv.boundingRect(contour)
        roi_image = img[y - 3 : y + h + 3, x - 3 : x + w + 3]
        roi_image = cv.resize(roi_image, dsize=(128, 128), interpolation=cv.INTER_CUBIC)
        letter_contours.append(roi_image)

    return letter_contours


# Get prediction from cropped letter
def get_prediction(img):
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    pred = ocr_model.predict(img_array)
    ind = np.argmax(pred[0])
    if pred[0][ind] >= 0.6:
        return ind
    else:
        return -1


# Get plate image and write ID to file
def plate_read(img_path):
    print(img_path)
    img = cv.imread(img_path)
    upsampled = sr.upsample(img)

    letters = get_letters(upsampled)
    id = ""

    for im in letters:
        ind = get_prediction(im)
        if ind != -1:
            id += classes[ind]

    if id == "":
        return

    if len(id) == 6:
        id = id[:2].replace("0", "O") + id[2:]
        id = id[:2].replace("1", "I") + id[2:]
    elif len(id) == 7:
        id = id[:3].replace("0", "O") + id[3:]
        id = id[:3].replace("1", "I") + id[3:]

    if id[-1] in LETTERS:
        id = id[:-1]

    if id[1] in LETTERS and id[0] in DIGITS:
        id = id[1:]

    print(id)

    WriteToFile(
        "./Final_Product/Test_Plates_Only.txt",
        "./Final_Product/Test_Plates.txt",
        img_path,
        id,
    )


def numerical_sort_key(file_name):
    # extract the numeric portion of the file name
    numeric_part = file_name.split(".")[0][6:]
    # convert the numeric part to an integer
    return int(numeric_part)


# Listener class
class MonitorFolder(PatternMatchingEventHandler):
    def on_created(self, event):
        plate_read(event.src_path)


if __name__ == "__main__":
    folder_path = "./Final_Product/cropped_plates"
    classes = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ]

    LETTERS = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ]

    DIGITS = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]

    ocr_model = tf.keras.models.load_model(
        "./Character Recognition Weights/model_on_target_data_2.hdf5", compile=False
    )

    ocr_model.compile(
        optimizer="adam",
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    sr = cv.dnn_superres.DnnSuperResImpl_create()

    path = "./OCR/EDSR_x3.pb"
    sr.readModel(path)
    sr.setModel("edsr", 3)

    img_list = [
        f for f in os.listdir("./Cropped License Plates/Video 19") if f.endswith(".jpg")
    ]
    img_list.sort(key=numerical_sort_key)
    for img_name in img_list:
        img_path = os.path.join("./Cropped License Plates/Video 19/", img_name)
        plate_read(img_path)

        if img_name == "detect115.jpg":
            break
