import cv2 as cv
import numpy as np
import tensorflow as tf
from Letter_Contours import filter_contours_without_overlap
from keras.losses import SparseCategoricalCrossentropy
import os


# Get prediction for character
def get_prediction(img):
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    pred = ocr_model.predict(img_array)
    ind = np.argmax(pred[0])
    print(pred[0][ind])
    if pred[0][ind] >= 0.6:
        return ind
    else:
        return -1


def skew_correction(gray):
    edges = cv.Canny(gray, 50, 150)

    # Apply the probabilistic Hough Line Transform to detect lines in the edge image
    lines = cv.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
    )

    # Find the longest line
    longest_line_length = 0
    longest_line_angle = 0

    if not (lines is None):
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if line_length > longest_line_length:
                longest_line_length = line_length
                longest_line_angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, longest_line_angle, 1.0)
    rotated_image = cv.warpAffine(gray, rotation_matrix, (w, h))
    return rotated_image, longest_line_angle


def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv.warpAffine(img, rotation_matrix, (w, h))
    return rotated_image


def get_characters(img):
    # Image Preprocessing
    # img = cv.copyMakeBorder(img, 5, 5, 5, 5, cv.BORDER_REFLECT)
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # blur = cv.GaussianBlur(gray, (3, 3), 0)
    # ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,15,4)
    # thresh = cv.adaptiveThreshold(
    #     gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 19, 5
    # )
    # cv.imshow("Binary", thresh)
    # cv.waitKey(0)

    rotated, angle = skew_correction(gray)

    thresh2 = cv.adaptiveThreshold(
        rotated, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, 3
    )
    noise = cv.fastNlMeansDenoising(thresh2, None, h=7, searchWindowSize=31)

    cv.imshow("Noise reduced", noise)
    cv.waitKey(0)

    # erode_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # eroded = cv.erode(noise, erode_kernel)

    # cv.imshow("Eroded", eroded)
    # cv.waitKey(0)

    dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dilated = cv.dilate(noise, dilate_kernel)

    cv.imshow("Dilated", dilated)
    cv.waitKey(0)

    # eroded2 = cv.erode(dilated, erode_kernel)

    # kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    # dilated = cv.morphologyEx(noise, cv.MORPH_CLOSE, kernel)

    # cv.imshow("Eroded", eroded2)
    # cv.waitKey(0)

    # find contours of regions of interest within license plate
    try:
        contours, hierarchy = cv.findContours(
            dilated, cv.RETR_TREE, cv.CHAIN_APPROX_NONE
        )
    except:
        ret_img, contours, hierarchy = cv.findContours(
            dilated, cv.RETR_TREE, cv.CHAIN_APPROX_NONE
        )

    sorted_contours = filter_contours_without_overlap(contours, hierarchy, img)

    img = rotate_image(img, angle)
    blank = np.zeros((img.shape[0], img.shape[1], 3))

    letter_contours = []

    # Return all detected letters
    for contour in sorted_contours:
        cv.drawContours(blank, contour, -1, (0, 255, 0), 1)
        x, y, w, h = cv.boundingRect(contour)
        print(x, y, w, h)
        if y >= 3 and x >= 3 and y + h < img.shape[0] and x + w < img.shape[1]:
            roi_image = img[y - 3 : y + h + 3, x - 3 : x + w + 3]
        else:
            roi_image = img[y : y + h, x : x + w]
        roi_image = cv.resize(roi_image, dsize=(128, 128), interpolation=cv.INTER_CUBIC)
        letter_contours.append(roi_image)
    cv.imshow("Co", blank)
    cv.waitKey(0)

    return letter_contours


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
    "./Character Recognition Weights/model_on_target_data_7.hdf5", compile=False
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

img = cv.imread("./Final_Product/cropped_plates/detect199.jpg")
# img = cv.imread("./Cropped License Plates/Video 19/detect60.jpg")
cv.imshow("Original", img)
upsampled = sr.upsample(img)
cv.imshow("Plate-Upsampled", upsampled)
cv.waitKey(0)


letters = get_characters(upsampled)
id = ""
for im in letters:
    cv.imshow("letter", im)
    cv.waitKey(0)
    cv.destroyAllWindows()
    ind = get_prediction(im)
    if not ind == -1:
        print(classes[ind])
        id += classes[ind]

length = len(id)

if length > 0:
    if length == 6:
        id = id[:2].replace("0", "O") + id[2:]
        id = id[:2].replace("1", "I") + id[2:]
        id = id[:2].replace("8", "B") + id[2:]
    elif length == 7:
        id = id[:3].replace("0", "O") + id[3:]
        id = id[:3].replace("1", "I") + id[3:]
        id = id[:3].replace("8", "B") + id[3:]

    if id[-1] in LETTERS:
        id = id[:-1]

    if length > 1:
        try:
            if id[1] in LETTERS and id[0] in DIGITS:
                id = id[1:]
        except:
            id = id

print(id)

# image_list = sorted(os.listdir("./Final_Product/cropped_plates/"))
# print(image_list)
# for image_path in image_list:
#     if image_path.lower().endswith(".jpg"):
#         print(image_path)
#         img = cv.imread("./Final_Product/cropped_plates/" + image_path)
#         cv.imshow("Original", img)
#         upsampled = sr.upsample(img)
#         cv.imshow("Plate-Upsampled", upsampled)
#         cv.waitKey(0)

#         letters = get_characters(upsampled)
#         id = ""
#         for im in letters:
#             cv.imshow("letter", im)
#             cv.waitKey(0)
#             cv.destroyAllWindows()
#             ind = get_prediction(im)
#             if not ind == -1:
#                 print(classes[ind])
#                 id += classes[ind]

#         if len(id) == 6:
#             id = id[:2].replace("0", "O") + id[2:]
#         elif len(id) == 7:
#             id = id[:3].replace("0", "O") + id[3:]
#         print(id)
