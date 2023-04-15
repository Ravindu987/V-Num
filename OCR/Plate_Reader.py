import cv2 as cv
import numpy as np
import tensorflow as tf
from Letter_Contours import filter_contours_without_overlap
from keras.losses import SparseCategoricalCrossentropy


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


def get_characters(img):
    # Image Preprocessing
    # img = cv.copyMakeBorder(img, 5, 5, 5, 5, cv.BORDER_REFLECT)
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # blur = cv.GaussianBlur(gray, (3, 3), 0)
    # ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,15,4)
    thresh = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 19, 5
    )
    cv.imshow("Binary", thresh)
    cv.waitKey(0)

    noise = cv.fastNlMeansDenoising(gray, None, h=7, searchWindowSize=31)
    thresh2 = cv.adaptiveThreshold(
        noise, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 2
    )
    cv.imshow("Binary2", thresh2)
    cv.waitKey(0)

    erode_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    eroded = cv.erode(thresh2, erode_kernel)

    cv.imshow("Eroded", eroded)
    cv.waitKey(0)

    dilate_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    eroded = cv.dilate(eroded, dilate_kernel)

    cv.imshow("Eroded", eroded)
    cv.waitKey(0)

    # find contours of regions of interest within license plate
    try:
        contours, hierarchy = cv.findContours(
            eroded, cv.RETR_TREE, cv.CHAIN_APPROX_NONE
        )
    except:
        ret_img, contours, hierarchy = cv.findContours(
            thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE
        )

    sorted_contours = filter_contours_without_overlap(contours, hierarchy, img)
    letter_contours = []

    blank = np.zeros((img.shape[0], img.shape[1], 3))
    # Return all detected letters
    for contour in sorted_contours:
        cv.drawContours(blank, contour, -1, (0, 255, 0), 1)
        x, y, w, h = cv.boundingRect(contour)
        roi_image = img[y - 3 : y + h + 3, x - 3 : x + w + 3]
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


img = cv.imread("./Final_Product/cropped_plates/detect79.jpg")
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

if len(id) == 6:
    id = id[:2].replace("0", "O") + id[2:]
elif len(id) == 7:
    id = id[:3].replace("0", "O") + id[3:]
print(id)


# image_list = sorted(os.listdir("./Cropped License Plates"))

# for image_path in image_list:
# print(image_path)
# img = cv.imread("./Cropped License Plates/"+image_path)
# upsampled = sr.upsample(img)
# # img = tf.keras.utils.load_img( "./Cropped License Plates/"+filename, target_size=(256,256))
# cv.imshow("Test",upsampled)

# letters = get_letters(upsampled)
# id=""
# # print(letters)
# for im in letters:
#     cv.imshow("letter",im)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
#     ind = get_prediction(im)
#     print(classes[ind])
#     id+=classes[ind]
# print(id)
# cv.waitKey(0)
# cv.destroyAllWindows()
