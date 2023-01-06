import cv2 as cv
import numpy as np
import os
import tensorflow as tf
from Letter_Contours import filter_contours, x_cord_contour


def get_prediction(img):
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    pred = ocr_model.predict(img_array)
    ind = np.argmax(pred[0])
    return ind

def get_letters(img):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # perform gaussian blur to smoothen image
    # blur = cv.GaussianBlur(gray, (3,3), 0)
    # threshold the image using Otsus method to preprocess for tesseract
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 17, 0)
    # create rectangular kernel for dilation
    # rect_kern = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    # apply dilation to make regions more clear
    # dilation = cv.dilate(thresh, rect_kern, iterations = 2)
    cv.imshow("Dilated",thresh)
    cv.waitKey(0)
    # find contours of regions of interest within license plate
    try:
        contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    except:
        ret_img, contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

    sorted_contours = filter_contours(contours, img)

    letter_contours = []

    for contour in sorted_contours:
        x,y,w,h = cv.boundingRect(contour)
        blank = np.zeros((img.shape[0],img.shape[1],3), dtype='uint8') + 255
        cv.drawContours(blank, contour, -1, (0,0,0), 1)
        roi_image = img[y-3:y+h+3,x-3:x+w+3]
        roi_image = cv.resize(roi_image, dsize=(128,128), interpolation=cv.INTER_CUBIC)
        letter_contours.append(roi_image)
        # cv.imshow('Contours blank', blank)
        # cv.waitKey(0) 
        # cv.destroyAllWindows()

    return letter_contours

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

ocr_model = tf.keras.models.load_model('./CNN letter Dataset/model_mixed_1.hdf5')

sr = cv.dnn_superres.DnnSuperResImpl_create()
 
path = "./OCR/EDSR_x3.pb"
sr.readModel(path)
sr.setModel("edsr",3)

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

img = cv.imread("./Final_Product/cropped_plates/detect4.jpg")
upsampled = sr.upsample(img)
cv.imshow("Plate",upsampled)
cv.waitKey(0)
# img = tf.keras.utils.load_img( "./Cropped License Plates/"+filename, target_size=(256,256))
# cv.imshow("Test",upsampled)

letters = get_letters(upsampled)
id=""
# print(letters)
for im in letters:
    cv.imshow("letter",im)
    cv.waitKey(0)
    cv.destroyAllWindows()
    ind = get_prediction(im)
    print(classes[ind])
    id+=classes[ind]
print(id)
# cv.waitKey(0)
# cv.destroyAllWindows()