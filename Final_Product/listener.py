from time import sleep
import cv2 as cv
import tensorflow as tf
import numpy as np
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer


# Write text to file
def WriteToFile(filename, text):
    file = open(filename, "a")
    file.write(text+"\n")


def x_cord_contour(contours):
    M = cv.moments(contours)
    return (int(M['m10']/M['m00']))


# Filter contours and sort for characters
def filter_contours(contours, img):
    filtered_contours = []

    for contour in contours:
        x,y,w,h = cv.boundingRect(contour)
        height, width, channels = img.shape
        if ( h/height < 0.3 or w/width<0.05 or h/height>0.8 or w/width>0.4):
            continue
        else:
            filtered_contours.append(contour)
        
    sorted_contours = sorted(filtered_contours, key = x_cord_contour, reverse = False)

    return sorted_contours


# Get cropped letters from image
def get_letters(img):

    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    ret, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    try:
        contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    except:
        ret_img, contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

    sorted_contours = filter_contours(contours, img)

    letter_contours = []

    for contour in sorted_contours:
        x,y,w,h = cv.boundingRect(contour)
        roi_image = img[y-3:y+h+3,x-3:x+w+3]
        roi_image = cv.resize(roi_image, dsize=(128,128), interpolation=cv.INTER_CUBIC)
        letter_contours.append(roi_image)

    return letter_contours

# Get prediction from cropped letter
def get_prediction(img):
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    pred = ocr_model.predict(img_array)
    ind = np.argmax(pred[0])
    return ind


# Get plate image and write ID to file
def plate_read(img_path):

        print(img_path)
        img = cv.imread(img_path)
        upsampled = sr.upsample(img)

        letters = get_letters(upsampled)
        id=""

        for im in letters:
            ind = get_prediction(im)
            id+=classes[ind]
        print(id)

        WriteToFile("./Final_Product/cropped_plates/Plates.txt", id)


# Listener class
class MonitorFolder(PatternMatchingEventHandler):

    def on_created(self, event):
        plate_read(event.src_path)


if __name__=="__main__":

    folder_path = "./Final_Product/cropped_plates"
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

    ocr_model = tf.keras.models.load_model('./Character Recognition Weights/model_mixed_1.hdf5')

    sr = cv.dnn_superres.DnnSuperResImpl_create()
    
    path = "./OCR/EDSR_x3.pb"
    sr.readModel(path)
    sr.setModel("edsr",3)

    event_handler = MonitorFolder((["*.jpg"]))
    observer = Observer()
    observer.schedule(event_handler, folder_path, True)
    observer.start()
    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()

