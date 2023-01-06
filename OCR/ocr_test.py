import tensorflow as tf
import cv2
import numpy as np
import os


def get_prediction(img):
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    pred = ocr_model.predict(img_array)
    ind = np.argmax(pred[0])
    return ind

if __name__=="__main__":
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    ocr_model = tf.keras.models.load_model('./Train Digit Data/model.hdf5')

    images = []

    for filename in os.listdir("./Cropped_Letters/Predict"):
        print(filename)
        img = tf.keras.utils.load_img( "./Cropped_Letters/Predict/"+filename, target_size=(256,256))
        im = cv2.imread("./Cropped_Letters/Predict/"+filename)
        im= cv2.resize(im,dsize=(256,256), interpolation = cv2.INTER_CUBIC)
        cv2.imshow("Test",im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ind = get_prediction(im)
        print(ind)

# img = cv2.imread("./Cropped_Letters/Predict/Ld755.jpg")
# img = cv2.resize(img, (256,256))
# img = img/255
# images.append(img)
# cv2.imshow("Test",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# stack = np.stack(images)
# pred = ocr_model.predict(stack)
# print(len(pred[0]))
# print(pred)
# for x in pred:
#     ind = np.argmax(x)
#     print(ind)
#     print(classes[ind])
# cv2.waitKey(0)
# cv2.destroyAllWindows()