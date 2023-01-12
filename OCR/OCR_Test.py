import tensorflow as tf
import cv2
import numpy as np
import os


# Get prediction from image using the model
def get_prediction(img):
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    pred = ocr_model.predict(img_array)
    ind = np.argmax(pred[0])
    return ind


if __name__ == "__main__":
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
               'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    ocr_model = tf.keras.models.load_model(
        './Character Recognition Weights/model_mixed_9.hdf5')

    images = []

    for filename in os.listdir("./Cropped Letters/Remaining after sort 1"):
        im = cv2.imread("./Cropped Letters/Remaining after sort 1/"+filename)
        im = cv2.resize(im, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Test", im)
        cv2.waitKey(0)
        ind = get_prediction(im)
        print(ind)
        cv2.destroyAllWindows()

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
