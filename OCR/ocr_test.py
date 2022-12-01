import tensorflow as tf
import cv2
import numpy as np
import os

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

ocr_model = tf.keras.models.load_model('./Cropped_Letters/DONE/splitted/model.hdf5')

images = []


for filename in os.listdir("./Cropped_Letters/Predict"):
    img = cv2.imread("./Cropped_Letters/Predict/"+filename)
    img = cv2.resize(img, (256,256))
    img = img/255
    images.append(img)


stack = np.stack(images)
pred = ocr_model.predict(stack)
print(len(pred[0]))
print(pred)
for x in pred:
    ind = np.argmax(x)
    print(ind)
    print(classes[ind])
cv2.waitKey(0)
cv2.destroyAllWindows()