import cv2 as cv
import numpy as np

img = cv.imread('Photos/text.png')
blank = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_canny = cv.Canny( img, 125, 175)

cv.imshow('Gray', img_canny)
contours, hierarchies = cv.findContours( img_canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(len(contours))

cv.drawContours( blank, contours, -1, 255, 1)

cv.imshow('Original', img)
cv.imshow("Contours", blank)

cv.waitKey(0)