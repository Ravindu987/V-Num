import cv2 as cv
import numpy as np
import os


def x_cord_contour(contours):
    M = cv.moments(contours)
    return (int(M['m10']/M['m00']))


def filter_contours(contours, img):
    filtered_contours = []

    for contour in contours:
        x,y,w,h = cv.boundingRect(contour)
        # print(x,y,w,h)
        height, width, channels = img.shape
        # print(h/height, w/width)
        if ( h/height < 0.3 or w/width<0.05 or h/height>0.8 or w/width>0.4):
            continue
        else:
            filtered_contours.append(contour)
        
    sorted_contours = sorted(filtered_contours, key = x_cord_contour, reverse = False)

    return sorted_contours

def draw_contours(img, path):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # resize image to three times as large as original for better readability
    # gray = cv.resize(gray, None, fx = 1.5, fy = 1.5, interpolation = cv.INTER_CUBIC)
    # perform gaussian blur to smoothen image
    blur = cv.GaussianBlur(gray, (3,3), 0)
    #cv.imshow("Gray", gray)
    #cv.waitKey(0)
    # threshold the image using Otsus method to preprocess for tesseract
    ret, thresh = cv.threshold(blur, 100, 255, cv.THRESH_BINARY_INV)
    #cv.imshow("Otsu Threshold", thresh)
    #cv.waitKey(0)
    # create rectangular kernel for dilation
    rect_kern = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    # apply dilation to make regions more clear
    dilation = cv.dilate(thresh, rect_kern, iterations = 2)
    # cv.imshow("Dilation", dilation)
    # cv.waitKey(0)
    # find contours of regions of interest within license plate
    try:
        contours, hierarchy = cv.findContours(dilation, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    except:
        ret_img, contours, hierarchy = cv.findContours(dilation, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

    print(len(contours))

    sorted_contours = filter_contours(contours, img)

    print(len(sorted_contours))

    # for contour in sorted_contours:
    #     blank = np.zeros((img.shape[0],img.shape[1],3), dtype='uint8')
    #     cv.drawContours(blank, contour, -1, (0,255,0), 1)
    #     cv.imshow('Contours blank', blank)
    #     cv.waitKey(0) 
    #     cv.destroyAllWindows()

    blank = np.zeros((img.shape[0],img.shape[1],3))
    for contour in sorted_contours:
        cv.drawContours(blank,contour, -1, (255,0,0),1)
    # cv.imshow("All contours", blank)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    save_contours(sorted_contours, img.shape, path )

def save_contours(contours, img_size, path):
    i = 1
    for contour in contours:
        print(path)
        x,y,w,h = cv.boundingRect(contour)
        blank = np.zeros((img_size[0]+6, img_size[1]+6 ,1))
        blank = blank+255
        cv.drawContours(blank,contour,-1,(0,0,0),2, offset=(3,3))
        roi_img = blank[y:y+h+6,x:x+w+6]
        cv.imwrite("./Test Letters/letter"+path+str(i)+".jpg",roi_img)
        i+=1


def recognize_image(imgpath, sr):
    # img  = cv.imread(imgpath)
    upsampled = upscale_image(imgpath, sr)
    # img = cv.resize(img, None, fx = 3, fy = 3, interpolation = cv.INTER_CUBIC)
    # print(img.shape)
    # cv.imshow("Original",img)
    # cv.imshow("Upsampled",upsampled)
    # cv.waitKey(0)
    # cv.destroyAllWindows() 

    draw_contours(upsampled, imgpath)
    # draw_contours(img)


def enhance_image(imgpath):
    img  = cv.imread(imgpath)
    print(img.shape)
    cv.imshow("Original",img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    enhance = cv.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    cv.imshow("Enhance",enhance)
    cv.waitKey(0)
    cv.destroyAllWindows()

def upscale_image(imgpath, sr):
    img  = cv.imread("./Cropped License Plates/"+imgpath)
    upsampled = sr.upsample(img)
    return upsampled

sr = cv.dnn_superres.DnnSuperResImpl_create()
 
path = "./OCR/ESPCN_x3.pb"
sr.readModel(path)
sr.setModel("espcn",3)

image_list = os.listdir("./Cropped License Plates")

for image_path in image_list:
    recognize_image(image_path, sr)
    # enhance_image("./Cropped_New/"+image_path)
    # upscale_image("./Cropped_New/"+image_path, sr)

